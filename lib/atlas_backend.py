"""Encode Atlas kernel programs through the Atlas compiler backend and prove the words equal the assembler's.

``compile(program_text)`` translates generator assembly into Atlas machine IR (``atlas_asm_mlir``), runs
``atlas-opt --verify-atlas-machine-stream`` and ``atlas-opt --verify-atlas-timing``, and encodes the words with
``atlas-emit``. ``cross_checked(program_text, assembler_words)`` also requires those words to equal npu_model's
assembler output, so a backend gap can never change a kernel binary silently.

Tiers (``TIERS``):

* ``verified``: the machine-stream verifier and the timing verifier both accept the program.
* ``stream-verified``: the machine-stream verifier accepts it and the timing verifier rejects it.
* ``encode-only``: no verifier accepted it; only ``atlas-emit`` vouches for the words (verification was not
  requested, or the machine-stream verifier rejected the program).

Every verifier diagnostic is kept. ``--verify-atlas-timing`` in atlas-mlir #26 stops at its first violation, so a
rejected program carries one timing diagnostic, and the checks after the DMA rules (issue spacing, resource
reservations, fixed-latency drain) have not run for it.

Builds and checks select the backend explicitly. ``python atlas_backend.py run --backend atlas-mlir --receipt R --
SCRIPT ARGS...`` runs one build or check script with every npu_model assembly (``npu_model.util.converter.
input_to_program``) cross-checked through the backend, and writes a receipt with the backend, the atlas-opt and
atlas-emit build identity, and each program's tier and diagnostics. ``--backend assembler`` records the same receipt
without the backend. Nothing here reads an environment variable to choose a backend or to fall back.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time

import atlas_asm_mlir

__all__ = ['TIERS', 'BACKENDS', 'BackendError', 'BackendUnavailable', 'Tools', 'Compiled', 'tools', 'compile',
           'cross_checked', 'Session', 'install', 'main']

TIERS = ('verified', 'stream-verified', 'encode-only')
BACKENDS = ('atlas-mlir', 'assembler')
RESTRICTED = ('hammer', 'vlsi')


class BackendError(RuntimeError):
    """The backend could not encode a program, or its words differ from the assembler's."""


class BackendUnavailable(BackendError):
    """atlas-opt or atlas-emit was not found."""


@dataclass(frozen=True)
class Tools:
    atlas_opt: str
    atlas_emit: str

    def identity(self):
        """Paths, SHA-256, version text, and (when recorded by CMake) the atlas-mlir source commit of the build."""
        out = {}
        for name, path in (('atlas-opt', self.atlas_opt), ('atlas-emit', self.atlas_emit)):
            version = subprocess.run([path, '--version'], capture_output=True, text=True)
            out[name] = dict(path=path, sha256=_file_sha(path),
                             version=' '.join((version.stdout or version.stderr).split())[:300])
        out['source'] = _build_source(Path(self.atlas_opt).absolute().parent.parent)
        return out


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_source(build):
    cache = build / 'CMakeCache.txt'
    if not cache.is_file():
        return None
    found = re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$', cache.read_text(errors='replace'), re.M)
    if not found:
        return None
    source = found.group(1).strip()
    record = dict(directory=source)
    if any(word in source.lower() or word in str(Path(source).resolve()).lower() for word in RESTRICTED):
        record['unavailable'] = 'restricted source path'
        return record
    if (Path(source) / '.git').exists():
        head = subprocess.run(['git', '-C', source, 'rev-parse', 'HEAD'], capture_output=True, text=True)
        if head.returncode == 0:
            # Filter names before accessing working files, including through Git.
            def tree_entries(tree, prefix=b''):
                # Prune restricted names before descending into their Git trees.
                listing = subprocess.run(['git', '-C', source, 'ls-tree', '-z', tree],
                                         capture_output=True, check=True).stdout
                for entry in listing.split(b'\0'):
                    if not entry:
                        continue
                    metadata, name = entry.split(b'\t', 1)
                    if any(word in os.fsdecode(name).lower() for word in RESTRICTED):
                        continue
                    mode, kind, blob = metadata.split()
                    relative = prefix + name
                    if kind == b'tree':
                        yield from tree_entries(blob.decode(), relative + b'/')
                    else:
                        yield metadata + b'\t' + relative

            listing = sorted(tree_entries('HEAD'), key=lambda entry: entry.split(b'\t', 1)[1])
            manifest, modified = hashlib.sha256(), False
            for entry in listing:
                if not entry:
                    continue
                metadata, raw_path = entry.split(b'\t', 1)
                relative = os.fsdecode(raw_path)
                if any(word in relative.lower() for word in RESTRICTED):
                    continue
                mode, kind, blob = metadata.split()
                if kind != b'blob':
                    continue
                path = Path(source) / relative
                if mode == b'120000' and path.is_symlink():
                    contents = os.fsencode(os.readlink(path))
                elif mode != b'120000' and not path.is_symlink() and not any(
                        word in str(path.resolve()).lower() for word in RESTRICTED) and path.is_file():
                    contents = path.read_bytes()
                else:
                    modified = True
                    contents = b''
                digest = hashlib.sha256(contents).hexdigest()
                manifest.update(raw_path + b'\0' + digest.encode() + b'\n')
                object_digest = hashlib.sha1(b'blob ' + str(len(contents)).encode() + b'\0' + contents).hexdigest()
                modified |= object_digest != blob.decode()
            record.update(commit=head.stdout.strip(), modified=modified,
                          tracked_source_sha256=manifest.hexdigest(),
                          provenance='observed CMake source checkout; binary digests identify the build')
    return record


def tools(bin_dir=None):
    """Locate atlas-opt and atlas-emit: ``bin_dir``, else ATLAS_MLIR_BIN or ATLAS_OOT_BIN_DIR, else PATH."""
    found = []
    for name in ('atlas-opt', 'atlas-emit'):
        path = None
        for directory in (bin_dir, os.environ.get('ATLAS_MLIR_BIN'), os.environ.get('ATLAS_OOT_BIN_DIR')):
            if directory:
                path = str(Path(directory) / name)
                break
        path = path or shutil.which(name)
        if not path or not Path(path).is_file():
            raise BackendUnavailable(f'{name} not found (looked in {path or "PATH"}); pass the atlas-mlir build bin '
                                     'directory with --atlas-mlir-bin or ATLAS_MLIR_BIN')
        found.append(str(Path(path).absolute()))
    return Tools(*found)


def _diagnostics(stderr):
    """Every ``error:`` diagnostic, as text with the source location kept."""
    found = [line.strip() for line in stderr.splitlines() if ': error:' in line]
    return tuple(found) if found else ((stderr.strip()[:400] or 'failed without a diagnostic'),)


@dataclass(frozen=True)
class Compiled:
    words: tuple
    tier: str                         # one of TIERS
    stream: str | None                # "pass", "reject", or None when not run
    stream_diagnostics: tuple
    timing: str | None                # "pass", "reject", or None when not run
    timing_diagnostics: tuple
    mlir: str

    def binary(self):
        return b''.join(word.to_bytes(4, 'little') for word in self.words)

    def record(self):
        return dict(tier=self.tier, words=len(self.words), words_sha256=hashlib.sha256(self.binary()).hexdigest(),
                    stream=self.stream, stream_diagnostics=list(self.stream_diagnostics), timing=self.timing,
                    timing_diagnostics=list(self.timing_diagnostics))


_cache = {}


def compile(program_text, *, verify=True, bin_dir=None, name='kernel.S', min_tier='stream-verified'):
    """Translate, verify (unless ``verify`` is false), and encode one program; returns a Compiled record."""
    if min_tier not in TIERS:
        raise BackendError(f'unknown minimum tier: {min_tier}')
    found = tools(bin_dir)
    key = (hashlib.sha256(program_text.encode()).hexdigest(), verify, found, min_tier)
    if key in _cache:
        return _cache[key]
    try:
        mlir = atlas_asm_mlir.translate(program_text, name=name)
    except atlas_asm_mlir.AsmError as error:
        raise BackendError(f'translation failed: {error}') from None
    with tempfile.TemporaryDirectory(prefix='atlas-backend-') as directory:
        path = Path(directory) / 'kernel.mlir'
        path.write_text(mlir)
        argv = {'emit': [found.atlas_emit, str(path)]}
        if verify:
            argv['stream'] = [found.atlas_opt, '--verify-atlas-machine-stream', str(path), '-o', os.devnull]
            argv['timing'] = [found.atlas_opt, '--verify-atlas-timing', str(path), '-o', os.devnull]
        running = {step: subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                   for step, command in argv.items()}
        done = {}
        for step, process in running.items():
            out, err = process.communicate()
            done[step] = (process.returncode, out, err)
    code, out, err = done['emit']
    if code:
        raise BackendError('atlas-emit failed: ' + '; '.join(_diagnostics(err)))
    words = tuple(int(word, 16) for word in out.split())
    stream = timing = None
    stream_diagnostics = timing_diagnostics = ()
    if verify:
        stream = 'pass' if done['stream'][0] == 0 else 'reject'
        stream_diagnostics = () if stream == 'pass' else _diagnostics(done['stream'][2])
        timing = 'pass' if done['timing'][0] == 0 else 'reject'
        timing_diagnostics = () if timing == 'pass' else _diagnostics(done['timing'][2])
    tier = 'encode-only' if stream != 'pass' else ('verified' if timing == 'pass' else 'stream-verified')
    result = Compiled(words, tier, stream, stream_diagnostics, timing, timing_diagnostics, mlir)
    if TIERS.index(tier) > TIERS.index(min_tier):
        raise BackendError(f'program tier {tier} is below required {min_tier}: ' +
                           '; '.join(stream_diagnostics + timing_diagnostics))
    _cache[key] = result
    return result


def cross_checked(program_text, assembler_words, **options):
    """Backend words for program_text, required to equal npu_model's assembler words; returns a Compiled record."""
    result = compile(program_text, **options)
    expected = tuple(int(word) for word in assembler_words)
    if result.words != expected:
        index = next((i for i, (a, b) in enumerate(zip(result.words, expected)) if a != b),
                     min(len(result.words), len(expected)))
        raise BackendError(f'atlas-emit words differ from the assembler at word {index} '
                           f'({len(result.words)} vs {len(expected)} words)')
    return result


def rule(diagnostic):
    """A diagnostic without its location and with integers normalized, for histograms."""
    text = diagnostic.split(': error:', 1)[-1].strip()
    return re.sub(r'\d+', 'N', text)


class Session:
    """Cross-checks assembly in the wrapped process (including inherited hooks after fork) and spools one record per program."""

    def __init__(self, backend, spool, *, verify=True, bin_dir=None):
        if backend not in BACKENDS:
            raise BackendError(f'backend must be one of {", ".join(BACKENDS)}, not {backend!r}')
        self.backend, self.spool, self.verify, self.bin_dir = backend, Path(spool), verify, bin_dir
        self.tools = tools(bin_dir) if backend == 'atlas-mlir' else None
        self._seen = set()

    def encode(self, text, assembler_words):
        words = [int(word) for word in assembler_words]
        digest = hashlib.sha256(text.encode()).hexdigest()
        key = (os.getpid(), digest)
        if key in self._seen:
            return
        self._seen.add(key)
        record = dict(source_sha256=digest, assembler_words=len(words),
                      assembler_words_sha256=hashlib.sha256(b''.join(w.to_bytes(4, 'little') for w in words)).hexdigest())
        if self.backend == 'atlas-mlir':
            compiled = cross_checked(text, words, verify=self.verify, bin_dir=self.bin_dir, min_tier='encode-only')
            record.update(compiled.record(), words_equal_assembler=True)
        else:
            record.update(tier='assembler')
        with self.spool.open('a') as spool:
            spool.write(json.dumps(record, sort_keys=True) + '\n')

    def programs(self):
        merged = {}
        if self.spool.is_file():
            for line in self.spool.read_text().splitlines():
                record = json.loads(line)
                merged.setdefault(record['source_sha256'], record)
        return [merged[key] for key in sorted(merged)]


def _wrap(module, session):
    original = module.input_to_program
    if getattr(original, '_atlas_backend_session', None) is session:
        return

    def input_to_program(source):
        if hasattr(source, 'read'):
            text = source.read()
            program = original(io.StringIO(text))
        else:
            lines = list(source)
            text = ''.join(line if line.endswith('\n') else line + '\n' for line in lines)
            program = original(lines)
        session.encode(text, program.assemble())
        return program

    input_to_program._atlas_backend_session = session
    input_to_program.__wrapped__ = original
    input_to_program.__doc__ = original.__doc__
    module.input_to_program = input_to_program


class _ConverterHook(importlib.abc.MetaPathFinder):
    """Wraps npu_model.util.converter.input_to_program whenever that module is imported, from any npu_model root."""
    NAME = 'npu_model.util.converter'

    def __init__(self, session):
        self.session = session

    def find_spec(self, fullname, path, target=None):
        if fullname != self.NAME:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _ConverterHook) or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        exec_module, session = spec.loader.exec_module, self.session

        def exec_and_wrap(module):
            exec_module(module)
            _wrap(module, session)
        spec.loader.exec_module = exec_and_wrap
        return spec


def install(session):
    """Route every later (and any current) import of npu_model's assembler through session.encode."""
    sys.meta_path.insert(0, _ConverterHook(session))
    module = sys.modules.get(_ConverterHook.NAME)
    if module is not None:
        _wrap(module, session)


def _npu_model_identity():
    module = sys.modules.get(_ConverterHook.NAME)
    if module is None:
        return None
    root = Path(module.__file__).absolute().parents[2]
    record = dict(converter=str(Path(module.__file__).absolute()), root=str(root))
    if (root / '.git').exists() and not any(word in str(root).lower() for word in RESTRICTED):
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        record['commit'] = head.stdout.strip() if head.returncode == 0 else None
    return record


def summarize(programs):
    tiers = Counter(record['tier'] for record in programs)
    timing = Counter(rule(d) for record in programs for d in record.get('timing_diagnostics', ()))
    stream = Counter(rule(d) for record in programs for d in record.get('stream_diagnostics', ()))
    return dict(programs=len(programs), words=sum(record['assembler_words'] for record in programs),
                tiers=dict(sorted(tiers.items())), stream_rules=dict(stream.most_common()),
                timing_rules=dict(timing.most_common()))


def _run(args):
    receipt = Path(args.receipt).absolute()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    spool = receipt.with_name(receipt.name + '.programs.jsonl')
    spool.unlink(missing_ok=True)
    session = Session(args.backend, spool, verify=not args.no_verify, bin_dir=args.atlas_mlir_bin)
    data = dict(schema='atlas-kernels.backend-run.v1', backend=args.backend,
                verification='none' if args.no_verify else 'machine-stream and timing',
                command=[args.script, *args.arguments], cwd=os.getcwd(), library=str(Path(__file__).absolute()),
                library_sha256=_file_sha(__file__), translator_sha256=_file_sha(atlas_asm_mlir.__file__),
                tools=session.tools.identity() if session.tools else None, started=time.time())
    install(session)
    script = str(Path(args.script).absolute())
    sys.argv = [script, *args.arguments]
    # As `python SCRIPT` would: the script's directory first, and this library directory only if the script adds it.
    here = Path(__file__).absolute().parent
    sys.path[:] = [str(Path(script).parent)] + [p for p in sys.path[1:] if Path(p or '.').absolute() != here]
    code, error = 0, None
    try:
        runpy.run_path(script, run_name='__main__')
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if stop.code is not None and not isinstance(stop.code, int):
            error = str(stop.code)
    except BaseException as failure:  # noqa: BLE001 - recorded, then re-raised as a nonzero exit
        code, error = 1, f'{type(failure).__name__}: {failure}'
        import traceback
        traceback.print_exc()
    programs = session.programs()
    policy_errors = []
    if len(programs) < args.expect_min_programs:
        policy_errors.append(f'encoded {len(programs)} distinct programs; expected at least {args.expect_min_programs}')
    if args.backend == 'atlas-mlir':
        rejected = Counter(p['tier'] for p in programs if TIERS.index(p['tier']) > TIERS.index(args.min_tier))
        if rejected:
            policy_errors.append(f'programs below required {args.min_tier}: {dict(rejected)}')
    if policy_errors:
        code = code or 1
        error = '; '.join(([error] if error else []) + policy_errors)
        print(f'atlas_backend: {error}', file=sys.stderr)
    data.update(expect_min_programs=args.expect_min_programs, min_tier=args.min_tier,
                policy_errors=policy_errors, finished=time.time(), exit_code=code, error=error, npu_model=_npu_model_identity(),
                summary=summarize(programs), programs=programs)
    receipt.write_text(json.dumps(data, indent=1) + '\n')
    spool.unlink(missing_ok=True)
    print(f'atlas_backend: {args.backend} {json.dumps(data["summary"]["tiers"])} over {len(programs)} programs; '
          f'receipt {receipt}', file=sys.stderr)
    return code


def _compile(args):
    text = Path(args.program).read_text()
    result = compile(text, verify=not args.no_verify, bin_dir=args.atlas_mlir_bin, name=Path(args.program).name, min_tier=args.min_tier)
    record = dict(program=args.program, tools=tools(args.atlas_mlir_bin).identity(), **result.record())
    if args.check_assembler:
        from npu_model.util.converter import input_to_program
        cross_checked(text, input_to_program(io.StringIO(text)).assemble(), verify=not args.no_verify,
                      bin_dir=args.atlas_mlir_bin, min_tier=args.min_tier)
        record['words_equal_assembler'] = True
    if args.output:
        Path(args.output).write_bytes(result.binary())
    if args.mlir:
        Path(args.mlir).write_text(result.mlir)
    print(json.dumps(record, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='run a build or check script with every assembly cross-checked')
    run.add_argument('--backend', choices=BACKENDS, required=True)
    run.add_argument('--receipt', required=True)
    run.add_argument('--atlas-mlir-bin', help='atlas-mlir build bin directory holding atlas-opt and atlas-emit')
    run.add_argument('--no-verify', action='store_true', help='encode only: skip both verifiers')
    run.add_argument('--expect-min-programs', type=int, default=1, metavar='N',
                     help='require at least N distinct program texts (default: 1)')
    run.add_argument('--min-tier', choices=TIERS, default='stream-verified',
                     help='minimum accepted compiler tier (default: stream-verified)')
    run.add_argument('script')
    run.add_argument('arguments', nargs=argparse.REMAINDER)
    one = sub.add_parser('compile', help='translate, verify, and encode one assembly file')
    one.add_argument('program')
    one.add_argument('--atlas-mlir-bin')
    one.add_argument('--no-verify', action='store_true')
    one.add_argument('--min-tier', choices=TIERS, default='stream-verified')
    one.add_argument('--check-assembler', action='store_true', help="also require npu_model's assembler words")
    one.add_argument('-o', '--output', help='write the little-endian program binary here')
    one.add_argument('--mlir', help='write the machine-IR module here')
    args = parser.parse_args(argv)
    if args.command == 'run' and args.expect_min_programs < 1:
        parser.error('--expect-min-programs must be at least 1')
    if args.no_verify and args.min_tier != 'encode-only' and (args.command == 'compile' or args.backend == 'atlas-mlir'):
        parser.error('--no-verify requires --min-tier encode-only')
    if args.command == 'run' and args.arguments[:1] == ['--']:
        args.arguments = args.arguments[1:]
    return _run(args) if args.command == 'run' else _compile(args)


if __name__ == '__main__':
    import atlas_backend          # this directory is sys.path[0]; run the module object, not __main__
    sys.exit(atlas_backend.main())
