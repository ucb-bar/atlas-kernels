"""Shared make operations for the nine representative family cases.

Generator CLIs remain available for arbitrary admitted geometries. This interface
uses the explicitly bounded, seeded cases documented in coverage/RTL-RESULTS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
import atlas_kernels_path  # noqa: E402,F401
from atlas_path_guard import checked  # noqa: E402
from rtl_cases import FAMILIES  # noqa: E402

RTL_PIN = '0079c0541111197741a231c002e3843fa6f545b2'
SIMULATOR_SHA256 = 'a568f3820eeb9dae1601a948353d040670bb622e63e2a72c14bd4b24ebe35c86'


def save(path, record):
    checked(path).write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def generate(family, out, seed):
    from family_data import admit_output_root, build_case
    from kernel_gen import write
    out = admit_output_root(out)
    spec, case = build_case(family, seed=seed)
    target = checked(out / family)
    for name in ('program.S', 'program.bin', 'program.json'):
        checked(target / name)
    return write(SimpleNamespace(out=target, name='program'), 'program', case.source,
                 dict(family=family, case=spec, output_offset=case.output_offset,
                      output_bytes=case.output_bytes))


def validated_case(family, out, seed):
    """Bind persisted data to its actual producer before executing any program."""
    import numpy as np
    from family_data import build_case
    spec, case = build_case(family, seed=seed)
    target = checked(out / family)
    manifest = json.loads(checked(target / 'manifest.json').read_text())
    if manifest['family'] != family or manifest['seed'] != seed:
        raise ValueError('data manifest selects a different family or seed')
    for name, artifact in manifest['artifacts'].items():
        if Path(name).name != name:
            raise ValueError('artifact must be a filename within the family output')
        raw = checked(target / name).read_bytes()
        if len(raw) != artifact['bytes'] or hashlib.sha256(raw).hexdigest() != artifact['sha256']:
            raise ValueError('data manifest artifact differs: ' + name)
    for name, digest in manifest['dependencies'].items():
        path = checked(ROOT / name)
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('data producer source differs: ' + name)
    if checked(target / 'program.S').read_text() != case.source:
        raise ValueError('generated source differs from the selected data producer')
    if checked(target / 'input_arena.bin').read_bytes() != case.arena:
        raise ValueError('staged input differs from the selected data producer')
    reference = np.load(checked(target / 'reference.npy'), allow_pickle=False)
    if not np.array_equal(reference, case.reference):
        raise ValueError('FP32 reference differs from the selected reference producer')
    golden = checked(target / 'golden.bin')
    if (case.golden is None and golden.exists() or
            case.golden is not None and golden.read_bytes() != case.golden):
        raise ValueError('independent golden differs from the selected reference producer')
    return spec, case


def compare_golden(spec, case, output):
    if case.golden is None:
        return dict(status='PENDING', reason='No independent Atlas arithmetic golden for this case')
    if len(output) != case.output_bytes or len(case.golden) != case.output_bytes:
        raise AssertionError('output or golden extent differs')
    if 'golden_logical_rows' in case.meta:
        import kernel_bench as bench
        m, _, n = spec['shape']
        rows = case.meta['golden_logical_rows']
        actual = bench.unpack_bf16_tiles(output, m, n)
        expected = bench.unpack_bf16_tiles(case.golden, m, n)
        if actual[:rows].tobytes() != expected[:rows].tobytes():
            raise AssertionError('logical rows differ from independent golden')
        differences = actual[rows:] != expected[rows:]
        nonzero = differences & (((actual[rows:] & 0x7fff) != 0) | ((expected[rows:] & 0x7fff) != 0))
        if nonzero.any():
            raise AssertionError('padded rows differ beyond documented BF16 zero signs')
        return dict(status='PASS', scope=f'first {rows} logical row(s)',
                    padding_different_words=int(differences.sum()), padding_nonzero_different_words=0)
    if output != case.golden:
        raise AssertionError('output differs from independent golden')
    return dict(status='PASS', scope='complete output including padding')


def test_family(family, out, seed, model_root, atlas_mlir_bin):
    import numpy as np
    import atlas_backend
    import kernel_bench as bench
    from atlas_execution import Executor
    target = checked(out / family)
    receipt = dict(family=family, seed=seed, status='FAIL')
    try:
        spec, case = validated_case(family, out, seed)
        executor = Executor(model_root)
        program, _ = executor.assemble(case.source)
        compiled = atlas_backend.cross_checked(case.source, program.assemble(), bin_dir=atlas_mlir_bin,
                                               min_tier='stream-verified')
        if checked(target / 'program.bin').read_bytes() != compiled.binary():
            raise AssertionError('persisted binary differs from compiler and assembler')
        receipt.update(backend=compiled.record(), tools=atlas_backend.tools(atlas_mlir_bin).identity(),
                       words_equal_assembler=True)
        checked(target / 'program.mlir').write_text(compiled.mlir)
        execution = executor.execute(case.source, case.arena, case.output_offset, case.output_bytes,
                                     requested_dma=case.meta.get('requested_dma'), max_instructions=50_000_000)
        receipt['independent_golden'] = compare_golden(spec, case, execution.output)
        actual = case.decode(execution.output)
        if not np.isfinite(actual).all():
            raise AssertionError('decoded output contains nonfinite values')
        receipt.update(status='PASS', case=spec, guards='PASS', execution=execution.metadata(),
                       fidelity=bench.fidelity(actual, case.reference),
                       fidelity_scope='FP32 comparison report; no fidelity acceptance threshold is claimed',
                       data_manifest_sha256=hashlib.sha256(checked(target / 'manifest.json').read_bytes()).hexdigest())
        checked(target / 'model-output.bin').write_bytes(execution.output)
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save(target / 'test-result.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('family', 'status', 'independent_golden')}, sort_keys=True))
    return receipt


def rtl(args):
    import rtl_runner
    if args.seed != 7:
        raise ValueError('pinned RTL representative cases require SEED=7')
    validated_case(args.family, args.out, args.seed)
    simulator, build_receipt = checked(args.simulator), checked(args.build_receipt)
    if hashlib.sha256(simulator.read_bytes()).hexdigest() != SIMULATOR_SHA256:
        raise ValueError('simulator differs from the pinned RTL-RESULTS.md artifact')
    build = json.loads(build_receipt.read_text())
    if (build.get('rtl_revision') != RTL_PIN or build.get('binary_sha256') != SIMULATOR_SHA256
            or build.get('returncode') != 0):
        raise ValueError('build receipt differs from the successful pinned RTL artifact')
    # The pinned runner stages the same seed-7 case; it retains full model/RTL
    # equality, both memory services, timing, and provenance validation.
    sys.argv = ['rtl_runner', '--family', args.family, '--out', str(args.out / args.family / 'rtl'),
                '--model-root', str(args.model_root), '--atlas-mlir-bin', str(args.atlas_mlir_bin),
                '--simulator', str(simulator), '--build-receipt', str(build_receipt),
                '--rtl-revision', RTL_PIN, '--memory', args.memory, '--timeout', str(args.timeout)]
    return rtl_runner.main()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('gen', 'data', 'test', 'rtl'))
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--out', type=checked, required=True, help='output root; artifacts go under FAMILY/')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--model-root')
    parser.add_argument('--atlas-mlir-bin')
    parser.add_argument('--simulator')
    parser.add_argument('--build-receipt')
    parser.add_argument('--memory', choices=('both', 'npu-model', 'synthetic'), default='both')
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    if args.operation in ('test', 'rtl'):
        if not args.model_root or not args.atlas_mlir_bin:
            parser.error('test and rtl require NPU_MODEL and ATLAS_MLIR_BIN')
        args.model_root, args.atlas_mlir_bin = map(checked, (args.model_root, args.atlas_mlir_bin))
    if args.operation == 'rtl' and (not args.simulator or not args.build_receipt):
        parser.error('rtl requires explicitly provisioned RTL_SIMULATOR and RTL_BUILD_RECEIPT; see RTL.md')
    if args.operation == 'gen':
        generate(args.family, args.out, args.seed)
    elif args.operation == 'data':
        from family_data import produce_family
        record = produce_family(args.family, args.out, seed=args.seed)
        print(json.dumps(dict(family=args.family, status=record['status'],
                              independent_arithmetic=record['independent_arithmetic'],
                              manifest=str(args.out / args.family / 'manifest.json')), sort_keys=True))
    elif args.operation == 'test':
        test_family(args.family, args.out, args.seed, args.model_root, args.atlas_mlir_bin)
    else:
        return rtl(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
