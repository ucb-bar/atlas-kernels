#!/usr/bin/env python3
"""Extract pinned Scala blobs, compile directly, and build standalone AtlasCore."""
import argparse, hashlib, json, os, re, shutil, subprocess, time
from pathlib import Path
ROOT = Path(__file__).absolute().parent
PIN = '0079c0541111197741a231c002e3843fa6f545b2'
ASSEMBLY_SHA256 = '25c454b8f2179b13b4ada30c24cf6dab3c8db0517bf2913ffbe64b15c68961d1'

def safe(p):
    p = Path(p).absolute()
    if any((re.search('hammer|vlsi', x, re.I) for x in p.parts)):
        raise ValueError('forbidden path')
    for a in [*reversed(p.parents), p]:
        if a.is_symlink():
            raise ValueError('symlink refused: ' + str(a))
    return p

def sha(p):
    return hashlib.sha256(safe(p).read_bytes()).hexdigest()

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(safe(repo)), *args])

def entries(repo, tree):
    for record in git(repo, 'ls-tree', '-z', tree).split(b'\x00'):
        if not record:
            continue
        meta, name = record.split(b'\t', 1)
        name = name.decode()
        if re.search('hammer|vlsi', name, re.I):
            continue
        mode, kind, oid = meta.decode().split()
        yield (mode, kind, oid, name)

def extract(repo, rev, prefix, destination, records):
    tree = git(repo, 'rev-parse', rev + ':' + prefix).decode().strip()

    def descend(tree, rel):
        for mode, kind, oid, name in entries(repo, tree):
            path = rel / name
            if mode == '120000':
                raise RuntimeError('symlink source entry refused')
            if kind not in ['tree', 'blob']:
                raise RuntimeError('unsupported source tree entry')
            if kind == 'tree':
                descend(oid, path)
            elif kind == 'blob' and name.endswith('.scala'):
                data = git(repo, 'cat-file', 'blob', oid)
                dest = safe(destination / path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                records.append({'repo': str(repo), 'revision': rev, 'path': str(Path(prefix) / path), 'git_blob': oid, 'sha256': hashlib.sha256(data).hexdigest(), 'copy': str(dest)})
    descend(tree, Path())

def execute(command, log, env):
    start = time.monotonic()
    with safe(log).open('w') as f:
        r = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
    print(json.dumps({'log': str(log), 'returncode': r.returncode, 'seconds': round(time.monotonic() - start, 3)}), flush=True)
    if r.returncode:
        raise RuntimeError('command failed: ' + str(log))
    return round(time.monotonic() - start, 3)

def main():
    total_start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rtl-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True, help='pinned npu_model checkout for the generated memory configuration')
    parser.add_argument('--classpath', type=Path, required=True, help='dependency assembly with the pinned SHA-256')
    parser.add_argument('--classpath-revision', required=True, help='observed Chipyard dependency checkout commit; not proof of historical cache origin')
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reflect', type=Path, required=True)
    parser.add_argument('--plugin', type=Path, required=True)
    parser.add_argument('--firtool', type=Path, required=True)
    parser.add_argument('--verilator', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--resume', action='store_true', help='reuse an existing output from this pinned build')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.classpath_revision):
        parser.error('--classpath-revision must be a full lowercase Git commit')
    if not 1 <= args.jobs <= 12:
        parser.error('--jobs must be between 1 and 12')
    OUT = safe(args.out)
    RTL = safe(args.rtl_root)
    if OUT.exists() and (not args.resume):
        parser.error('output exists; use --resume explicitly')
    space_root = OUT
    while not space_root.exists():
        space_root = space_root.parent
    if shutil.disk_usage(safe(space_root)).free < 20 * 1024 ** 3:
        raise RuntimeError('below 20 GiB free')
    safe(OUT).mkdir(parents=True, exist_ok=True)
    script_snapshot = safe(OUT / 'build-script-executed.py')
    shutil.copyfile(safe(Path(__file__)), script_snapshot)
    from rtl_memory_config import generate
    memory_config = generate(safe(args.model_root), OUT / 'dma_config.h')
    records = []
    submodules = {name: oid for mode, kind, oid, name in entries(RTL, PIN + ':dependencies') if mode == '160000'}
    extract(RTL, PIN, 'src/main/scala', OUT / 'sources/atlas', records)
    for name in ['fpex', 'sp26-fp-units']:
        extract(RTL / 'dependencies' / name, submodules[name], 'src/main/scala', OUT / 'sources' / name, records)
    if len(records) != 131:
        raise RuntimeError('expected 131 selected Scala sources; got ' + str(len(records)))
    compiler = safe(args.compiler)
    library = safe(args.library)
    reflect = safe(args.reflect)
    plugin = safe(args.plugin)
    assembly = safe(args.classpath)
    if sha(assembly) != ASSEMBLY_SHA256:
        raise ValueError('dependency assembly SHA-256 does not match the pinned artifact')
    firtool = safe(args.firtool)
    verilator = safe(args.verilator)
    for path in [compiler, library, reflect, plugin, assembly, firtool, ROOT / 'EmitPinnedAtlasCore.scala', ROOT / 'atlas_kernel_replay.cpp']:
        safe(path)
    provenance = {'schema': 'atlas.pinned_core_build.v1', 'build_script_sha256': sha(script_snapshot), 'build_script_snapshot': str(script_snapshot), 'verilator_sha256': sha(verilator), 'verilator_version': subprocess.check_output([str(verilator), '--version'], text=True).strip(), 'java_version': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True).strip(), 'rtl_revision': PIN, 'submodules': submodules, 'dependency_assembly_sha256': ASSEMBLY_SHA256, 'dependency_checkout_revision': args.classpath_revision, 'dependency_origin_status': 'checkout revision observed; historical cache build and cleanliness unattested', 'memory_config': memory_config, 'memory_generator_sha256': sha(ROOT / 'rtl_memory_config.py'), 'source_count': len(records), 'sources': records, 'tools': {str(p): sha(p) for p in [compiler, library, reflect, plugin, assembly, firtool]}, 'emitter_sha256': sha(ROOT / 'EmitPinnedAtlasCore.scala'), 'harness_sha256': sha(ROOT / 'atlas_kernel_replay.cpp')}
    safe(OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    classes = safe(OUT / 'classes')
    classes.mkdir(exist_ok=True)
    java_home = safe(OUT / 'java-home')
    java_home.mkdir(exist_ok=True)
    java_tmp = safe(OUT / 'tmp')
    java_tmp.mkdir(exist_ok=True)
    java_props = ['-Duser.home=' + str(java_home), '-Djava.io.tmpdir=' + str(java_tmp), '-Duser.dir=' + str(OUT)]
    rtl = safe(OUT / 'rtl')
    rtl.mkdir(exist_ok=True)
    env = dict(os.environ, CHISEL_FIRTOOL_PATH=str(firtool.parent), VERILATOR_ROOT=str(safe(verilator.parent.parent / 'share/verilator')))
    compiler_cp = os.pathsep.join(map(str, [compiler, library, reflect]))
    command = ['java', *java_props, '-Xmx8g', '-cp', compiler_cp, 'scala.tools.nsc.Main', '-classpath', str(assembly), '-Xplugin:' + str(plugin), '-d', str(classes), *[r['copy'] for r in records], str(ROOT / 'EmitPinnedAtlasCore.scala')]
    scalac_seconds = execute(command, OUT / 'scalac.log', env)
    elaborate_seconds = execute(['java', *java_props, '-Xmx8g', '-cp', str(classes) + os.pathsep + str(assembly), 'atlas.tile.EmitPinnedAtlasCore', str(rtl)], OUT / 'elaborate.log', env)
    sources = []
    for directory, dirs, files in os.walk(rtl):
        dirs[:] = [d for d in dirs if not re.search('hammer|vlsi', d, re.I)]
        for name in files:
            if re.search('hammer|vlsi', name, re.I):
                continue
            if name.endswith(('.sv', '.v')):
                sources.append(safe(Path(directory) / name))
    command = [str(verilator), '--cc', '--exe', '--build', '--top-module', 'AtlasCore', '--prefix', 'VAtlasCore', '--Mdir', str(OUT / 'obj'), '--trace', '--trace-depth', '3', '--assert', '--public-flat-rw', '--output-split', '20000', '--output-split-cfuncs', '500', '-Wno-fatal', '-j', str(args.jobs), '-CFLAGS', '-std=c++17 -I' + str(OUT), '-MAKEFLAGS', 'CXX=/usr/bin/g++ LINK=/usr/bin/g++ AR=/usr/bin/ar', *map(str, sorted(sources)), str(ROOT / 'atlas_kernel_replay.cpp')]
    provenance.update(generated_sources={str(p): sha(p) for p in sources}, command=command)
    safe(OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    start = time.monotonic()
    verilator_seconds = execute(command, OUT / 'build.log', env)
    provenance.update(binary_sha256=sha(OUT / 'obj/VAtlasCore'), seconds=round(time.monotonic() - start, 3), scalac_seconds=scalac_seconds, elaborate_seconds=elaborate_seconds, verilator_seconds=verilator_seconds, total_seconds=round(time.monotonic() - total_start, 3), returncode=0)
    safe(OUT / 'build-result.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
