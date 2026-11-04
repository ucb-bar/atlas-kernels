"""Write a portable, compact manifest from a successful pinned build receipt."""
import argparse
import hashlib
import json
from pathlib import Path

from atlas_path_guard import checked


def compact(receipt):
    build = json.loads(checked(receipt).read_text())
    if build.get('returncode') != 0:
        raise ValueError('build receipt does not record success')
    generated = {Path(path).name: value for path, value in build['generated_sources'].items()}
    if len(generated) != len(build['generated_sources']):
        raise ValueError('generated source basenames are not unique')
    sources = sorted((r['revision'], r['path'], r['git_blob'], r['sha256']) for r in build['sources'])
    source_digest = hashlib.sha256(json.dumps(sources, separators=(',', ':')).encode()).hexdigest()
    return dict(schema='atlas-kernels.rtl-provenance.v1', rtl_revision=build['rtl_revision'],
                fp_revisions=build['submodules'], scala_source_count=build['source_count'],
                scala_source_closure_sha256=source_digest,
                dependency_checkout_revision=build['dependency_checkout_revision'],
                dependency_origin_status=build['dependency_origin_status'],
                dependency_assembly_sha256=build['dependency_assembly_sha256'],
                dependency_sha256={Path(p).name: value for p, value in build['tools'].items()},
                java_version=build['java_version'], verilator_version=build['verilator_version'],
                verilator_sha256=build['verilator_sha256'],
                build_script_sha256=build['build_script_sha256'], emitter_sha256=build['emitter_sha256'],
                harness_sha256=build['harness_sha256'], memory_generator_sha256=build['memory_generator_sha256'],
                memory_config=build['memory_config'], generated_verilog_sha256=dict(sorted(generated.items())),
                simulator_sha256=build['binary_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    checked(args.out).write_text(json.dumps(compact(args.build_receipt), indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
