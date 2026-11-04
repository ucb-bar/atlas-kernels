"""Deterministic input and independent reference artifacts for all kernel families.

This module stages the existing physical cases and runs their host input and
reference producers. It never executes a device program. FP32 fidelity outputs
and independent arithmetic golden bytes have separate provenance and status.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
import atlas_kernels_path  # noqa: E402
from atlas_path_guard import checked, lexical_path  # noqa: E402
import kernel_bench as bench  # noqa: E402
import rtl_cases  # noqa: E402

FAMILIES = rtl_cases.FAMILIES
SEED = rtl_cases.SEED


def admit_output_root(value):
    """Admit a physical output path, requiring ignored storage inside the repo."""
    path = checked(value)
    if path.is_relative_to(ROOT) and not {'out', 'build'}.intersection(path.relative_to(ROOT).parts):
        raise ValueError('generated data inside the repository must be under ignored out/ or build/')
    return path


def build_case(family, *, seed=SEED, reduced=True):
    """Return the same coherent ``(spec, Case)`` used by generation and tests.

    The default case is exactly the existing RTL staging case. An alternate
    seed reruns that family's real producer with the identical case geometry.
    It changes synthetic values, not generator policy or program selection.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    import torch
    torch.set_num_threads(1)
    spec, case = rtl_cases.load_case(family, reduced=reduced)
    if seed != SEED:
        rng = np.random.default_rng(seed)
        if family in ('resnet_pool', 'resnet_patch_gather'):
            _, case = (rtl_cases._pool if family == 'resnet_pool' else rtl_cases._patch_gather)(rng)
        else:
            case = bench.load_reference(family).build(spec, rng)
        # These fields describe case selection rather than producer values.
        case.meta.update({key: spec[key] for key in (
            'original_model_shape', 'shape_label', 'reduced', 'reduction',
            'reduction_label', 'reduction_reason')})
        if family == 'rope' and spec.get('logical_rows') == 1:
            case.meta.update(golden_logical_rows=1,
                             golden_padding_note='Independent comparison covers logical row 0; '
                             'padded rows may differ in BF16 zero signs.')
    spec['seed'] = case.meta['seed'] = seed
    return spec, case


def _input_tensors(family, spec, case):
    """Recreate named logical inputs with the real producer's RNG stream.

    The staged arena is also persisted verbatim, binding these readable tensors
    to the exact byte layout consumed by the generated program.
    """
    rng = np.random.default_rng(spec['seed'])
    if family == 'resnet_pool':
        channels, height, width = spec['shape']
        return {'image_fp8': np.frombuffer(case.arena[:height * width * channels], dtype=np.uint8)
                .reshape(height, width, channels).copy()}
    if family == 'resnet_patch_gather':
        channels, height, width = spec['shape']
        source = np.frombuffer(case.arena[64:64 + 64 * channels], dtype=np.uint8).copy()
        return {'source_bytes': source, 'image_bytes': source[:height * width * channels]
                .reshape(height, width, channels).copy()}
    if family == 'attention':
        rows, keys, heads, kv_heads = spec['shape']
        dim = spec.get('head_dim', 64)
        tensors = {name: bench.bf16_words(rng.standard_normal(shape, dtype=np.float32))
                   for name, shape in (('q_bf16', (heads, rows, dim)),
                                       ('k_bf16', (kv_heads, keys, dim)),
                                       ('v_bf16', (kv_heads, keys, dim)))}
        tensors['allowed_mask'] = bench.load_reference(family).allowed(spec)
        return tensors
    if family == 'rmsnorm':
        rows, width = spec['shape']
        logical = spec.get('logical_rows', rows)
        hidden = np.zeros((rows, width), dtype=np.uint16)
        hidden[:logical] = bench.bf16_words(rng.standard_normal((logical, width), dtype=np.float32))
        gamma = bench.bf16_words(1 + 0.1 * rng.standard_normal(width, dtype=np.float32))
        return {'hidden_bf16': hidden, 'gamma_bf16': gamma}
    a, w, (a_codes, _), (w_codes, _) = bench.projection_operands(spec, rng)
    tensors = {'activations_fp32': a, 'weights_fp32': w,
               'activations_fp8': a_codes, 'weights_fp8': w_codes}
    rows, _, columns = spec['shape']
    if family == 'rope':
        cosine, sine = bench.load_reference(family).tables(rows)
        tensors.update(cosine_bf16=cosine, sine_bf16=sine)
    elif family == 'rne_add':
        tensors['residual_bf16'] = bench.bf16_words(rng.standard_normal((rows, columns), dtype=np.float32))
    elif family == 'epilogue' and spec['step'] == 'mul':
        import torch
        tensors['gate_bf16'] = bench.bf16_words(torch.nn.functional.silu(
            torch.from_numpy(rng.standard_normal((rows, columns), dtype=np.float32))).numpy())
    elif family == 'smol_fusion':
        tensors['bias_bf16'] = bench.bf16_words(0.1 * rng.standard_normal(columns, dtype=np.float32))
    return tensors


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _dependencies(family):
    """Hash repository source imports using admitted explicit paths, no walk."""
    from atlas_program import _TRAFFIC_REUSE_MODULES

    queue = [Path(__file__), ROOT / 'atlas_kernels_path.py',
             ROOT / 'lib' / 'rtl_cases.py', ROOT / 'lib' / 'kernel_bench.py']
    queue.extend(atlas_kernels_path.StandaloneFallback.FILES.values())
    resolved = {}

    def local_module(module):
        module = module.split('.')[0]
        if module not in resolved:
            filename = lexical_path(module + '.py')
            resolved[module] = None
            for directory in atlas_kernels_path.DIRECTORIES:
                candidate = checked(directory / filename)
                if candidate.is_file():
                    resolved[module] = candidate
                    break
        return resolved[module]

    # These generators are selected through a runtime registry rather than a
    # Python import statement. Bind every declared recipe deterministically.
    for module in sorted(set(_TRAFFIC_REUSE_MODULES.values())):
        dependency = local_module(module)
        if dependency is None:
            raise FileNotFoundError(f'missing registered recipe: {module}')
        queue.append(dependency)
    if family not in ('resnet_pool', 'resnet_patch_gather'):
        queue.append(ROOT / 'kernels' / family / 'reference.py')
    result = {}
    while queue:
        path = checked(queue.pop())
        name = str(path.relative_to(ROOT))
        if name in result:
            continue
        raw = path.read_bytes()
        result[name] = _sha(raw)
        for node in ast.walk(ast.parse(raw, filename=str(path))):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                       else [])
            for module in modules:
                dependency = local_module(module)
                if dependency is not None:
                    queue.append(dependency)
    return dict(sorted(result.items()))


def _policy(family, spec, case):
    common = {'device_execution': False, 'fidelity_reference': 'host reference on producer inputs',
              'reference_device': 'CPU', 'reference_threads': 1,
              'input_generation': 'NumPy default_rng (PCG64), synthetic values, never captured model values'}
    if family == 'resnet_pool':
        return dict(common, arithmetic='nonnegative finite FP8 byte maximum; 3x3, stride 2, zero padding 1')
    if family == 'resnet_patch_gather':
        return dict(common, arithmetic='exact byte movement; 3x3 im2col (ky,kx,channel), zero boundary/padded rows')
    common.update(bf16='round to nearest, ties to even',
                  fp8='E4M3, power-of-two tensor scale; exponent-zero decode flushes to +0',
                  scales={key: value for key, value in case.meta.items() if 'scale' in key or 'exponent' in key})
    if family in ('projection_gemm', 'rope', 'rne_add', 'epilogue', 'smol_fusion'):
        common['projection'] = 'ascending-K FP8 products, BF16 rounding after each MAC, exact power-of-two BF16 rescale'
    operations = {
        'projection_gemm': 'projection output in BF16',
        'rope': '64-channel rotate-half; RNE(RNE(x*cos) + RNE(rotate_half(x)*sin)); positions 0..M-1',
        'rne_add': 'RNE_bf16(projection + BF16 residual)',
        'attention': 'FP32 PyTorch grouped-query scaled softmax attention fidelity; no independent Atlas arithmetic oracle',
        'rmsnorm': 'FP32 PyTorch RMSNorm fidelity, epsilon 1e-5; no independent Atlas arithmetic oracle',
        'epilogue': f"FP32 PyTorch {spec.get('step', 'epilogue')} fidelity; no independent Atlas arithmetic oracle",
        'smol_fusion': 'FP32 PyTorch bias + GELU(tanh) fidelity; no independent Atlas arithmetic oracle',
    }
    common['operation'] = operations[family]
    return common


def produce_family(family, output_root, *, seed=SEED, reduced=True):
    """Write ``output_root/family`` artifacts and return their manifest.

    Existing generator companions (program.bin/program.json) are preserved.
    Callers should use an ignored out/ or build/ directory inside this repo.
    """
    output_root = admit_output_root(output_root)
    spec, case = build_case(family, seed=seed, reduced=reduced)
    out = checked(output_root / family)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    def record(name, raw):
        path = checked(out / name)
        path.write_bytes(raw)
        artifacts[name] = {'sha256': _sha(raw), 'bytes': len(raw)}

    def tensor(name, value):
        import io
        value = np.asarray(value)
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        record(name + '.npy', stream.getvalue())
        artifacts[name + '.npy'].update(dtype=value.dtype.str, shape=list(value.shape))

    record('program.S', case.source.encode())
    record('input_arena.bin', case.arena)
    tensor('reference', case.reference)
    for name, value in sorted(_input_tensors(family, spec, case).items()):
        tensor(name, value)
    if case.golden is not None:
        record('golden.bin', case.golden)
        tensor('golden_decoded', case.decode(case.golden))
    else:
        # Reusing an output directory must never leave an obsolete oracle.
        for name in ('golden.bin', 'golden_decoded.npy'):
            checked(out / name).unlink(missing_ok=True)
    import torch
    manifest = {
        'schema_version': 1, 'family': family, 'status': 'PASS',
        'producer': 'lib/family_data.py:produce_family', 'seed': int(seed),
        'case': spec, 'physical_shape': spec['shape'],
        'logical_shape': [spec.get('logical_rows', spec['shape'][0]), *spec['shape'][1:]]
            if family not in ('resnet_pool', 'resnet_patch_gather') else spec['shape'],
        'output_offset': case.output_offset, 'output_bytes': case.output_bytes,
        'reference': {'status': 'PASS', 'kind': 'FP32 fidelity' if family not in ('resnet_pool', 'resnet_patch_gather')
                      else 'independent host byte algorithm', 'artifact': 'reference.npy'},
        'independent_arithmetic': {'status': 'PASS', 'producer': case.golden_name, 'artifact': 'golden.bin'}
            if case.golden is not None else {'status': 'PENDING', 'artifact': None,
                'reason': 'No independent Atlas-arithmetic reference exists for this family; device output is never substituted.'},
        'arithmetic_policy': _policy(family, spec, case), 'case_metadata': case.meta,
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__},
        'dependency_scope': 'Conservative static repository Python import closure, selected family reference, dynamic recipe registry, and standalone fallbacks; package versions recorded separately.',
        'dependencies': _dependencies(family), 'artifacts': artifacts,
        'device_validation': {'status': 'PENDING', 'reason': 'Data production does not execute device programs.'},
    }
    checked(out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def produce_families(output_root, *, families=None, seed=SEED, reduced=True):
    """Run every selected family's actual input and host reference producer."""
    selected = list(FAMILIES if families is None else families)
    unknown = [family for family in selected if family not in FAMILIES]
    if unknown:
        raise ValueError(f'unknown data families: {unknown}')
    if not reduced and any(family.startswith('resnet_') for family in selected):
        raise ValueError('ResNet adapters expose reduced execution-test cases only')
    return [produce_family(family, output_root, seed=seed, reduced=reduced)
            for family in selected]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'out' / 'data')
    parser.add_argument('--family', action='append', choices=FAMILIES)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--published', action='store_true', help='use first published benchmark shape (seven families only)')
    args = parser.parse_args(argv)
    rows = produce_families(args.out, families=args.family, seed=args.seed, reduced=not args.published)
    print(json.dumps([{'family': row['family'], 'status': row['status'],
                       'independent_arithmetic': row['independent_arithmetic']['status']} for row in rows], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
