"""Derive the C++ DMA curve table from the pinned Python memory backend.

Only explicit admitted paths are read. Package initializers are deliberately
not executed: they discover unrelated modules by traversing the model tree.
The five implementation modules below execute unchanged in namespace shells.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types

from atlas_path_guard import checked

MODEL_PIN = '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'
MODULES = (
    'npu_model.isa_types', 'npu_model.isa', 'npu_model.hardware.config',
    'npu_model.configs.hardware.default', 'npu_model.hardware.memory_backend',
)
PACKAGES = ('npu_model', 'npu_model.hardware', 'npu_model.configs', 'npu_model.configs.hardware')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pinned_bytes(root, relative):
    path = checked(root / relative)
    data = path.read_bytes()
    try:
        expected = subprocess.check_output(
            ['git', '-C', str(root), 'show', f'{MODEL_PIN}:{relative}'],
            stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('model dependency is absent from the pinned revision: ' + relative) from error
    if data != expected:
        raise ValueError('model dependency differs from the pinned revision: ' + relative)
    return data


@contextmanager
def model_backend(model_root):
    """Yield (backend module, fresh backend, provenance), restoring imports.

    Execute pinned source bytes directly, so Python never follows bytecode
    caches, package initializers, or an ambient alternate npu_model import.
    """
    root = checked(model_root)
    if not checked(root / '.git').is_dir():
        raise ValueError('model root must be a physical git clone containing the pin')
    source = {name: _pinned_bytes(root, name.replace('.', '/') + '.py') for name in MODULES}
    saved = {name: module for name, module in sys.modules.copy().items()
             if name == 'npu_model' or name.startswith('npu_model.')}
    for name in saved:
        del sys.modules[name]
    try:
        for name in PACKAGES:
            package = types.ModuleType(name)
            package.__path__ = []  # Never search the model filesystem.
            package.__package__ = name
            package.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
            sys.modules[name] = package
        for name in MODULES:
            relative = name.replace('.', '/') + '.py'
            filename = str(checked(root / relative))
            module = types.ModuleType(name)
            module.__file__ = filename
            module.__package__ = name.rpartition('.')[0]
            sys.modules[name] = module
            exec(compile(source[name], filename, 'exec'), module.__dict__)
        cfg = sys.modules['npu_model.configs.hardware.default'].DefaultHardwareConfig()
        if cfg.dma_memory_backend != 'curve':
            raise ValueError('C++ memory model requires the curve backend')
        curve_name = cfg.dma_memory_params['curves']
        if not isinstance(curve_name, str) or Path(curve_name).name != curve_name:
            raise ValueError('default curve must name one pinned memory curve file')
        relative_curve = 'npu_model/configs/memory_curves/' + curve_name
        curve_data = _pinned_bytes(root, relative_curve)
        # The backend receives already admitted JSON, avoiding its path search.
        cfg.dma_memory_params = dict(cfg.dma_memory_params, curves=json.loads(curve_data))
        module = sys.modules['npu_model.hardware.memory_backend']
        backend = module.make_memory_backend(cfg)
        if (not isinstance(backend, module.CurveMemoryBackend)
                or backend.latency_mode != 'unloaded' or backend.converge != 1.0
                or not backend.enforce_peak):
            raise ValueError('C++ memory model requires unloaded, converged, peak-paced curves')
        if cfg.dma_beat_bytes != 32 or backend.beat_bytes != cfg.dma_beat_bytes:
            raise ValueError('C++ TileLink interface requires 32-byte DMA beats')
        hashes = {name.replace('.', '/') + '.py': _sha(data) for name, data in source.items()}
        hashes[relative_curve] = _sha(curve_data)
        metadata = dict(model_revision=MODEL_PIN, source_sha256=dict(sorted(hashes.items())),
                        curve_sha256=_sha(curve_data))
        yield module, backend, metadata
    finally:
        for name in list(sys.modules):
            if name == 'npu_model' or name.startswith('npu_model.'):
                del sys.modules[name]
        sys.modules.update(saved)


def config(model_root):
    """Return deterministic (C++ header text, metadata) without writing files."""
    with model_backend(model_root) as (_, backend, metadata):
        rows = [dict(read_pct=pct, unloaded_latency_cycles=backend.latency_at(0.0, pct),
                     peak_beats_per_cycle=backend.curve_peak(pct)[0])
                for pct in sorted(backend._curves)]
        if (backend.queue_depth < 1 or backend.window_beats < 1 or not rows
                or any(not math.isfinite(row[key]) or row[key] <= 0
                       for row in rows for key in ('unloaded_latency_cycles', 'peak_beats_per_cycle'))):
            raise ValueError('invalid memory parameters for C++ table')
        params = dict(latency_mode=backend.latency_mode, beat_bytes=backend.beat_bytes,
                      queue_depth=backend.queue_depth, window_beats=backend.window_beats,
                      converge=backend.converge, min_latency=backend.min_latency,
                      enforce_peak=backend.enforce_peak, frequency_ghz=backend.frequency_ghz,
                      channels=backend.channels, rows=rows)
        lines = ['// Generated by rtl_memory_config.py from pinned model ' + MODEL_PIN,
                 '#pragma once',
                 'struct AtlasDefaultDmaCurve { int read_pct; double latency; double peak; };',
                 f'static constexpr int atlas_dma_queue_depth = {backend.queue_depth};',
                 f'static constexpr int atlas_dma_window_beats = {backend.window_beats};',
                 'static constexpr AtlasDefaultDmaCurve atlas_default_dma_curves[] = {']
        lines += ['  {' + str(row['read_pct']) + ', ' + repr(row['unloaded_latency_cycles'])
                  + ', ' + repr(row['peak_beats_per_cycle']) + '},' for row in rows]
        header = '\n'.join([*lines, '};', ''])
        metadata.update(schema='atlas-kernels.rtl-memory-config.v1', params=params,
                        header_sha256=_sha(header.encode()))
        return header, metadata


def inspect_model(model_root):
    """Return current pinned-source and generated-header provenance."""
    return config(model_root)[1]


def generate(model_root, output_header):
    """Write a header and adjacent .json sidecar, returning provenance."""
    output = checked(output_header)
    sidecar = checked(output.with_suffix('.json'))
    if output == sidecar:
        raise ValueError('header output must have an extension other than .json')
    header, metadata = config(model_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    checked(output).write_text(header)
    checked(sidecar).write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.model_root, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
