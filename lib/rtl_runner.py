#!/usr/bin/env python3
"""Compare generated kernel programs on standalone AtlasCore and pinned npu_model.

Build the simulator with rtl_build_pinned.py. Both memory services use the same
pinned RTL; the matched service derives its configuration from npu_model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lib'))
import atlas_kernels_path  # noqa: E402,F401
from atlas_path_guard import checked  # noqa: E402

FAMILIES = ('projection_gemm', 'attention', 'rmsnorm', 'rope', 'epilogue',
            'rne_add', 'smol_fusion', 'resnet_pool', 'resnet_patch_gather')
DRAM_BASE = 0x90000000


def digest(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def save(path, value):
    checked(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run_case(args, family):
    from atlas_execution import Executor
    import atlas_backend
    import kernel_bench as bench
    from rtl_cases import load_case

    memory = getattr(args, 'memory', 'synthetic')
    target = args.out / memory / family
    target.mkdir(parents=True)
    started = time.perf_counter()
    row = dict(family=family, memory=memory, status='PENDING', rtl_revision=args.rtl_revision,
               model_revision='47b8599d8ffbe5ddd96059ffe2b5be2427a410f5',
               numerics='rtl', input_seed=7, initial_state_seed=42)
    try:
        row['simulator_sha256'] = digest(args.simulator)
        if not args.build_receipt:
            raise ValueError('running requires a build receipt')
        if args.program:
            source = checked(args.program).read_text()
            arena = checked(args.input).read_bytes()
            case = bench.Case(source, arena, args.output_offset, args.output_bytes, None, None)
            spec = dict(name=Path(args.program).stem, model='provided', shape=args.shape or 'provided program',
                        reduced=False, reduction_reason='provided generator assembly and staged input arena')
        else:
            spec, case = load_case(family)
        row.update(spec)
        row['case_metadata'] = case.meta
        if args.build_receipt:
            build = json.loads(checked(args.build_receipt).read_text())
            build_revision = build.get('rtl_revision')
            if build_revision != args.rtl_revision:
                raise ValueError('RTL revision does not match build provenance')
            if build.get('binary_sha256') != row['simulator_sha256'] or build.get('returncode') != 0:
                raise ValueError('simulator does not match successful build receipt')
            if memory == 'npu-model':
                from rtl_memory_config import inspect_model
                current = inspect_model(args.model_root)
                if build.get('memory_config') != current:
                    raise ValueError('model memory configuration does not match build provenance')
                row['memory_config'] = current
            row['build_receipt_sha256'] = digest(args.build_receipt)
            row['rtl_source_closure_sha256'] = hashlib.sha256(json.dumps(
                [(Path(item['copy']).name, item['sha256']) for item in build['sources']],
                separators=(',', ':')).encode()).hexdigest()
        executor = Executor(args.model_root)
        program, _ = executor.assemble(case.source)
        compiled = atlas_backend.cross_checked(
            case.source, program.assemble(), bin_dir=args.atlas_mlir_bin,
            min_tier='encode-only')
        row['backend'] = compiled.record()
        (target / 'program.S').write_text(case.source)
        (target / 'program.mlir').write_text(compiled.mlir)
        (target / 'program.bin').write_bytes(compiled.binary())
        (target / 'program.hex').write_text(''.join(f'{word:08x}\n' for word in compiled.words))
        (target / 'input.bin').write_bytes(case.arena)
        row.update(program_sha256=digest(target / 'program.bin'),
                   input_sha256=digest(target / 'input.bin'),
                   output_offset=case.output_offset, output_bytes=case.output_bytes,
                   arena_bytes=len(case.arena), program_words=len(compiled.words))
        phase = time.perf_counter()
        execution = executor.execute(case.source, case.arena, case.output_offset,
                                     case.output_bytes, requested_dma=case.meta.get('requested_dma'),
                                     max_instructions=50_000_000)
        row['functional_seconds'] = time.perf_counter() - phase
        row['functional_execution'] = execution.metadata()
        (target / 'model-output.bin').write_bytes(execution.output)
        row['golden_full_exact'] = None if case.golden is None else execution.output == case.golden
        row['golden_exact'] = row['golden_full_exact']
        if case.golden is not None and 'golden_logical_rows' in case.meta:
            m, _, n = spec['shape']
            logical = case.meta['golden_logical_rows']
            actual_words = bench.unpack_bf16_tiles(execution.output, m, n)
            golden_words = bench.unpack_bf16_tiles(case.golden, m, n)
            row['golden_exact'] = actual_words[:logical].tobytes() == golden_words[:logical].tobytes()
            row['golden_scope'] = f'first {logical} logical row(s)'
            row['golden_padding_different_words'] = int((actual_words[logical:] != golden_words[logical:]).sum())
            row['golden_padding_nonzero_different_words'] = int(((actual_words[logical:] != golden_words[logical:]) &
                (((actual_words[logical:] & 0x7fff) != 0) | ((golden_words[logical:] & 0x7fff) != 0))).sum())
        if row['golden_exact'] is False:
            raise RuntimeError('functional npu_model output differs from independent golden in its declared scope')
        phase = time.perf_counter()
        model_cycles, timed_arena = bench.timed_cycles(args.model_root, case.source,
                                                     case.arena, args.max_cycles)
        row['model_seconds'] = time.perf_counter() - phase
        row['npu_model_cycles'] = model_cycles
        row['timed_output_exact'] = timed_arena[case.output_offset:case.output_offset + case.output_bytes] == execution.output
        if not row['timed_output_exact']:
            raise RuntimeError('timed and functional npu_model outputs differ')
        image = target / 'case.image'
        image.write_text(f'dram {DRAM_BASE} {case.arena.hex()}\n')
        command = [str(args.simulator), str(target / 'program.hex'), str(args.max_cycles),
                   str(image), str(target / 'rtl-output.bin'),
                   str(DRAM_BASE + case.output_offset), str(case.output_bytes), '--memory', memory]
        if memory == 'npu-model':
            command += ['--dma-trace', str(target / 'dma.csv')]
        row['command'] = command
        save(target / 'result.json', row)
        phase = time.perf_counter()
        with (target / 'rtl.log').open('w') as stream:
            try:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        cwd=target, timeout=args.timeout, check=False)
            finally:
                row['rtl_wall_seconds'] = time.perf_counter() - phase
        row['rtl_returncode'] = result.returncode
        log = (target / 'rtl.log').read_text()
        match = re.search(r'ATLAS_KERNEL_RESULT (.*)', log)
        if match:
            fields = dict((key, int(value)) for key, value in re.findall(r'(\w+)=(\d+)', match[1]))
            row['rtl_observation'] = fields
            row['rtl_cycles'] = fields['execution_cycles']
            row['model_minus_rtl_cycles'] = model_cycles - fields['execution_cycles']
            row['instruction_count_exact'] = fields['observed_instructions'] == execution.instructions
        output_path = target / 'rtl-output.bin'
        if output_path.is_file():
            output = output_path.read_bytes()
            row['exact'] = output == execution.output
            row['rtl_output_sha256'] = digest(output_path)
            row['model_output_sha256'] = digest(target / 'model-output.bin')
            row['different_bytes'] = sum(a != b for a, b in zip(output, execution.output)) + abs(len(output) - len(execution.output))
        if result.returncode or not match:
            raise RuntimeError(f'RTL exited {result.returncode}; see rtl.log')
        if f'memory={memory} ' not in log:
            raise RuntimeError('simulator did not acknowledge requested memory mode')
        if memory == 'npu-model':
            from rtl_validate_memory import validate
            row['memory_validation'] = validate(args.model_root, target / 'dma.csv')
        if not row.get('exact'):
            raise RuntimeError('RTL output differs from npu_model')
        if not row.get('instruction_count_exact'):
            raise RuntimeError('RTL instruction count differs from functional model')
        row['status'] = 'PASS'
    except Exception as error:
        row.update(status='FAIL', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    row['total_wall_seconds'] = time.perf_counter() - started
    save(target / 'result.json', row)
    print(json.dumps({key: row.get(key) for key in ('family', 'status', 'rtl_cycles', 'npu_model_cycles', 'exact', 'error')}), flush=True)
    return row


def table(rows):
    lines = ['| Family | Shape | Exact | npu_model cycles | RTL cycles, matched memory | Model minus matched RTL | RTL cycles, synthetic memory | RTL commit |',
             '| --- | --- | :---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        shape = row.get('shape', 'PENDING')
        if isinstance(shape, list):
            shape = 'x'.join(map(str, shape))
        shape = str(shape) + (' (reduced)' if row.get('reduced', row.get('reduction_label') == 'reduced') else '')
        def field(key): return str(row[key]) if key in row else 'PENDING'
        exact = ('yes' if row['exact'] else 'no') if 'exact' in row else 'PENDING'
        lines.append(f"| {row['family']} | {shape} | {exact} | {field('npu_model_cycles')} | {field('matched_rtl_cycles')} | {field('model_minus_matched_rtl_cycles')} | {field('synthetic_rtl_cycles')} | {row['rtl_revision']} |")
    return '\n'.join(lines) + '\n'


def report(out, report_path):
    # Preserve the separately archived streamed-projection experiment.
    archived = ''
    if report_path.is_file():
        previous = report_path.read_text()
        heading = '## Opt-in streamed SmolVLA projection'
        if heading in previous:
            archived = previous[previous.index(heading):].split('## Whole-kit captured programs')[0].rstrip()
    rows = []
    for name in FAMILIES:
        pair = {}
        for memory in ('npu-model', 'synthetic'):
            path = checked(out / memory / name / 'result.json')
            if path.is_file():
                pair[memory] = json.loads(path.read_text())
        row = dict(next(iter(pair.values()))) if pair else dict(family=name, rtl_revision='PENDING')
        row['memories'] = pair
        row['status'] = 'PASS' if len(pair) == 2 and all(r['status'] == 'PASS' for r in pair.values()) else 'PENDING'
        if any(r['status'] == 'FAIL' for r in pair.values()):
            row['status'] = 'FAIL'
        # A combined row must describe the same experiment and simulator.
        identities = ('family', 'shape', 'rtl_revision', 'model_revision', 'simulator_sha256',
                      'program_sha256', 'input_sha256', 'model_output_sha256', 'npu_model_cycles', 'output_bytes')
        if len(pair) == 2 and any(pair['npu-model'].get(key) != pair['synthetic'].get(key) for key in identities):
            row.update(status='FAIL', error='memory runs have different experiment identities')
        row.pop('exact', None)
        if len(pair) == 2 and all('exact' in r for r in pair.values()):
            row['exact'] = all(r['exact'] for r in pair.values())
        if 'rtl_cycles' in pair.get('npu-model', {}):
            row['matched_rtl_cycles'] = pair['npu-model']['rtl_cycles']
            row['model_minus_matched_rtl_cycles'] = row['npu_model_cycles'] - row['matched_rtl_cycles']
        if 'rtl_cycles' in pair.get('synthetic', {}):
            row['synthetic_rtl_cycles'] = pair['synthetic']['rtl_cycles']
        rows.append(row)
    lines = ['# Per-family standalone AtlasCore RTL results', '',
             'The same generated instruction words and seeded DRAM inputs run on standalone AtlasCore Verilator and unmodified npu_model 47b8599 with numerics="rtl". Exact means every output byte equals the model output with both memory services, including padding.', '',
             table(rows).rstrip(), '',
             ('The matched-memory column reproduces the default npu_model curve backend and differs from npu_model by 1 cycle in every family because the model counts its startup cycle before first issue.'
              if all(row['status'] == 'PASS' and row.get('model_minus_matched_rtl_cycles') == 1 for row in rows) else
              'The matched-memory column compares the default npu_model curve backend; incomplete or failed families remain visible, and npu_model includes a startup cycle before first issue.'), '',
             'The synthetic-memory column uses a harness choice of request acceptance during 4 of every 11 cycles and FIFO responses after 43 cycles; it does not measure SoC performance.', '',
             'RTL cycles count first instruction issue through ECALL inclusive, excluding program loading and completion polling. Completion additionally requires ECALL status, no illegal PC, and no outstanding memory responses. Matched-memory request due cycles and response identities are replayed against the unchanged Python backend.', '',
             'Build identities and source hashes are retained in [rtl-provenance.json](rtl-provenance.json). Both columns use the same simulator binary and pinned generated RTL.', '',
             'Every shape reduction below preserves the family operation and uses the existing generator and reference staging where available. Full model shapes are not qualified by reduced runs.', '']
    completed = [row for row in rows if row['status'] != 'PENDING']
    if completed:
        identities = sorted({row.get('simulator_sha256', 'PENDING') for row in completed})
        lines += ['Simulator SHA-256: ' + ', '.join(f'`{value}`' for value in identities) + '.', '']
        from collections import Counter
        tiers = Counter(row.get('backend', {}).get('tier', 'PENDING') for row in completed)
        lines += ['Compiler tiers: ' + ', '.join(f'{count} {tier}' for tier, count in sorted(tiers.items())) + '. No program silently falls back to assembler encoding.', '']
    rope = next((row for row in rows if row['family'] == 'rope'), {})
    if 'golden_padding_different_words' in rope:
        lines += [f"Decode RoPE's independent golden is compared over its logical row. It differs in {rope['golden_padding_different_words']} padded BF16 words, including {rope['golden_padding_nonzero_different_words']} nonzero differences; the complete padded output still participates in RTL/model equality.", '']
    for row in rows:
        original = 'x'.join(map(str, row.get('original_model_shape', []))) or 'PENDING'
        scope = row.get('case_metadata', {}).get('scope', '')
        lines.append(f"- {row['family']}: source model shape {original}. {row.get('reduction_reason', row.get('note', 'PENDING'))}. {scope} Status: {row['status']}." + (f" Error: {row['error']}" if 'error' in row else ''))
    if archived:
        lines += ['', archived]
    report_path.write_text('\n'.join(lines) + '\n')
    whole_kit_report(report_path)
    save(out / 'summary.json', dict(status='PASS' if all(row['status'] == 'PASS' for row in rows) else 'INCOMPLETE_OR_FAILED', cases=rows))
    return rows



def whole_kit_report(report_path):
    """Render archived whole-kit evidence without executing any program."""
    evidence = json.loads(checked(ROOT / 'coverage/rtl-whole-kit.json').read_text())
    assert sum(row['programs'] for row in evidence['summary'].values()) == 1589
    assert all(row['exact'] == row['programs'] and row['mismatched'] == row['pending'] == 0
               for row in evidence['summary'].values())
    assert all(set(histogram) == {'0'} for histogram in evidence['cycle_delta_by_family'].values())
    assert sum(sum(h.values()) for h in evidence['cycle_delta_by_family'].values()) == 1589
    assert evidence['cycle_delta_definition'] == 'RTL - (Core - 1)'
    assert evidence['instruction_count_exact'] == 1589 and evidence['output_bytes'] == 364589056
    assert evidence['synthetic_runs'] == evidence['synthetic_exact'] == 50
    assert evidence['initial_state_seed'] == 42 and evidence['simulator_qualification_cases'] == 18
    assert evidence['synthetic_model_family_groups'] == 36 and evidence['verilator_version'] == '5.022'
    lines = ['## Whole-kit captured programs', '',
             'All 1,589 distinct TinyLlama, SmolVLA, and ResNet50 kit programs match every physical output byte (364,589,056 bytes, including padding and holes), instruction counts including ECALL, and matched-memory timing. There are 0 mismatches. Inputs are real captured values and linked weights; goldens use full unmodified npu_model Core execution with numerics="rtl" and seed 42. RTL = Core - 1 cycles throughout because Core includes one startup cycle.', '',
             '| Model | Programs | Exact | Cycle relation |',
             '| --- | ---: | ---: | --- |']
    for model, row in evidence['summary'].items():
        lines.append(f"| {model} | {row['programs']} | {row['exact']} | RTL = Core - 1 |")
    lines += ['', '| Model / family | Programs | Exact | Cycle relation |',
              '| --- | ---: | ---: | --- |']
    for family, histogram in evidence['cycle_delta_by_family'].items():
        count = sum(histogram.values())
        lines.append(f'| {family} | {count} | {count} | RTL = Core - 1 |')
    lines += ['',
              'The synthetic-memory sample is 50/50 full-byte exact across all 36 model/family groups; its scope is output exactness, without a matched-cycle claim. These are standalone program replays, not whole-model accuracy or SoC performance measurements.', '',
              f"RTL commit: `{evidence['rtl_revision']}`. Model commit: `{evidence['model_revision']}`. Verilator 5.022 simulator variants (four-thread, spin64, and single-thread) each qualified 18/18 nine-family cases across both memory modes. Executed matched-memory simulator SHA-256 identities:", '']
    lines += [f'- `{identity}`' for identity in evidence['simulator_sha256']]
    lines += ['',
              'The portable [evidence manifest](rtl-whole-kit.json) retains per-model counts, all family cycle histograms, and the source results.json SHA-256. Regenerate this section without simulation with `python3 lib/rtl_runner.py --whole-kit-report-only --out build/report --report coverage/RTL-RESULTS.md`.', '',
              'To replay each distinct program, use the committed [lib/rtl_runner.py](../lib/rtl_runner.py) custom-program interface in [RTL.md](../RTL.md#run-a-generated-program-and-input-arena): supply its generator assembly, complete captured DRAM arena beginning at 0x90000000 (real tensors and linked weights), output offset and physical byte extent, one runner family label, pinned model checkout, compiler binaries, simulator, and matching build receipt. Use a separate output directory per program, `--memory npu-model`, `--max-cycles 1000000000000`, and `--timeout 14400`; use `--memory synthetic` for the sample. The captured arenas and generated programs are external inputs, not distributed by the compact manifest. No new simulation was run for this documentation update.', '']
    text = report_path.read_text()
    heading = '## Whole-kit captured programs'
    section = '\n'.join(lines).rstrip() + '\n'
    if heading in text:
        start = text.index(heading)
        end = text.find('\n## ', start + len(heading))
        end = len(text) if end < 0 else end + 1
        report_path.write_text(text[:start] + section + ('\n' + text[end:] if end < len(text) else ''))
    else:
        report_path.write_text(text.rstrip() + '\n\n' + section)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator', type=Path)
    parser.add_argument('--memory', choices=('synthetic', 'npu-model', 'both'), default='both')
    parser.add_argument('--rtl-revision')
    parser.add_argument('--build-receipt', type=Path, help='hash-checked simulator build provenance')
    parser.add_argument('--program', type=Path, help='generator assembly, instead of a built-in family case')
    parser.add_argument('--input', type=Path, help='complete DRAM arena beginning at 0x90000000')
    parser.add_argument('--output-offset', type=int)
    parser.add_argument('--output-bytes', type=int)
    parser.add_argument('--shape', help='shape label for a provided program')
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--atlas-mlir-bin', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--family', choices=FAMILIES, action='append')
    parser.add_argument('--max-cycles', type=int, default=2_000_000)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--whole-kit-report-only', action='store_true', help='refresh archived whole-kit section without simulation')
    parser.add_argument('--no-report', action='store_true', help='write only per-case receipts for parallel workers')
    args = parser.parse_args()
    args.out = checked(args.out)
    if args.whole_kit_report_only:
        whole_kit_report(checked(args.report) if args.report else ROOT / 'coverage/RTL-RESULTS.md')
        return 0
    if args.no_report and args.report_only:
        parser.error('report-only cannot be combined with no-report')
    selected_rows = []
    if not args.report_only:
        if args.program and (not args.input or args.output_offset is None or args.output_bytes is None or
                             not args.family or len(args.family) != 1):
            parser.error('provided programs require input, output-offset, output-bytes, and exactly one family')
        if args.program and (args.output_bytes <= 0 or args.output_offset < 0):
            parser.error('provided programs require positive output-bytes and nonnegative output-offset')
        if not all((args.simulator, args.rtl_revision, args.model_root, args.atlas_mlir_bin, args.build_receipt)):
            parser.error('running requires simulator, RTL revision, model root, atlas-mlir bin, and build receipt')
        args.simulator, args.model_root, args.atlas_mlir_bin = map(checked, (args.simulator, args.model_root, args.atlas_mlir_bin))
        args.out.mkdir(parents=True, exist_ok=True)
        memories = ('npu-model', 'synthetic') if args.memory == 'both' else (args.memory,)
        for memory in memories:
            args.memory = memory
            for family in args.family or FAMILIES:
                if (args.out / memory / family).exists():
                    parser.error(f'refusing to overwrite case: {memory}/{family}')
                if __import__('shutil').disk_usage(args.out).free < 20 * 1024**3:
                    raise RuntimeError('less than 20 GiB free; stopping writes')
                selected_rows.append(run_case(args, family))
    rows = selected_rows if args.no_report else report(args.out, checked(args.report) if args.report else args.out / 'RTL-RESULTS.md')
    return int(any(row['status'] != 'PASS' for row in (rows if args.report_only else selected_rows))
               or any(row['status'] == 'FAIL' for row in rows))


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    raise SystemExit(main())
