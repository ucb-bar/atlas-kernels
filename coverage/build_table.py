"""Build the model coverage table from the exported op inventories and mapping.yaml.

Every fixture row must match a mapping rule; an unmapped record fails the build. Writes COVERAGE.md
and coverage.json next to this script (or into --out). Requires PyYAML.

    python3 coverage/build_table.py [--check-cites ATLAS_MLIR MERLIN ATLAS_KERNELS]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fnmatch import fnmatchcase
import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).absolute().parent
MODELS = {'tinyllama': 'TL', 'smolvla': 'SV'}
COLUMNS = ('TL', 'SV', 'GM', 'pi0')


def patterns(value):
    return value if isinstance(value, list) else [value]


def shape_matches(pattern, shape):
    """A pattern is one shape (items int, '*', or a list of ints) or a list of such shapes."""
    if pattern and all(isinstance(item, list) for item in pattern) and any(
            all(not isinstance(x, list) for x in item) for item in pattern):
        return any(shape_matches(item, shape) for item in pattern)
    return len(pattern) == len(shape) and all(
        item == '*' or (shape_item in item if isinstance(item, list) else shape_item == item)
        for item, shape_item in zip(pattern, shape))


def matches(rule, record):
    output = record['outputs'][0] if record['outputs'] else [[], '']
    for key, value in rule.items():
        if key == 'out':
            if not shape_matches(value, output[0]):
                return False
        elif key == 'dtype':
            if output[1] not in patterns(value):
                return False
        elif key in ('op', 'module', 'source', 'phase'):
            if not any(fnmatchcase(record[key], pattern) for pattern in patterns(value)):
                return False
        else:
            raise ValueError(f'unknown match field {key}')
    return True


def place(model, inventory, rules, kernels):
    placed, unmapped, hits = [], [], Counter()
    for record in inventory['ops']:
        index = next((i for i, rule in enumerate(rules) if matches(rule['match'], record)), None)
        if index is None:
            unmapped.append(record)
            continue
        where = rules[index]['where']
        kind, _, target = where.partition(':')
        if kind not in ('atlas', 'host', 'build') or kind == 'atlas' and target not in kernels:
            raise ValueError(f'{model} rule {index}: bad placement {where}')
        hits[index] += record['count']
        placed.append(dict(record, where=where, kind=kind, rule=index))
    return placed, unmapped, hits


def derived_launches(placed):
    """Device work derived from the export, in the units the kit counts launches."""
    count = Counter()
    for row in placed:
        if row['kind'] != 'atlas':
            continue
        kernel = row['where'].split(':', 1)[1]
        if row['op'] in ('aten.linear.default', 'aten.conv2d.default', 'aten.conv2d.padding'):
            count['projection_calls'] += row['count']
        elif row['op'] in ('aten.softmax.int', 'aten.scaled_dot_product_attention.default'):
            count['attention'] += row['count']
        elif kernel == 'rmsnorm' and row['op'] == 'aten.rsqrt.default':
            count['rmsnorm'] += row['count']
        elif kernel == 'smol_fusion' and row['op'].startswith('aten.gelu'):
            count['smol_fusion_calls'] += row['count']
    return dict(count)


def shape_text(record):
    inputs = ', '.join('x'.join(map(str, shape)) or 'scalar' for shape, _ in record['inputs'][:2])
    output = record['outputs'][0] if record['outputs'] else [[], '']
    return f"{inputs or '-'} -> {'x'.join(map(str, output[0])) or 'scalar'} {output[1]}"


def gflop(value):
    return f'{value / 1e9:,.3f}'


def check_cites(mapping, roots):
    missing = []
    names = dict(zip(('atlas-mlir', 'merlin', 'atlas-kernels'), roots))
    for rules in mapping['rules'].values():
        for rule in rules:
            for cite in rule.get('cite', []):
                repo, _, rest = cite.partition(':')
                if repo in names and not (Path(names[repo]) / rest.split(':')[0]).is_file():
                    missing.append(cite)
            kind, _, target = rule['where'].partition(':')
            repo, _, path = target.partition('/')
            if kind in ('host', 'build') and not (Path(names[repo]) / path).is_file():
                missing.append(rule['where'])
    for family in mapping['kernels']:
        manifest = Path(names['atlas-kernels']) / 'kernels' / family / 'kernel.yaml'
        if not manifest.is_file() or yaml.safe_load(manifest.read_text()).get('family') != family:
            missing.append(f'atlas-kernels:kernels/{family}/kernel.yaml family {family}')
    return sorted(set(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--check-cites', nargs=3, metavar=('ATLAS_MLIR', 'MERLIN', 'ATLAS_KERNELS'))
    args = parser.parse_args()
    mapping = yaml.safe_load((HERE / 'mapping.yaml').read_text())
    kernels = mapping['kernels']
    summary, sections, failures = {'schema': 'atlas.coverage.summary.v1', 'models': {}}, [], []
    usage = defaultdict(set)
    for model, column in MODELS.items():
        inventory = json.loads((HERE / 'fixture' / f'{model}.json').read_text())
        placed, unmapped, hits = place(model, inventory, mapping['rules'][model], kernels)
        for record in unmapped:
            failures.append(f"{model}: unmapped {record['phase']} {record['module']!r} {record['source']} "
                            f"{record['op']} {shape_text(record)} x{record['count']}")
        unused = [i for i in range(len(mapping['rules'][model])) if not hits[i]]
        if unused:
            failures.append(f'{model}: rules match nothing: {unused}')
        expected = mapping['builds'][model]['launches']
        derived = derived_launches(placed)
        for key, value in derived.items():
            if expected.get(key) != value:
                failures.append(f'{model}: derived {key} {value} differs from kit {expected.get(key)}')
        flops = Counter()
        for row in placed:
            flops[row['kind']] += row['count'] * row['flops_each']
            if row['kind'] == 'atlas':
                usage[row['where'].split(':', 1)[1]].add(column)
        runtime = flops['atlas'] + flops['host']
        by_where = defaultdict(lambda: Counter())
        for row in placed:
            by_where[row['where']]['ops'] += row['count']
            by_where[row['where']]['flops'] += row['count'] * row['flops_each']
        summary['models'][model] = {
            'inventory_nodes': inventory['node_count'], 'aggregated_records': len(inventory['ops']),
            'capture_crosscheck': inventory.get('capture_crosscheck', {}).get('status', 'NOT_RUN'),
            'derived_launches': derived, 'kit_launches': expected,
            'flops': dict(flops), 'atlas_share_of_runtime_flops': flops['atlas'] / runtime if runtime else 0,
            'by_where': {key: dict(value) for key, value in sorted(by_where.items())},
            'builds': {name: text for name, text in mapping['builds'][model].items() if name != 'launches'},
        }
        lines = [f'## {model} ({column})', '']
        lines += [f"Export: `{inventory['method']}` in Python {inventory['environment']['python'].split()[0]}, torch "
                  f"{inventory['environment']['torch']}, transformers {inventory['environment']['transformers']}"
                  + (f", lerobot {inventory['environment']['lerobot']}" if 'lerobot' in inventory['environment'] else '')
                  + f"; {inventory['node_count']:,} ATen records in {len(inventory['ops'])} aggregated rows. "
                  f"Merlin capture cross-check: {summary['models'][model]['capture_crosscheck']} "
                  f"({inventory.get('capture_crosscheck', {}).get('exported_calls')} exported calls against "
                  f"{inventory.get('capture_crosscheck', {}).get('captured_calls')} captured).", '']
        lines += [f"Default: {mapping['builds'][model]['default']}", '', f"Fast: {mapping['builds'][model]['fast']}", '']
        lines += ['| device work | derived from export | qualified build |', '| --- | ---: | ---: |']
        lines += [f"| {key} | {f'{derived[key]:,}' if key in derived else 'not derived'} | {expected[key]:,} |"
                  for key in expected]
        lines += ['', f"FLOPs: Atlas {gflop(flops['atlas'])} GFLOP, host {gflop(flops['host'])} GFLOP "
                  f"({100 * flops['atlas'] / runtime:.4f}% of runtime FLOPs on Atlas); build-time constants "
                  f"{gflop(flops['build'])} GFLOP, not run on the target.", '']
        lines += ['| where | op instances | GFLOP |', '| --- | ---: | ---: |']
        lines += [f"| `{key}` | {value['ops']:,} | {gflop(value['flops'])} |" for key, value in sorted(by_where.items())]
        lines += ['', '| phase | module | op | shape | count | where | GFLOP |', '| --- | --- | --- | --- | ---: | --- | ---: |']
        layout = defaultdict(Counter)
        for row in placed:
            if row['flops_each'] == 0:
                layout[(row['phase'], row['where'])][row['op'].replace('aten.', '')] += row['count']
                continue
            lines.append(f"| {row['phase']} | `{row['module'] or '(top)'}` | `{row['op'].replace('aten.', '')}` | "
                         f"{shape_text(row)} | {row['count']:,} | `{row['where']}` | "
                         f"{gflop(row['count'] * row['flops_each'])} |")
        lines += ['', 'Zero-FLOP records (views, casts, copies, and index construction):', '',
                  '| phase | where | ops |', '| --- | --- | --- |']
        lines += [f"| {phase} | `{where}` | {', '.join(f'{op} x{n}' for op, n in ops.items())} |"
                  for (phase, where), ops in layout.items()]
        sections.append('\n'.join(lines))
    if args.check_cites:
        missing = check_cites(mapping, args.check_cites)
        failures += [f'missing cited file: {item}' for item in missing]
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        sys.exit(1)
    table = ['| kernel | op | ' + ' | '.join(COLUMNS) + ' |', '| --- | --- |' + ' :-: |' * len(COLUMNS)]
    for kernel, op in kernels.items():
        table.append(f'| `{kernel}` | {op} | ' + ' | '.join('x' if c in usage[kernel] else '' for c in COLUMNS) + ' |')
    summary['coverage'] = {kernel: sorted(usage[kernel]) for kernel in kernels}
    text = [(HERE / 'COVERAGE.head.md').read_text().rstrip('\n'), '', *table, '', *sections, '']
    (args.out / 'COVERAGE.md').write_text('\n'.join(text))
    (args.out / 'coverage.json').write_text(json.dumps(summary, indent=1) + '\n')
    for model, value in summary['models'].items():
        print(model, value['derived_launches'], f"atlas share {100 * value['atlas_share_of_runtime_flops']:.4f}%")


if __name__ == '__main__':
    main()
