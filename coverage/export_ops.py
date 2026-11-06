"""Export every PyTorch op instance of a focus model at the shapes of its Merlin capture.

`torch.export.export(strict=False)` traces one complete inference with fake tensors (no arithmetic runs):
TinyLlama is a 128-token prefill plus greedy decode steps, and SmolVLA is two-image vision, the VLM prefix, and
ten denoising steps. Every ATen call, including calls inside higher-order-op subgraphs, becomes one record with
its op, innermost module path, model source line, phase, tensor input and output shapes and dtypes, and FLOPs.

`--fixture` receives the aggregated rows (identical records merged across layers and steps, with the ordered
projection and attention call indices) that build_table.py and the atlas-mlir coverage test read. `--out`
receives the full node list, ordered calls, attention cores, and graph text for audit; it is not committed.
See coverage/COVERAGE.md for the interpreters, model revisions, and inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

SCHEMA = 'atlas.coverage.inventory.v2'
SKIP = {'aten._assert_tensor_metadata.default', 'aten._assert_scalar.default', 'aten.lift_fresh_copy.default',
        'aten.detach_.default', 'aten.alias.default', 'aten.dropout.default', '<built-in function getitem>'}
PROJECTION = ('aten.linear.default', 'aten.conv2d.default', 'aten.conv2d.padding')
ATTENTION = ('aten.softmax.int', 'aten.scaled_dot_product_attention.default')
LAYOUT = {'view', 'reshape', 'transpose', 'permute', 'unsqueeze', 'squeeze', 'expand', 'expand_as', 'slice',
          'select', 'contiguous', 'clone', 'cat', 'stack', 'split', 'unbind', 'flatten', 'to', 'copy_', 'empty_like',
          'arange', 'ones', 'full', 'zeros', 'linspace', 'index_put_', 'embedding', 'repeat_interleave'}


# Pins from Merlin examples/atlas/comparisons (capture_tinyllama_native.py, capture_smolvla.py,
# evaluate_smolvla_retained_prefix.py) and the observation manifest.
TINY_REVISION = 'fe8a4ea1ffedaf415f4da2f062534de366a451e6'
TINY_WEIGHTS = '6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933'
TINY_MODELING = '46d313ec0f1116bcc1bb7f2379467ab249c140415c64fb758e2d5915cd8b89a5'
TINY_CORPUS = '573e1d244efc95bd7bb4ee3df0735bfdd688d3abf7f70a849ac2061c6de8a4d9'
TINY_TOKENS = '24e1d064a71f5aa75b560cb0881cab1fac5312cb3c534d8e6c0eaa0528f4231a'
SMOL_REVISION = 'c83c3163b8ca9b7e67c509fffd9121e66cb96205'
SMOL_WEIGHTS = '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb'
SMOL_BACKBONE_CONFIG = 'ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17'
SMOL_OBSERVATION = '167d7bbec48362a3dbb177b9afc97ec6ebfe50855e07f3858e40369de87f36e9'
SMOL_FRAME = {'dataset': 'lerobot/svla_so100_pickplace', 'dataset_revision': '3d6d687a25cdf1565cdf24550814f72d999a861d',
              'split': 'train', 'episode_index': 0, 'frame_index': 0, 'noise_seed': 0}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path, expected=None):
    """SHA-256 and size of a local input; the local path itself is never recorded in the fixture."""
    path = Path(path).resolve()
    if any(re.search('hammer|vlsi', part, re.I) for part in path.parts):
        raise ValueError(f'restricted path: {path}')
    value = sha256(path)
    if expected and value != expected:
        raise ValueError(f'SHA-256 differs for {path.name}: {value}')
    return {'file': path.name, 'sha256': value, 'bytes': path.stat().st_size}


def dimension(value):
    """Static sizes as ints; a data-dependent (unbacked) size keeps its symbol name."""
    try:
        return int(value)
    except Exception:
        return str(value)


def tensors(value):
    """Flatten a node argument or meta value into [shape, dtype] pairs for every tensor it holds."""
    import torch
    from torch.fx import Node
    if isinstance(value, Node):
        value = value.meta.get('val')
    if isinstance(value, torch.Tensor):
        return [[[dimension(d) for d in value.shape], str(value.dtype).replace('torch.', '')]]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in tensors(child)]
    return []


def flops(op, inputs, outputs):
    """Multiply-adds count twice; other compute ops count one per output (or reduced input) element."""
    numel = lambda shape: math.prod(shape) if all(type(d) is int for d in shape) else 0
    if not outputs:
        return 0
    out = outputs[0][0]
    if op == 'aten.linear.default':
        k = inputs[0][0][-1]
        return 2 * numel(out) * k + bias_flops(op, inputs, outputs)
    if op in ('aten.matmul.default', 'aten.bmm.default', 'aten.mm.default'):
        return 2 * numel(out) * inputs[0][0][-1]
    if op.startswith('aten.conv2d'):
        weight = inputs[1][0]
        return 2 * numel(out) * numel(weight[1:]) + bias_flops(op, inputs, outputs)
    if op == 'aten.scaled_dot_product_attention.default':
        q, k = inputs[0][0], inputs[1][0]
        return 4 * numel(q[:-1]) * k[-2] * q[-1] + 5 * numel(q[:-1]) * k[-2]
    # Export can retain functorch batching wrappers around mask construction. They change
    # tensor layout, not values; other Python callables still use the compute fallback.
    if op in {f'{module}.{name}' for module in ('torch._C._functorch', 'torch._functorch.predispatch')
              for name in ('_add_batch_dim', '_remove_batch_dim')}:
        return 0
    if not op.startswith('aten.'):
        return numel(out)
    name = op.split('.')[1]
    if name in LAYOUT:
        return 0
    if name in ('mean', 'sum', 'min', 'max', 'argmax', 'cumsum'):
        return numel(inputs[0][0])
    if name in ('softmax', 'layer_norm'):
        return 5 * numel(out)
    return numel(out)


def bias_flops(op, inputs, outputs):
    """Linear and Conv2d bias broadcasts add once per output element, when a bias tensor exists."""
    if op not in PROJECTION or len(inputs) < 3 or not outputs:
        return 0
    shape = outputs[0][0]
    return math.prod(shape) if all(type(d) is int for d in shape) else 0


def source(node):
    """Innermost model-source frame (file:function:line) outside torch, from the export stack trace."""
    frames = re.findall(r'File "([^"]+)", line (\d+), in (\S+)', node.meta.get('stack_trace') or '')
    frames = [frame for frame in frames if '/torch/' not in frame[0] and not frame[0].endswith('export_ops.py')]
    if not frames:
        return ''
    path, line, function = frames[-1]
    return f'{Path(path).name}:{function}:{line}'


def walk(graph_module, strip, records, inherited=None):
    """Append one record per ATen call in execution order, recursing into higher-order-op subgraphs."""
    import torch
    for node in graph_module.graph.nodes:
        if node.op != 'call_function':
            continue
        stack = node.meta.get('nn_module_stack') or inherited or {}
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            for arg in node.args:
                if getattr(arg, 'op', None) == 'get_attr':
                    walk(getattr(graph_module, arg.target), strip, records, stack)
            continue
        op = str(node.target)
        if op in SKIP:
            continue
        if not isinstance(node.target, torch._ops.OpOverload):
            # Builtin reprs may contain process addresses and no dotted operator name.
            name = getattr(node.target, '__name__', None)
            module_name = getattr(node.target, '__module__', '')
            if name:
                op = f'{module_name}.{name}' if module_name else name
        module, call = '', 0
        if stack:
            key, (path, _) = list(stack.items())[-1]
            module = path[len(strip):] if path.startswith(strip) else '' if path == strip[:-1] else path
            call = int(key.rsplit('@', 1)[1]) if '@' in key else 0
        inputs = [item for arg in (*node.args, *node.kwargs.values()) for item in tensors(arg)]
        outputs = tensors(node.meta.get('val'))
        records.append({'index': len(records), 'op': op, 'module': module, 'call': call, 'source': source(node),
                        'inputs': inputs, 'outputs': outputs, 'flops': flops(op, inputs, outputs)})
        if op == 'aten.rms_norm.default':
            eps = node.kwargs.get('eps', node.args[3] if len(node.args) > 3 else None)
            # PyTorch's omitted epsilon is the input dtype's machine epsilon.
            records[-1]['eps'] = float(eps) if eps is not None else torch.finfo(getattr(torch, inputs[0][1])).eps
        bias = bias_flops(op, inputs, outputs)
        if bias:
            records[-1]['bias_flops'] = bias


def template(module):
    match = re.search(r'\.layers\.(\d+)\.', module + '.')
    return (re.sub(r'\.layers\.\d+(\.|$)', r'.layers.*\1', module), int(match.group(1)) if match else None)


def aggregate(records):
    """Merge identical records across layers and steps; projection and attention rows keep their order as
    [call index, layer, step] triples (for attention, the number of projection calls before it)."""
    groups, calls = OrderedDict(), 0
    for row in records:
        name, layer = template(row['module'])
        key = json.dumps([row['phase'], name, row['source'], row['op'], row['inputs'], row['outputs'], row.get('eps')])
        item = groups.setdefault(key, {'phase': row['phase'], 'module': name, 'source': row['source'],
                                       'op': row['op'], 'inputs': row['inputs'],
                                       'outputs': row['outputs'], 'flops_each': row['flops'], 'count': 0,
                                       'layers': set(), 'steps': set()})
        item['count'] += 1
        if 'eps' in row:
            item['eps'] = row['eps']
        if row.get('bias_flops'):
            item['bias_flops_each'] = row['bias_flops']
        if layer is not None:
            item['layers'].add(layer)
        item['steps'].add(row['step'])
        if row['op'] in PROJECTION + ATTENTION:
            item.setdefault('calls', []).append([calls, layer, row['step']])
        calls += row['op'] in PROJECTION
    return [dict(item, layers=sorted(item['layers']), steps=sorted(item['steps'])) for item in groups.values()]


def calls(records):
    """Ordered projection calls: every Linear and Conv2d invocation, the unit Merlin captures hook."""
    return [{'index': i, 'module': row['module'], 'phase': row['phase'], 'step': row['step'], 'op': row['op'],
             'input': row['inputs'][0], 'weight': row['inputs'][1], 'output': row['outputs'][0]}
            for i, row in enumerate(r for r in records if r['op'] in PROJECTION)]


def attention(records):
    """Ordered attention cores, one per softmax (eager) or scaled_dot_product_attention call, with the
    number of projection calls that precede each."""
    result, before = [], 0
    for i, row in enumerate(records):
        common = {'module': row['module'], 'phase': row['phase'], 'step': row['step'], 'calls_before': before}
        if row['op'] in PROJECTION:
            before += 1
        elif row['op'] == 'aten.scaled_dot_product_attention.default':
            q, k, v = row['inputs'][:3]
            result.append(dict(common, kind='sdpa', q=q, k=k, v=v, output=row['outputs'][0]))
        elif row['op'] == 'aten.softmax.int':
            score = next(r for r in reversed(records[:i]) if r['op'] == 'aten.matmul.default')
            value = next(r for r in records[i + 1:] if r['op'] == 'aten.matmul.default')
            result.append(dict(common, kind='eager', scores=row['inputs'][0], qk=score['inputs'], pv=value['inputs'],
                               output=value['outputs'][0]))
    return result


def environment():
    import numpy
    import torch
    result = {'python': sys.version.split()[0], 'torch': torch.__version__,
              'numpy': numpy.__version__}
    try:
        import transformers
        result['transformers'] = transformers.__version__
    except ImportError:
        pass
    try:
        import importlib.metadata
        result['lerobot'] = importlib.metadata.version('lerobot')
    except Exception:
        pass
    return result


def load_adapter(path):
    """Load a read-only Python adapter exposing get_model_and_inputs() -> (Module, tuple[Tensor, ...])."""
    import importlib.util
    import torch
    path = Path(path).absolute()
    if any(re.search('hammer|vlsi', part, re.I) for part in path.parts):
        raise ValueError(f'restricted path: {path}')
    path = path.resolve(strict=True)
    if any(re.search('hammer|vlsi', part, re.I) for part in path.parts):
        raise ValueError(f'restricted path: {path}')
    loader_identity = identity(path)
    spec = importlib.util.spec_from_file_location('atlas_coverage_model_adapter', path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    model, inputs = module.get_model_and_inputs()
    if not isinstance(model, torch.nn.Module) or not isinstance(inputs, tuple):
        raise ValueError('loader must return (torch.nn.Module, tuple of positional inputs)')
    model.eval().requires_grad_(False)
    return model, inputs, '', {'loader': loader_identity}, {'inputs': tensors(inputs)}


def tinyllama(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from transformers.models.llama import modeling_llama
    sources = {'model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'revision': TINY_REVISION,
               'weights': identity(args.checkpoint / 'model.safetensors', TINY_WEIGHTS),
               'modeling_llama': identity(modeling_llama.__file__, TINY_MODELING),
               'corpus': identity(args.corpus, TINY_CORPUS)}
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True, use_fast=True)
    # capture_tinyllama_native.py: the corpus's first 160 tokens; the full capture prefills the first 128.
    first = tokenizer(args.corpus.read_text(encoding='utf-8'), return_tensors='pt',
                      add_special_tokens=True)['input_ids'][:, :160].contiguous()
    if hashlib.sha256(first.numpy().tobytes()).hexdigest() != TINY_TOKENS:
        raise ValueError('corpus token identity differs from the Merlin capture')
    tokens = first[:, :args.prefill].contiguous()
    prefill, decode = tokens.shape[1], args.decode_count
    model = AutoModelForCausalLM.from_pretrained(str(args.checkpoint), dtype=torch.bfloat16, attn_implementation='eager',
                                                 local_files_only=True).eval().requires_grad_(False)

    class Generate(torch.nn.Module):
        """Prefill, then greedy decode with a DynamicCache, the capture's step structure."""
        def __init__(self, model):
            super().__init__()
            self.m = model
            self.register_buffer('positions', torch.arange(prefill, prefill + decode).reshape(decode, 1, 1))

        def forward(self, ids):
            out = self.m(input_ids=ids, past_key_values=DynamicCache(config=self.m.config), use_cache=True)
            chosen = [out.logits[:, -1].argmax(-1)]
            for position in self.positions:
                out = self.m(input_ids=chosen[-1][:, None], past_key_values=out.past_key_values, use_cache=True,
                             position_ids=position, cache_position=position[0])
                chosen.append(out.logits[:, -1].argmax(-1))
            return torch.stack(chosen, -1)

    wrapper = Generate(model)
    with torch.inference_mode():
        native = wrapper(tokens).tolist()[0]
    return wrapper, (tokens,), 'm.', sources, {'prefill_rows': prefill, 'decode_count': decode,
                                                'native_greedy_tokens': native}


def smolvla(args):
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    sources = {'model': 'lerobot/smolvla_base', 'revision': SMOL_REVISION,
               'weights': identity(args.policy / 'model.safetensors', SMOL_WEIGHTS),
               'backbone': 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
               'backbone_config': identity(args.backbone / 'config.json', SMOL_BACKBONE_CONFIG),
               'observation': dict(identity(args.observation, SMOL_OBSERVATION), **SMOL_FRAME)}
    names = ('images', 'image_masks', 'language_tokens', 'language_mask', 'state', 'noise')
    with np.load(args.observation, allow_pickle=False) as archive:
        inputs = tuple(torch.from_numpy(archive[name].copy()) for name in names)
    config = PreTrainedConfig.from_pretrained(str(args.policy), local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError('wrong policy configuration')
    config.device = 'cpu'
    config.vlm_model_name = str(args.backbone)
    policy = SmolVLAPolicy.from_pretrained(str(args.policy), config=config, local_files_only=True,
                                           strict=True).eval().requires_grad_(False)

    class Sample(torch.nn.Module):
        """policy.model.sample_actions with the capture's image split and fixed noise."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images, image_masks, language_tokens, language_mask, state, noise):
            return self.model.sample_actions([images[:, i] for i in range(images.shape[1])],
                                             [image_masks[:, i] for i in range(image_masks.shape[1])],
                                             language_tokens, language_mask, state, noise=noise)

    shapes = {name: [list(value.shape), str(value.dtype).replace('torch.', '')] for name, value in zip(names, inputs)}
    return Sample(policy.model), inputs, 'model.', sources, {'inputs': shapes, 'denoising_steps': config.num_steps}


def phases(model, records):
    """TinyLlama: step = embedding calls so far minus one. SmolVLA: vision, prefix, then one step per action_in_proj."""
    seen = 0
    for row in records:
        if model not in ('tinyllama', 'smolvla'):
            row['step'], row['phase'] = 0, 'inference'
            continue
        if model == 'tinyllama':
            seen += row['op'] == 'aten.embedding.default'
            row['step'] = max(seen - 1, 0)
            row['phase'] = 'prefill' if row['step'] == 0 else 'decode'
            continue
        seen += row['module'] == 'action_in_proj' and row['op'] == 'aten.linear.default'
        row['step'] = max(seen - 1, 0)
        if seen:
            row['phase'] = 'denoise'
        elif '.vision_model.' in row['module'] + '.' or '.connector.' in row['module'] + '.':
            row['phase'] = 'vision'
        else:
            row['phase'] = 'prefix'


def crosscheck(model, ordered, receipt_path):
    """Compare the exported call order and geometry with a local Merlin capture receipt's hooked calls."""
    receipt = json.loads(Path(receipt_path).read_text())
    captured = receipt['calls']
    mismatches = []
    for mine, theirs in zip(ordered, captured):
        if model == 'tinyllama':
            rows = math.prod(mine['input'][0][:-1])
            width = max(panel['columns'][1] for panel in theirs['panels'])
            equal = (mine['module'] == theirs['site'] and mine['step'] == theirs['step']
                     and [rows, mine['input'][0][-1]] == theirs['A_bf16']['shape'] and mine['output'][0][-1] == width)
        else:
            equal = (mine['module'] == theirs['module'] and mine['input'][0] == theirs['input']['shape']
                     and mine['output'][0] == theirs['output']['shape']
                     and 'torch.' + mine['input'][1] == theirs['input']['dtype']
                     and 'torch.' + mine['output'][1] == theirs['output']['dtype'])
        if not equal:
            mismatches.append(mine['index'])
    equal = not mismatches and len(ordered) == len(captured)
    return {'scope': 'local Merlin capture receipt', 'receipt_status': receipt.get('status'),
            'receipt_sha256': sha256(receipt_path), 'exported_calls': len(ordered), 'captured_calls': len(captured),
            'mismatched_indices': mismatches[:20], 'status': 'PASS' if equal else 'FAIL'}


def dumps(value):
    """One line per list element at the top level, so inventories diff by record."""
    lines = []
    for key, item in value.items():
        if isinstance(item, list) and item and isinstance(item[0], dict):
            body = ',\n'.join('  ' + json.dumps(row, separators=(',', ':')) for row in item)
            lines.append(f' {json.dumps(key)}: [\n{body}\n ]')
        else:
            lines.append(f' {json.dumps(key)}: {json.dumps(item)}')
    return '{\n' + ',\n'.join(lines) + '\n}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help='tinyllama, smolvla, or a new model name with --loader')
    parser.add_argument('--loader', type=Path, help='Python file exposing get_model_and_inputs()')
    parser.add_argument('--out', type=Path, required=True, help='new directory for the full node list and graph')
    parser.add_argument('--fixture', type=Path, help='write the aggregated rows here (coverage/fixture/<model>.json)')
    parser.add_argument('--checkpoint', type=Path, help='TinyLlama snapshot directory at the pinned revision')
    parser.add_argument('--corpus', type=Path, help="Merlin's pinned TinyLlama development corpus text")
    parser.add_argument('--prefill', type=int, default=128)
    parser.add_argument('--decode-count', type=int, default=2)
    parser.add_argument('--policy', type=Path, help='lerobot/smolvla_base snapshot directory at the pinned revision')
    parser.add_argument('--backbone', type=Path, help='SmolVLM2-500M-Video-Instruct snapshot directory')
    parser.add_argument('--observation', type=Path, help='SmolVLA compiled observation archive (observation.npz)')
    parser.add_argument('--capture-receipt', help='optional local Merlin capture receipt to cross-check calls against')
    args = parser.parse_args()
    if args.loader and args.model in ('tinyllama', 'smolvla'):
        parser.error('--loader requires a distinct new model name to preserve the pinned focus-model adapters')
    if not args.loader and args.model not in ('tinyllama', 'smolvla'):
        parser.error('new models require --loader')
    import torch
    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    wrapper, inputs, strip, sources, trace = (load_adapter(args.loader) if args.loader else
        (tinyllama if args.model == 'tinyllama' else smolvla)(args))
    program = torch.export.export(wrapper, inputs, strict=False)
    exported = time.monotonic() - started
    (args.out / 'graph.txt').write_text(program.graph_module.print_readable(print_output=False))
    records = []
    walk(program.graph_module, strip, records)
    phases(args.model, records)
    ordered = calls(records)
    fixture = {
        'schema': SCHEMA, 'model': args.model, 'method': 'torch.export.export(strict=False)',
        'method_reason': 'Non-strict export traces the Python inference loop (cache objects, greedy feedback, '
                         'denoising steps) with fake tensors; strict TorchDynamo export was not needed.',
        'environment': environment(), 'sources': sources, 'trace': trace,
        'node_count': len(records), 'flops': sum(row['flops'] for row in records),
        'op_counts': dict(sorted(Counter(row['op'] for row in records).items())),
    }
    if args.capture_receipt:
        fixture['capture_crosscheck'] = crosscheck(args.model, ordered, args.capture_receipt)
    fixture['ops'] = aggregate(records)
    full = dict(fixture, argv=sys.argv, export_seconds=round(exported, 1), calls=ordered, attention=attention(records))
    (args.out / 'nodes.json').write_text(json.dumps(records) + '\n')
    (args.out / 'inventory.json').write_text(dumps(full))
    if args.fixture:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_text(dumps(fixture))
    print(json.dumps({'model': args.model, 'nodes': len(records), 'aggregated': len(fixture['ops']),
                      'calls': len(ordered), 'attention': len(full['attention']),
                      'crosscheck': fixture.get('capture_crosscheck', {}).get('status'),
                      'export_seconds': full['export_seconds']}))


if __name__ == '__main__':
    main()
