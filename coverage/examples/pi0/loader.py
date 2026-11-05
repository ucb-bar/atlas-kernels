"""Full LeRobot PI0 inference with deterministic random parameters and synthetic inputs.

Requires LeRobot 0.5.1, Transformers 4.57.1, and torch 2.10.0. No checkpoint is downloaded.
Shapes follow the public lerobot/pi0 configuration and the full PI0Config architecture.
"""
import copy
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
from types import FunctionType, MethodType, SimpleNamespace

import torch
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch
from lerobot.policies import pi_gemma
from transformers.masking_utils import create_causal_mask as _create_causal_mask
from transformers.cache_utils import Cache, CacheLayerMixin


def _clone_cache(value, memo=None):
    """Deep-copy only cache objects and their tensor/container state with traceable Tensor.clone."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.clone()
    elif isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(_clone_cache(item, memo) for item in value)
    elif isinstance(value, tuple):
        result = tuple(_clone_cache(item, memo) for item in value)
    elif isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update((_clone_cache(key, memo), _clone_cache(item, memo)) for key, item in value.items())
    elif isinstance(value, (Cache, CacheLayerMixin)):
        result = copy.copy(value)
        memo[id(value)] = result
        for key, item in vars(value).items():
            setattr(result, key, _clone_cache(item, memo))
    else:
        result = copy.deepcopy(value, memo)
    memo[id(value)] = result
    return result


def _create_causal_mask_compat(**kwargs):
    """Adapt a renamed Transformers keyword without changing the prepared four-dimensional mask."""
    kwargs['input_embeds'] = kwargs.pop('inputs_embeds')
    return _create_causal_mask(**kwargs)


def _get_image_features_compat(self, pixel_values):
    """Transformers 5 API/math, required by this LeRobot source, on Transformers 4.57.1."""
    outputs = self.vision_tower(pixel_values)
    outputs.pooler_output = self.multi_modal_projector(outputs.last_hidden_state)
    return outputs


class Sample(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, image_masks, tokens, token_masks, state, noise):
        return self.model.sample_actions(
            [images[:, camera] for camera in range(3)],
            [image_masks[:, camera] for camera in range(3)],
            tokens, token_masks, state, noise=noise, num_steps=10,
        )


def get_model_and_inputs():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    pi_gemma.create_causal_mask = _create_causal_mask_compat
    config = PI0Config(
        device='cpu', dtype='float32', paligemma_variant='gemma_2b',
        action_expert_variant='gemma_300m', image_resolution=(224, 224),
        chunk_size=50, n_action_steps=50, max_state_dim=32, max_action_dim=32,
        tokenizer_max_length=48, num_inference_steps=10, compile_model=False,
    )
    model = PI0Pytorch(config).eval().requires_grad_(False)
    # Preserve the original denoise_step bytecode. Substitute only its copy dependency in
    # a private function globals dictionary, leaving Python's copy module and LeRobot unchanged.
    original = PI0Pytorch.denoise_step
    denoise = FunctionType(original.__code__, dict(original.__globals__, copy=SimpleNamespace(deepcopy=_clone_cache)),
                           original.__name__, original.__defaults__, original.__closure__)
    model.denoise_step = MethodType(denoise, model)
    paligemma = model.paligemma_with_expert.paligemma.model
    paligemma.get_image_features = MethodType(_get_image_features_compat, paligemma)
    model.paligemma_with_expert.paligemma.config._attn_implementation = 'eager'
    model.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation = 'eager'
    inputs = (
        torch.rand(1, 3, 3, 224, 224),
        torch.ones(1, 3, dtype=torch.bool),
        torch.arange(48, dtype=torch.int64).reshape(1, 48) + 1,
        torch.ones(1, 48, dtype=torch.bool),
        torch.cat([torch.randn(1, 6), torch.zeros(1, 26)], dim=-1),
        torch.randn(1, 50, 32),
    )
    if destination := os.environ.get('PI0_METADATA'):
        import importlib.metadata
        from lerobot.policies.pi0 import modeling_pi0, configuration_pi0
        from transformers.models.gemma import modeling_gemma
        from transformers.models.siglip import modeling_siglip
        from transformers.models.paligemma import modeling_paligemma
        weight_hash = hashlib.sha256()
        parameter_count = 0
        for name, parameter in model.named_parameters():
            parameter_count += parameter.numel()
            weight_hash.update(json.dumps([name, list(parameter.shape), str(parameter.dtype)]).encode())
            weight_hash.update(memoryview(parameter.detach().numpy()).cast('B'))
        modules = [modeling_pi0, configuration_pi0, pi_gemma, modeling_gemma, modeling_siglip, modeling_paligemma]
        metadata = {
            'checkpoint': None, 'weights': 'deterministic random initialization', 'seed': 0,
            'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'lerobot', 'transformers')},
            'config': dataclasses.asdict(config), 'camera_order': [0, 1, 2],
            'input_scope': 'preprocessed synthetic model inputs, no dataset, tokenizer, or raw image processor',
            'num_steps': 10, 'use_cache': True, 'attention_requested': 'eager',
            'parameter_count': parameter_count, 'unique_parameter_bytes_sha256': weight_hash.hexdigest(),
            'parameter_hash_order': 'named_parameters default deduplication; JSON name/shape/dtype then contiguous bytes',
            'sources': {module.__name__: {'file': Path(inspect.getfile(module)).name, 'sha256': hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()} for module in modules},
            'inputs': [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()} for name, value in zip(('images', 'image_masks', 'tokens', 'token_masks', 'state', 'noise'), inputs)],
            'token_ids': inputs[2].tolist(),
            'compatibility': ['get_image_features returns projected pooler_output using Transformers 5 API/math; installed Transformers 4.57.1 returns a scaled Tensor', 'create_causal_mask keyword inputs_embeds renamed input_embeds for Transformers 4.57.1; delegated function preserves prepared 4D masks', 'private denoise_step function globals substitute tensor-cloning cache deep copy; bytecode and module globals unchanged'],
        }
        Path(destination).write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    return Sample(model), inputs
