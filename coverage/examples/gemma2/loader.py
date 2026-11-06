"""Full-size Gemma 2 2B, seeded random BF16 weights, 128-token prefill and one decode.

No network or checkpoint access occurs. GEMMA2_OUT selects the scratch metadata directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_model_and_inputs():
    import torch
    import transformers
    from transformers import Gemma2Config, Gemma2ForCausalLM, DynamicCache
    from transformers import cache_utils
    from transformers.models.gemma2 import configuration_gemma2, modeling_gemma2

    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    print('Gemma2: initializing complete 2B BF16 model', flush=True)
    config = Gemma2Config()
    config._attn_implementation = 'eager'
    # These are the real 2B defaults, independently checked against Google's public config.
    assert (config.num_hidden_layers, config.hidden_size, config.intermediate_size,
            config.num_attention_heads, config.num_key_value_heads, config.head_dim,
            config.vocab_size) == (26, 2304, 9216, 8, 4, 256, 256000)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Gemma2ForCausalLM(config).eval().requires_grad_(False)
    finally:
        torch.set_default_dtype(previous)
    ids = torch.arange(3, 131, dtype=torch.long)[None, :]
    ids[0, 0] = config.bos_token_id

    class Generate(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.m = model

        def forward(self, tokens):
            cache = DynamicCache(config=self.m.config)
            prefill = self.m(input_ids=tokens, past_key_values=cache, use_cache=True)
            chosen = prefill.logits[:, -1].argmax(-1)
            decode = self.m(input_ids=chosen[:, None], past_key_values=prefill.past_key_values,
                            use_cache=True, position_ids=torch.full((1, 1), 128, device=tokens.device),
                            cache_position=torch.full((1,), 128, device=tokens.device))
            return torch.stack((chosen, decode.logits[:, -1].argmax(-1)), -1)

    wrapper = Generate(model)
    print('Gemma2: hashing unique parameter stream', flush=True)
    weights = hashlib.sha256()
    parameter_count = 0
    for name, value in model.named_parameters():
        parameter_count += value.numel()
        weights.update(name.encode())
        weights.update(value.detach().view(torch.uint8).numpy().tobytes())
    out = Path(os.environ['GEMMA2_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get('GEMMA2_SKIP_NATIVE') == '1':
        prior = json.loads((out / 'loader-metadata.json').read_text())
        assert weights.hexdigest() == prior['unique_parameter_stream_sha256']
        assert digest(modeling_gemma2.__file__) == prior['modeling_gemma2_sha256']
        native_tokens = prior['native_greedy_tokens']
    else:
        print('Gemma2: running native prefill and cached decode', flush=True)
        with torch.inference_mode():
            native_tokens = wrapper(ids).tolist()
    config_data = config.to_dict()
    config_data['_attn_implementation'] = 'eager'
    (out / 'config.json').write_text(json.dumps(config_data, indent=2) + '\n')
    metadata = dict(model='google/gemma-2-2b', checkpoint='NONE: gated files, random initialization',
                    seed=20261104, dtype='bfloat16', attention='eager', prefill_rows=128,
                    decode_count=1, cache='DynamicCache(config=Gemma2Config)',
                    parameters=parameter_count, unique_parameter_stream_sha256=weights.hexdigest(),
                    parameter_hash_format='named_parameters order: UTF-8 name then contiguous raw BF16 bytes',
                    input_ids=ids.tolist(), input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                    native_greedy_tokens=native_tokens, python=__import__('sys').version,
                    torch=torch.__version__, transformers=transformers.__version__,
                    configuration_gemma2_sha256=digest(configuration_gemma2.__file__),
                    modeling_gemma2_sha256=digest(modeling_gemma2.__file__),
                    cache_utils_sha256=digest(cache_utils.__file__),
                    public_config_repository='google/gemma_pytorch',
                    public_config_commit='014acb7ac4563a5f77c76d7ff98f31b568c16508',
                    public_config_path='gemma/config.py:get_config_for_2b_v2',
                    public_config_git_blob='710f6d18be53c99c1170f6f8f6408b51c7c3741f',
                    config_sha256=digest(out / 'config.json'))
    (out / 'loader-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Gemma2: loader ready for torch.export', flush=True)
    return wrapper, (ids,)
