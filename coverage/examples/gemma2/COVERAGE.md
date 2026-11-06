# gemma2 starter coverage

Historical random-weight starter report from `edce231`, edited to retain model-specific data. This is an INCOMPLETE candidate mapping, with no demonstrated whole-model Atlas execution. See [STARTER.md](STARTER.md) for workload, reproduction, and pending gates.

## gemma2 (gemma2)

Export: `torch.export.export(strict=False)` in Python 3.12.12, torch 2.10.0+cu128, transformers 4.57.1, lerobot 0.5.1; 5,118 ATen records in 286 aggregated rows. Merlin real-value capture cross-check: NOT_RUN.

Candidate placement rules are in [mapping.yaml](mapping.yaml); no qualified model build is registered.

Projection panel counts are informational; the export derives calls, not physical panels.

Estimated candidate matrix-product work: 674.438 GFLOP, host 0.000 GFLOP (99.4219% of exported runtime FLOPs assigned to candidates); build-time constants 0.000 GFLOP, not run on the target.

**INCOMPLETE: 270 unmapped rows, 4,752 op instances, 3.921 GFLOP.** Unmapped FLOPs are included in the runtime denominator; these rows are an implementation work list, not demonstrated Atlas support.

| phase | module | source | op/component | shape | count | GFLOP |
| --- | --- | --- | --- | --- | ---: | ---: |
| inference | `m.model.embed_tokens` | `modeling_gemma2.py:forward:413` | `aten.embedding.default` | 256000x2304, 1x128 -> 1x128x2304 bfloat16 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:420` | `aten.arange.start` | - -> 128 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:425` | `aten.unsqueeze.default` | 128 -> 1x128 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.arange.default` | - -> 128 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.add_.Tensor` | 128 -> 128 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.arange.default` | - -> 1 int64 | 4 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch.lazy_load_decompositions` | - -> scalar  | 8 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._vmap_increment_nesting` | - -> scalar  | 8 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._add_batch_dim` | 1 -> scalar int64 | 5 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._add_batch_dim` | 128 -> scalar int64 | 2 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.le.Tensor` | scalar, scalar -> scalar bool | 2 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._remove_batch_dim` | scalar -> 128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._vmap_decrement_nesting` | - -> scalar  | 8 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._remove_batch_dim` | 128 -> 128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._remove_batch_dim` | 128x128 -> 1x128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.expand.default` | 128x128 -> 1x128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._remove_batch_dim` | 1x128x128 -> 1x1x128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.expand.default` | 1x128x128 -> 1x1x128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.where.ScalarOther` | 1x1x128x128, scalar -> 1x1x128x128 bfloat16 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.arange.default` | - -> 128 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.add_.Tensor` | 128 -> 128 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.arange.default` | - -> 1 int64 | 4 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch.lazy_load_decompositions` | - -> scalar  | 8 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._vmap_increment_nesting` | - -> scalar  | 8 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._add_batch_dim` | 1 -> scalar int64 | 5 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._add_batch_dim` | 128 -> scalar int64 | 2 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.new_ones.default` | scalar -> scalar bool | 2 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.sub.Tensor` | scalar -> scalar int64 | 2 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.gt.Tensor` | scalar, scalar -> scalar bool | 2 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.to.dtype_layout` | scalar -> scalar bool | 4 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.__and__.Tensor` | scalar, scalar -> scalar bool | 4 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.le.Tensor` | scalar, scalar -> scalar bool | 2 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._remove_batch_dim` | scalar -> 128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._vmap_decrement_nesting` | - -> scalar  | 8 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._remove_batch_dim` | 128 -> 128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._remove_batch_dim` | 128x128 -> 1x128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.expand.default` | 128x128 -> 1x128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._remove_batch_dim` | 1x128x128 -> 1x1x128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.expand.default` | 1x128x128 -> 1x1x128x128 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.where.ScalarOther` | 1x1x128x128, scalar -> 1x1x128x128 bfloat16 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:109` | `aten.unsqueeze.default` | 128 -> 1x128 float32 | 2 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:109` | `aten.unsqueeze.default` | 1x128 -> 1x128x1 float32 | 2 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:109` | `aten.to.dtype` | 1x128x1 -> 1x128x1 float32 | 2 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:109` | `aten.expand.default` | 1x128x1 -> 1x128x1 float32 | 2 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:109` | `aten.to.dtype_layout` | 1x128x1 -> 1x128x1 float32 | 2 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:110` | `aten.unsqueeze.default` | 1x128 -> 1x1x128 int64 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:110` | `aten.to.dtype` | 1x1x128 -> 1x1x128 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:114` | `aten.to.dtype` | 1x128x1 -> 1x128x1 float32 | 2 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:114` | `aten.to.dtype` | 1x1x128 -> 1x1x128 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:114` | `aten.matmul.default` | 1x128x1, 1x1x128 -> 1x128x128 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:114` | `aten.transpose.int` | 1x128x128 -> 1x128x128 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:115` | `aten.cat.default` | 1x128x128, 1x128x128 -> 1x128x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:116` | `aten.cos.default` | 1x128x256 -> 1x128x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:116` | `aten.mul.Tensor` | 1x128x256 -> 1x128x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:117` | `aten.sin.default` | 1x128x256 -> 1x128x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:117` | `aten.mul.Tensor` | 1x128x256 -> 1x128x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:119` | `aten.to.dtype` | 1x128x256 -> 1x128x256 bfloat16 | 2 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:454` | `aten.mul.Tensor` | 1x128x2304, scalar -> 1x128x2304 bfloat16 | 1 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x128x2304 -> 1x128x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x128x2304 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x128x2304 -> 1x128x1 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x128x1 -> 1x128x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x128x1 -> 1x128x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x128x2304, 1x128x1 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:63` | `aten.to.dtype` | 2304 -> 2304 float32 | 52 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:63` | `aten.add.Tensor` | 2304 -> 2304 float32 | 52 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x128x2304, 2304 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x128x2304, 1x128x2304 -> 1x128x2304 bfloat16 | 26 | 0.008 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:244` | `aten.view.default` | 1x128x2048 -> 1x128x8x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:244` | `aten.transpose.int` | 1x128x8x256 -> 1x8x128x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:245` | `aten.view.default` | 1x128x1024 -> 1x128x4x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:245` | `aten.transpose.int` | 1x128x4x256 -> 1x4x128x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:246` | `aten.view.default` | 1x128x1024 -> 1x128x4x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:246` | `aten.transpose.int` | 1x128x4x256 -> 1x4x128x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.unsqueeze.default` | 1x128x256 -> 1x1x128x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.mul.Tensor` | 1x8x128x256, 1x1x128x256 -> 1x8x128x256 bfloat16 | 52 | 0.014 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.slice.Tensor` | 1x8x128x256 -> 1x8x128x128 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.neg.default` | 1x8x128x128 -> 1x8x128x128 bfloat16 | 26 | 0.003 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.cat.default` | 1x8x128x128, 1x8x128x128 -> 1x8x128x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.add.Tensor` | 1x8x128x256, 1x8x128x256 -> 1x8x128x256 bfloat16 | 26 | 0.007 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.mul.Tensor` | 1x4x128x256, 1x1x128x256 -> 1x4x128x256 bfloat16 | 52 | 0.007 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.slice.Tensor` | 1x4x128x256 -> 1x4x128x128 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.neg.default` | 1x4x128x128 -> 1x4x128x128 bfloat16 | 26 | 0.002 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.cat.default` | 1x4x128x128, 1x4x128x128 -> 1x4x128x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.add.Tensor` | 1x4x128x256, 1x4x128x256 -> 1x4x128x256 bfloat16 | 26 | 0.003 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:254` | `aten.cat.default` | 0, 1x4x128x256 -> 1x4x128x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:254` | `aten.slice.Tensor` | 1x4x128x256 -> 1x4x128x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.unsqueeze.default` | 1x4x128x256 -> 1x4x1x128x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.expand.default` | 1x4x1x128x256 -> 1x4x2x128x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.reshape.default` | 1x4x2x128x256 -> 1x8x128x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.transpose.int` | 1x8x128x256 -> 1x8x256x128 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.matmul.default` | 1x8x128x256, 1x8x256x128 -> 1x8x128x128 bfloat16 | 26 | 1.745 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.mul.Tensor` | 1x8x128x128 -> 1x8x128x128 bfloat16 | 52 | 0.007 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.div.Tensor` | 1x8x128x128 -> 1x8x128x128 bfloat16 | 26 | 0.003 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.tanh.default` | 1x8x128x128 -> 1x8x128x128 bfloat16 | 26 | 0.003 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.add.Tensor` | 1x8x128x128, 1x1x128x128 -> 1x8x128x128 bfloat16 | 26 | 0.003 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.softmax.int` | 1x8x128x128 -> 1x8x128x128 float32 | 26 | 0.017 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.to.dtype` | 1x8x128x128 -> 1x8x128x128 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.matmul.default` | 1x8x128x128, 1x8x128x256 -> 1x8x128x256 bfloat16 | 26 | 1.745 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.transpose.int` | 1x8x128x256 -> 1x128x8x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.contiguous.default` | 1x128x8x256 -> 1x128x8x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:273` | `aten.reshape.default` | 1x128x8x256 -> 1x128x2048 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x128x2304 -> 1x128x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x128x2304 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x128x2304 -> 1x128x1 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x128x1 -> 1x128x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x128x1 -> 1x128x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x128x2304, 1x128x1 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:63` | `aten.to.dtype` | 2304 -> 2304 float32 | 52 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:63` | `aten.add.Tensor` | 2304 -> 2304 float32 | 52 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x128x2304, 2304 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x128x2304, 1x128x2304 -> 1x128x2304 bfloat16 | 26 | 0.008 |
| inference | `m.model.layers.*` | `modeling_gemma2.py:forward:322` | `aten.add.Tensor` | 1x128x2304, 1x128x2304 -> 1x128x2304 bfloat16 | 26 | 0.008 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x128x2304 -> 1x128x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x128x2304 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x128x2304 -> 1x128x1 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x128x1 -> 1x128x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x128x1 -> 1x128x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x128x2304, 1x128x1 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:63` | `aten.to.dtype` | 2304 -> 2304 float32 | 52 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:63` | `aten.add.Tensor` | 2304 -> 2304 float32 | 52 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x128x2304, 2304 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x128x2304, 1x128x2304 -> 1x128x2304 bfloat16 | 26 | 0.008 |
| inference | `m.model.layers.*.mlp.act_fn` | `activations.py:forward:51` | `aten.gelu.default` | 1x128x9216 -> 1x128x9216 bfloat16 | 26 | 0.031 |
| inference | `m.model.layers.*.mlp` | `modeling_gemma2.py:forward:82` | `aten.mul.Tensor` | 1x128x9216, 1x128x9216 -> 1x128x9216 bfloat16 | 26 | 0.031 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x128x2304 -> 1x128x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x128x2304 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x128x2304 -> 1x128x1 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x128x1 -> 1x128x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x128x1 -> 1x128x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x128x2304, 1x128x1 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:63` | `aten.to.dtype` | 2304 -> 2304 float32 | 52 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:63` | `aten.add.Tensor` | 2304 -> 2304 float32 | 52 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x128x2304, 2304 -> 1x128x2304 float32 | 26 | 0.008 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x128x2304, 1x128x2304 -> 1x128x2304 bfloat16 | 26 | 0.008 |
| inference | `m.model.layers.*` | `modeling_gemma2.py:forward:328` | `aten.add.Tensor` | 1x128x2304, 1x128x2304 -> 1x128x2304 bfloat16 | 26 | 0.008 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x128x2304 -> 1x128x2304 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x128x2304 -> 1x128x2304 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x128x2304 -> 1x128x1 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x128x1 -> 1x128x1 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x128x1 -> 1x128x1 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x128x2304, 1x128x1 -> 1x128x2304 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:63` | `aten.to.dtype` | 2304 -> 2304 float32 | 2 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:63` | `aten.add.Tensor` | 2304 -> 2304 float32 | 2 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x128x2304, 2304 -> 1x128x2304 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x128x2304, 1x128x2304 -> 1x128x2304 bfloat16 | 1 | 0.000 |
| inference | `m` | `modeling_gemma2.py:forward:567` | `aten.div.Tensor` | 1x128x256000 -> 1x128x256000 bfloat16 | 1 | 0.033 |
| inference | `m` | `modeling_gemma2.py:forward:568` | `aten.tanh.default` | 1x128x256000 -> 1x128x256000 bfloat16 | 1 | 0.033 |
| inference | `m` | `modeling_gemma2.py:forward:569` | `aten.mul.Tensor` | 1x128x256000 -> 1x128x256000 bfloat16 | 1 | 0.033 |
| inference | `(top)` | `loader.py:forward:50` | `aten.select.int` | 1x128x256000 -> 1x256000 bfloat16 | 1 | 0.000 |
| inference | `(top)` | `loader.py:forward:50` | `aten.argmax.default` | 1x256000 -> 1 int64 | 1 | 0.000 |
| inference | `(top)` | `loader.py:forward:51` | `aten.unsqueeze.default` | 1 -> 1x1 int64 | 1 | 0.000 |
| inference | `(top)` | `loader.py:forward:52` | `aten.full.default` | - -> 1x1 int64 | 1 | 0.000 |
| inference | `(top)` | `loader.py:forward:53` | `aten.full.default` | - -> 1 int64 | 1 | 0.000 |
| inference | `m.model.embed_tokens` | `modeling_gemma2.py:forward:413` | `aten.embedding.default` | 256000x2304, 1x1 -> 1x1x2304 bfloat16 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.arange.default` | - -> 129 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.add_.Tensor` | 129 -> 129 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._add_batch_dim` | 129 -> scalar int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._remove_batch_dim` | scalar -> 129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._remove_batch_dim` | 129 -> 1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._remove_batch_dim` | 1x129 -> 1x1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.expand.default` | 1x129 -> 1x1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `torch._functorch.predispatch._remove_batch_dim` | 1x1x129 -> 1x1x1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.expand.default` | 1x1x129 -> 1x1x1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:440` | `aten.where.ScalarOther` | 1x1x1x129, scalar -> 1x1x1x129 bfloat16 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.arange.default` | - -> 129 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.add_.Tensor` | 129 -> 129 int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._add_batch_dim` | 129 -> scalar int64 | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._remove_batch_dim` | scalar -> 129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._remove_batch_dim` | 129 -> 1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._remove_batch_dim` | 1x129 -> 1x1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.expand.default` | 1x129 -> 1x1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `torch._functorch.predispatch._remove_batch_dim` | 1x1x129 -> 1x1x1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.expand.default` | 1x1x129 -> 1x1x1x129 bool | 1 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:441` | `aten.where.ScalarOther` | 1x1x1x129, scalar -> 1x1x1x129 bfloat16 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:110` | `aten.unsqueeze.default` | 1x1 -> 1x1x1 int64 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:110` | `aten.to.dtype` | 1x1x1 -> 1x1x1 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:114` | `aten.to.dtype` | 1x1x1 -> 1x1x1 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:114` | `aten.matmul.default` | 1x128x1, 1x1x1 -> 1x128x1 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:114` | `aten.transpose.int` | 1x128x1 -> 1x1x128 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:115` | `aten.cat.default` | 1x1x128, 1x1x128 -> 1x1x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:116` | `aten.cos.default` | 1x1x256 -> 1x1x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:116` | `aten.mul.Tensor` | 1x1x256 -> 1x1x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:117` | `aten.sin.default` | 1x1x256 -> 1x1x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:117` | `aten.mul.Tensor` | 1x1x256 -> 1x1x256 float32 | 1 | 0.000 |
| inference | `m.model.rotary_emb` | `modeling_gemma2.py:forward:119` | `aten.to.dtype` | 1x1x256 -> 1x1x256 bfloat16 | 2 | 0.000 |
| inference | `m.model` | `modeling_gemma2.py:forward:454` | `aten.mul.Tensor` | 1x1x2304, scalar -> 1x1x2304 bfloat16 | 1 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x1x2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x1x2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x1x2304 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x1x1 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x1x1 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x1x2304, 1x1x1 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x1x2304, 2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.input_layernorm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x1x2304, 1x1x2304 -> 1x1x2304 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:244` | `aten.view.default` | 1x1x2048 -> 1x1x8x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:244` | `aten.transpose.int` | 1x1x8x256 -> 1x8x1x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:245` | `aten.view.default` | 1x1x1024 -> 1x1x4x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:245` | `aten.transpose.int` | 1x1x4x256 -> 1x4x1x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:246` | `aten.view.default` | 1x1x1024 -> 1x1x4x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:246` | `aten.transpose.int` | 1x1x4x256 -> 1x4x1x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.unsqueeze.default` | 1x1x256 -> 1x1x1x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.mul.Tensor` | 1x8x1x256, 1x1x1x256 -> 1x8x1x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.slice.Tensor` | 1x8x1x256 -> 1x8x1x128 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.neg.default` | 1x8x1x128 -> 1x8x1x128 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.cat.default` | 1x8x1x128, 1x8x1x128 -> 1x8x1x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.add.Tensor` | 1x8x1x256, 1x8x1x256 -> 1x8x1x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.mul.Tensor` | 1x4x1x256, 1x1x1x256 -> 1x4x1x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.slice.Tensor` | 1x4x1x256 -> 1x4x1x128 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.neg.default` | 1x4x1x128 -> 1x4x1x128 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.cat.default` | 1x4x1x128, 1x4x1x128 -> 1x4x1x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:249` | `aten.add.Tensor` | 1x4x1x256, 1x4x1x256 -> 1x4x1x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:254` | `aten.cat.default` | 1x4x128x256, 1x4x1x256 -> 1x4x129x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:254` | `aten.slice.Tensor` | 1x4x129x256 -> 1x4x129x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.unsqueeze.default` | 1x4x129x256 -> 1x4x1x129x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.expand.default` | 1x4x1x129x256 -> 1x4x2x129x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.reshape.default` | 1x4x2x129x256 -> 1x8x129x256 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.transpose.int` | 1x8x129x256 -> 1x8x256x129 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.matmul.default` | 1x8x1x256, 1x8x256x129 -> 1x8x1x129 bfloat16 | 26 | 0.014 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.mul.Tensor` | 1x8x1x129 -> 1x8x1x129 bfloat16 | 52 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.div.Tensor` | 1x8x1x129 -> 1x8x1x129 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.tanh.default` | 1x8x1x129 -> 1x8x1x129 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.add.Tensor` | 1x8x1x129, 1x1x1x129 -> 1x8x1x129 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.softmax.int` | 1x8x1x129 -> 1x8x1x129 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.to.dtype` | 1x8x1x129 -> 1x8x1x129 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.matmul.default` | 1x8x1x129, 1x8x129x256 -> 1x8x1x256 bfloat16 | 26 | 0.014 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:260` | `aten.transpose.int` | 1x8x1x256 -> 1x1x8x256 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.self_attn` | `modeling_gemma2.py:forward:273` | `aten.reshape.default` | 1x1x8x256 -> 1x1x2048 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x1x2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x1x2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x1x2304 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x1x1 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x1x1 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x1x2304, 1x1x1 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x1x2304, 2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_attention_layernorm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x1x2304, 1x1x2304 -> 1x1x2304 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*` | `modeling_gemma2.py:forward:322` | `aten.add.Tensor` | 1x1x2304, 1x1x2304 -> 1x1x2304 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x1x2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x1x2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x1x2304 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x1x1 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x1x1 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x1x2304, 1x1x1 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x1x2304, 2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.pre_feedforward_layernorm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x1x2304, 1x1x2304 -> 1x1x2304 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.mlp.act_fn` | `activations.py:forward:51` | `aten.gelu.default` | 1x1x9216 -> 1x1x9216 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.mlp` | `modeling_gemma2.py:forward:82` | `aten.mul.Tensor` | 1x1x9216, 1x1x9216 -> 1x1x9216 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x1x2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x1x2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x1x2304 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x1x1 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x1x1 -> 1x1x1 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x1x2304, 1x1x1 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x1x2304, 2304 -> 1x1x2304 float32 | 26 | 0.000 |
| inference | `m.model.layers.*.post_feedforward_layernorm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x1x2304, 1x1x2304 -> 1x1x2304 bfloat16 | 26 | 0.000 |
| inference | `m.model.layers.*` | `modeling_gemma2.py:forward:328` | `aten.add.Tensor` | 1x1x2304, 1x1x2304 -> 1x1x2304 bfloat16 | 26 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.to.dtype` | 1x1x2304 -> 1x1x2304 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.pow.Tensor_Scalar` | 1x1x2304 -> 1x1x2304 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.mean.dim` | 1x1x2304 -> 1x1x1 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.add.Tensor` | 1x1x1 -> 1x1x1 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.rsqrt.default` | 1x1x1 -> 1x1x1 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:60` | `aten.mul.Tensor` | 1x1x2304, 1x1x1 -> 1x1x2304 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:63` | `aten.mul.Tensor` | 1x1x2304, 2304 -> 1x1x2304 float32 | 1 | 0.000 |
| inference | `m.model.norm` | `modeling_gemma2.py:forward:64` | `aten.type_as.default` | 1x1x2304, 1x1x2304 -> 1x1x2304 bfloat16 | 1 | 0.000 |
| inference | `m` | `modeling_gemma2.py:forward:567` | `aten.div.Tensor` | 1x1x256000 -> 1x1x256000 bfloat16 | 1 | 0.000 |
| inference | `m` | `modeling_gemma2.py:forward:568` | `aten.tanh.default` | 1x1x256000 -> 1x1x256000 bfloat16 | 1 | 0.000 |
| inference | `m` | `modeling_gemma2.py:forward:569` | `aten.mul.Tensor` | 1x1x256000 -> 1x1x256000 bfloat16 | 1 | 0.000 |
| inference | `(top)` | `loader.py:forward:54` | `aten.select.int` | 1x1x256000 -> 1x256000 bfloat16 | 1 | 0.000 |
| inference | `(top)` | `loader.py:forward:54` | `aten.argmax.default` | 1x256000 -> 1 int64 | 1 | 0.000 |
| inference | `(top)` | `loader.py:forward:54` | `aten.stack.default` | 1, 1 -> 1x2 int64 | 1 | 0.000 |

| where | op instances | GFLOP |
| --- | ---: | ---: |
| `atlas:projection_gemm` | 366 | 674.438 |

| phase | module | op | shape | count | where | GFLOP |
| --- | --- | --- | --- | ---: | --- | ---: |
| inference | `m.model.layers.*.self_attn.q_proj` | `linear.default` | 1x128x2304, 2048x2304 -> 1x128x2048 bfloat16 | 26 | `atlas:projection_gemm` | 31.407 |
| inference | `m.model.layers.*.self_attn.k_proj` | `linear.default` | 1x128x2304, 1024x2304 -> 1x128x1024 bfloat16 | 26 | `atlas:projection_gemm` | 15.703 |
| inference | `m.model.layers.*.self_attn.v_proj` | `linear.default` | 1x128x2304, 1024x2304 -> 1x128x1024 bfloat16 | 26 | `atlas:projection_gemm` | 15.703 |
| inference | `m.model.layers.*.self_attn.o_proj` | `linear.default` | 1x128x2048, 2304x2048 -> 1x128x2304 bfloat16 | 26 | `atlas:projection_gemm` | 31.407 |
| inference | `m.model.layers.*.mlp.gate_proj` | `linear.default` | 1x128x2304, 9216x2304 -> 1x128x9216 bfloat16 | 26 | `atlas:projection_gemm` | 141.331 |
| inference | `m.model.layers.*.mlp.up_proj` | `linear.default` | 1x128x2304, 9216x2304 -> 1x128x9216 bfloat16 | 26 | `atlas:projection_gemm` | 141.331 |
| inference | `m.model.layers.*.mlp.down_proj` | `linear.default` | 1x128x9216, 2304x9216 -> 1x128x2304 bfloat16 | 26 | `atlas:projection_gemm` | 141.331 |
| inference | `m.lm_head` | `linear.default` | 1x128x2304, 256000x2304 -> 1x128x256000 bfloat16 | 1 | `atlas:projection_gemm` | 150.995 |
| inference | `m.model.layers.*.self_attn.q_proj` | `linear.default` | 1x1x2304, 2048x2304 -> 1x1x2048 bfloat16 | 26 | `atlas:projection_gemm` | 0.245 |
| inference | `m.model.layers.*.self_attn.k_proj` | `linear.default` | 1x1x2304, 1024x2304 -> 1x1x1024 bfloat16 | 26 | `atlas:projection_gemm` | 0.123 |
| inference | `m.model.layers.*.self_attn.v_proj` | `linear.default` | 1x1x2304, 1024x2304 -> 1x1x1024 bfloat16 | 26 | `atlas:projection_gemm` | 0.123 |
| inference | `m.model.layers.*.self_attn.o_proj` | `linear.default` | 1x1x2048, 2304x2048 -> 1x1x2304 bfloat16 | 26 | `atlas:projection_gemm` | 0.245 |
| inference | `m.model.layers.*.mlp.gate_proj` | `linear.default` | 1x1x2304, 9216x2304 -> 1x1x9216 bfloat16 | 26 | `atlas:projection_gemm` | 1.104 |
| inference | `m.model.layers.*.mlp.up_proj` | `linear.default` | 1x1x2304, 9216x2304 -> 1x1x9216 bfloat16 | 26 | `atlas:projection_gemm` | 1.104 |
| inference | `m.model.layers.*.mlp.down_proj` | `linear.default` | 1x1x9216, 2304x9216 -> 1x1x2304 bfloat16 | 26 | `atlas:projection_gemm` | 1.104 |
| inference | `m.lm_head` | `linear.default` | 1x1x2304, 256000x2304 -> 1x1x256000 bfloat16 | 1 | `atlas:projection_gemm` | 1.180 |

Zero-FLOP records (views, casts, copies, and index construction):

| phase | where | ops |
| --- | --- | --- |
