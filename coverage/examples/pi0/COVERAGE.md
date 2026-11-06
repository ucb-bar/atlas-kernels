# pi0 starter coverage

Historical random-weight starter report from `edce231`, edited to retain model-specific data. This is an INCOMPLETE candidate mapping, with no demonstrated whole-model Atlas execution. See [STARTER.md](STARTER.md) for workload, reproduction, and pending gates.

## pi0 (pi0)

Export: `torch.export.export(strict=False)` in Python 3.12.12, torch 2.10.0+cu128, transformers 4.57.1, lerobot 0.5.1; 17,440 ATen records in 289 aggregated rows. Merlin real-value capture cross-check: NOT_RUN.

Candidate placement rules are in [mapping.yaml](mapping.yaml); no qualified model build is registered.

Projection panel counts are informational; the export derives calls, not physical panels.

Estimated candidate matrix-product work: 4,191.248 GFLOP, host 0.000 GFLOP (95.6348% of exported runtime FLOPs assigned to candidates); build-time constants 0.000 GFLOP, not run on the target.

**INCOMPLETE: 275 unmapped rows, 16,054 op instances, 191.308 GFLOP.** Unmapped FLOPs are included in the runtime denominator; these rows are an implementation work list, not demonstrated Atlas support.

| phase | module | source | op/component | shape | count | GFLOP |
| --- | --- | --- | --- | --- | ---: | ---: |
| prefix | `(top)` | `loader.py:forward:72` | `aten.select.int` | 1x3x3x224x224 -> 1x3x224x224 float32 | 3 | 0.000 |
| prefix | `(top)` | `loader.py:forward:73` | `aten.select.int` | 1x3 -> 1 bool | 3 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings` | `modeling_siglip.py:forward:278` | `aten.to.dtype` | 1x3x224x224 -> 1x3x224x224 float32 | 3 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding` | `modeling_siglip.py:forward:278` | `aten.add.Tensor` bias | 1x3x224x224, 1152x3x14x14 -> 1x1152x16x16 float32 | 3 | 0.001 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings` | `modeling_siglip.py:forward:279` | `aten.flatten.using_ints` | 1x1152x16x16 -> 1x1152x256 float32 | 3 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings` | `modeling_siglip.py:forward:279` | `aten.transpose.int` | 1x1152x256 -> 1x256x1152 float32 | 3 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding` | `modeling_siglip.py:forward:284` | `aten.embedding.default` | 256x1152, 1x256 -> 1x256x1152 float32 | 3 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings` | `modeling_siglip.py:forward:284` | `aten.add.Tensor` | 1x256x1152, 1x256x1152 -> 1x256x1152 float32 | 3 | 0.001 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.layer_norm1` | `modeling_siglip.py:forward:448` | `aten.layer_norm.default` | 1x256x1152, 1152 -> 1x256x1152 float32 | 81 | 0.119 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.q_proj` | `modeling_siglip.py:forward:385` | `aten.add.Tensor` bias | 1x256x1152, 1152x1152 -> 1x256x1152 float32 | 81 | 0.024 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.k_proj` | `modeling_siglip.py:forward:386` | `aten.add.Tensor` bias | 1x256x1152, 1152x1152 -> 1x256x1152 float32 | 81 | 0.024 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.v_proj` | `modeling_siglip.py:forward:387` | `aten.add.Tensor` bias | 1x256x1152, 1152x1152 -> 1x256x1152 float32 | 81 | 0.024 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:389` | `aten.view.default` | 1x256x1152 -> 1x256x16x72 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:389` | `aten.transpose.int` | 1x256x16x72 -> 1x16x256x72 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:390` | `aten.view.default` | 1x256x1152 -> 1x256x16x72 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:390` | `aten.transpose.int` | 1x256x16x72 -> 1x16x256x72 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:391` | `aten.view.default` | 1x256x1152 -> 1x256x16x72 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:391` | `aten.transpose.int` | 1x256x16x72 -> 1x16x256x72 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:397` | `aten.transpose.int` | 1x16x256x72 -> 1x16x72x256 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:397` | `aten.matmul.default` | 1x16x256x72, 1x16x72x256 -> 1x16x256x256 float32 | 81 | 12.231 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:397` | `aten.mul.Tensor` | 1x16x256x256 -> 1x16x256x256 float32 | 81 | 0.085 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:397` | `aten.softmax.int` | 1x16x256x256 -> 1x16x256x256 float32 | 81 | 0.425 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:397` | `aten.to.dtype` | 1x16x256x256 -> 1x16x256x256 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:397` | `aten.matmul.default` | 1x16x256x256, 1x16x256x72 -> 1x16x256x72 float32 | 81 | 12.231 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:397` | `aten.transpose.int` | 1x16x256x72 -> 1x256x16x72 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:397` | `aten.contiguous.default` | 1x256x16x72 -> 1x256x16x72 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn` | `modeling_siglip.py:forward:408` | `aten.reshape.default` | 1x256x16x72 -> 1x256x1152 float32 | 81 | 0.000 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.out_proj` | `modeling_siglip.py:forward:409` | `aten.add.Tensor` bias | 1x256x1152, 1152x1152 -> 1x256x1152 float32 | 81 | 0.024 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*` | `modeling_siglip.py:forward:454` | `aten.add.Tensor` | 1x256x1152, 1x256x1152 -> 1x256x1152 float32 | 81 | 0.024 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.layer_norm2` | `modeling_siglip.py:forward:457` | `aten.layer_norm.default` | 1x256x1152, 1152 -> 1x256x1152 float32 | 81 | 0.119 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.mlp.fc1` | `modeling_siglip.py:forward:424` | `aten.add.Tensor` bias | 1x256x1152, 4304x1152 -> 1x256x4304 float32 | 81 | 0.089 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.mlp.activation_fn` | `activations.py:forward:51` | `aten.gelu.default` | 1x256x4304 -> 1x256x4304 float32 | 81 | 0.089 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.mlp.fc2` | `modeling_siglip.py:forward:426` | `aten.add.Tensor` bias | 1x256x4304, 1152x4304 -> 1x256x1152 float32 | 81 | 0.024 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*` | `modeling_siglip.py:forward:459` | `aten.add.Tensor` | 1x256x1152, 1x256x1152 -> 1x256x1152 float32 | 81 | 0.024 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm` | `modeling_siglip.py:forward:707` | `aten.layer_norm.default` | 1x256x1152, 1152 -> 1x256x1152 float32 | 3 | 0.004 |
| prefix | `model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear` | `modeling_paligemma.py:forward:95` | `aten.add.Tensor` bias | 1x256x1152, 2048x1152 -> 1x256x2048 float32 | 3 | 0.002 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.mul.Tensor` | 1x256x2048 -> 1x256x2048 float32 | 3 | 0.002 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1 -> 1x1 bool | 3 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.expand.default` | 1x1 -> 1x256 bool | 3 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.embed_tokens` | `loader.py:forward:71` | `aten.embedding.default` | 257152x2048, 1x48 -> 1x48x2048 float32 | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.mul.Tensor` | 1x48x2048 -> 1x48x2048 float32 | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.cat.default` | 1x256x2048, 1x256x2048 -> 1x816x2048 float32 | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.cat.default` | 1x256, 1x256 -> 1x816 bool | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 816 -> 1x816 bool | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.expand.default` | 1x816 -> 1x816 bool | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.cumsum.default` | 1x816 -> 1x816 int64 | 2 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x816 -> 1x1x816 int64 | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x816 -> 1x816x1 int64 | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.le.Tensor` | 1x1x816, 1x816x1 -> 1x816x816 bool | 1 | 0.001 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x816 -> 1x1x816 bool | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x816 -> 1x816x1 bool | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.mul.Tensor` | 1x1x816, 1x816x1 -> 1x816x816 bool | 1 | 0.001 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.__and__.Tensor` | 1x816x816, 1x816x816 -> 1x816x816 bool | 1 | 0.001 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.sub.Tensor` | 1x816 -> 1x816 int64 | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x816x816 -> 1x1x816x816 bool | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.where.Scalar` | 1x1x816x816 -> 1x1x816x816 float32 | 1 | 0.001 |
| prefix | `(top)` | `pi_gemma.py:forward:254` | `aten.arange.start` | - -> 816 int64 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.unsqueeze.default` | 128 -> 1x128 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.unsqueeze.default` | 1x128 -> 1x128x1 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.to.dtype` | 1x128x1 -> 1x128x1 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.expand.default` | 1x128x1 -> 1x128x1 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.to.dtype_layout` | 1x128x1 -> 1x128x1 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:106` | `aten.unsqueeze.default` | 1x816 -> 1x1x816 int64 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:106` | `aten.to.dtype` | 1x1x816 -> 1x1x816 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:110` | `aten.to.dtype` | 1x128x1 -> 1x128x1 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:110` | `aten.to.dtype` | 1x1x816 -> 1x1x816 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:110` | `aten.matmul.default` | 1x128x1, 1x1x816 -> 1x128x816 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:110` | `aten.transpose.int` | 1x128x816 -> 1x816x128 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:111` | `aten.cat.default` | 1x816x128, 1x816x128 -> 1x816x256 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:112` | `aten.cos.default` | 1x816x256 -> 1x816x256 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:112` | `aten.mul.Tensor` | 1x816x256 -> 1x816x256 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:113` | `aten.sin.default` | 1x816x256 -> 1x816x256 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:113` | `aten.mul.Tensor` | 1x816x256 -> 1x816x256 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.rotary_emb` | `modeling_gemma.py:forward:115` | `aten.to.dtype` | 1x816x256 -> 1x816x256 float32 | 2 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.to.dtype` | 1x816x2048 -> 1x816x2048 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.square.default` | 1x816x2048 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.mean.dim` | 1x816x2048 -> 1x816x1 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.add.Tensor` | 1x816x1 -> 1x816x1 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.rsqrt.default` | 1x816x1 -> 1x816x1 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.mul.Tensor` | 1x816x2048, 1x816x1 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:119` | `aten.to.dtype` | 2048 -> 2048 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:119` | `aten.add.Tensor` | 2048 -> 2048 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:119` | `aten.mul.Tensor` | 1x816x2048, 2048 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.input_layernorm` | `pi_gemma.py:forward:120` | `aten.type_as.default` | 1x816x2048, 1x816x2048 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:229` | `aten.view.default` | 1x816x2048 -> 1x816x8x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:229` | `aten.transpose.int` | 1x816x8x256 -> 1x8x816x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:230` | `aten.view.default` | 1x816x256 -> 1x816x1x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:230` | `aten.transpose.int` | 1x816x1x256 -> 1x1x816x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:231` | `aten.view.default` | 1x816x256 -> 1x816x1x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:231` | `aten.transpose.int` | 1x816x1x256 -> 1x1x816x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.unsqueeze.default` | 1x816x256 -> 1x1x816x256 float32 | 36 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.mul.Tensor` | 1x8x816x256, 1x1x816x256 -> 1x8x816x256 float32 | 36 | 0.060 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.slice.Tensor` | 1x8x816x256 -> 1x8x816x128 float32 | 36 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.neg.default` | 1x8x816x128 -> 1x8x816x128 float32 | 18 | 0.015 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.cat.default` | 1x8x816x128, 1x8x816x128 -> 1x8x816x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.add.Tensor` | 1x8x816x256, 1x8x816x256 -> 1x8x816x256 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.mul.Tensor` | 1x1x816x256, 1x1x816x256 -> 1x1x816x256 float32 | 36 | 0.008 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.slice.Tensor` | 1x1x816x256 -> 1x1x816x128 float32 | 36 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.neg.default` | 1x1x816x128 -> 1x1x816x128 float32 | 18 | 0.002 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.cat.default` | 1x1x816x128, 1x1x816x128 -> 1x1x816x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.add.Tensor` | 1x1x816x256, 1x1x816x256 -> 1x1x816x256 float32 | 18 | 0.004 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:239` | `aten.cat.default` | 0, 1x1x816x256 -> 1x1x816x256 float32 | 36 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.unsqueeze.default` | 1x1x816x256 -> 1x1x1x816x256 float32 | 36 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.expand.default` | 1x1x1x816x256 -> 1x1x8x816x256 float32 | 36 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.reshape.default` | 1x1x8x816x256 -> 1x8x816x256 float32 | 36 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.transpose.int` | 1x8x816x256 -> 1x8x256x816 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.matmul.default` | 1x8x816x256, 1x8x256x816 -> 1x8x816x816 float32 | 18 | 49.092 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.mul.Tensor` | 1x8x816x816 -> 1x8x816x816 float32 | 18 | 0.096 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.add.Tensor` | 1x8x816x816, 1x1x816x816 -> 1x8x816x816 float32 | 18 | 0.096 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.softmax.int` | 1x8x816x816 -> 1x8x816x816 float32 | 18 | 0.479 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.to.dtype` | 1x8x816x816 -> 1x8x816x816 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.matmul.default` | 1x8x816x816, 1x8x816x256 -> 1x8x816x256 float32 | 18 | 49.092 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.transpose.int` | 1x8x816x256 -> 1x816x8x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.contiguous.default` | 1x816x8x256 -> 1x816x8x256 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn` | `modeling_gemma.py:forward:256` | `aten.reshape.default` | 1x816x8x256 -> 1x816x2048 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*` | `pi_gemma.py:forward:182` | `aten.add.Tensor` | 1x816x2048, 1x816x2048 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.to.dtype` | 1x816x2048 -> 1x816x2048 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.square.default` | 1x816x2048 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.mean.dim` | 1x816x2048 -> 1x816x1 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.add.Tensor` | 1x816x1 -> 1x816x1 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.rsqrt.default` | 1x816x1 -> 1x816x1 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.mul.Tensor` | 1x816x2048, 1x816x1 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:119` | `aten.to.dtype` | 2048 -> 2048 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:119` | `aten.add.Tensor` | 2048 -> 2048 float32 | 18 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:119` | `aten.mul.Tensor` | 1x816x2048, 2048 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:120` | `aten.type_as.default` | 1x816x2048, 1x816x2048 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.mlp.act_fn` | `activations.py:forward:51` | `aten.gelu.default` | 1x816x16384 -> 1x816x16384 float32 | 18 | 0.241 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.mlp` | `modeling_gemma.py:forward:78` | `aten.mul.Tensor` | 1x816x16384, 1x816x16384 -> 1x816x16384 float32 | 18 | 0.241 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*` | `pi_gemma.py:forward:187` | `aten.add.Tensor` | 1x816x2048, 1x816x2048 -> 1x816x2048 float32 | 18 | 0.030 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:117` | `aten.to.dtype` | 1x816x2048 -> 1x816x2048 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:117` | `aten.square.default` | 1x816x2048 -> 1x816x2048 float32 | 1 | 0.002 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:117` | `aten.mean.dim` | 1x816x2048 -> 1x816x1 float32 | 1 | 0.002 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:117` | `aten.add.Tensor` | 1x816x1 -> 1x816x1 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:117` | `aten.rsqrt.default` | 1x816x1 -> 1x816x1 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:117` | `aten.mul.Tensor` | 1x816x2048, 1x816x1 -> 1x816x2048 float32 | 1 | 0.002 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:119` | `aten.to.dtype` | 2048 -> 2048 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:119` | `aten.add.Tensor` | 2048 -> 2048 float32 | 1 | 0.000 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:119` | `aten.mul.Tensor` | 1x816x2048, 2048 -> 1x816x2048 float32 | 1 | 0.002 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.norm` | `pi_gemma.py:forward:120` | `aten.type_as.default` | 1x816x2048, 1x816x2048 -> 1x816x2048 float32 | 1 | 0.002 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.expand.default` | scalar -> 1 float32 | 1 | 0.000 |
| prefix | `(top)` | `loader.py:forward:71` | `aten.to.dtype` | 1x32 -> 1x32 float32 | 1 | 0.000 |
| denoise | `model.state_proj` | `loader.py:forward:71` | `aten.add.Tensor` bias | 1x32, 1024x32 -> 1x1024 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x1024 -> 1x1x1024 float32 | 20 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.ones.default` | - -> 1x1 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.linspace.default` | - -> 512 float64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.pow.Scalar` | 512 -> 512 float64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.mul.Tensor` | 512 -> 512 float64 | 40 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.reciprocal.default` | 512 -> 512 float64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 512 -> 1x512 float64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1 -> 1x1 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.mul.Tensor` | 1x512, 1x1 -> 1x512 float64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.sin.default` | 1x512 -> 1x512 float64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.cos.default` | 1x512 -> 1x512 float64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.cat.default` | 1x512, 1x512 -> 1x1024 float64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.to.dtype` | 1x1024 -> 1x1024 float32 | 10 | 0.000 |
| denoise | `model.action_in_proj` | `loader.py:forward:71` | `aten.add.Tensor` bias | 1x50x32, 1024x32 -> 1x50x1024 float32 | 10 | 0.001 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.expand_as.default` | 1x1x1024, 1x50x1024 -> 1x50x1024 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.cat.default` | 1x50x1024, 1x50x1024 -> 1x50x2048 float32 | 10 | 0.000 |
| denoise | `model.action_time_mlp_in` | `loader.py:forward:71` | `aten.add.Tensor` bias | 1x50x2048, 1024x2048 -> 1x50x1024 float32 | 10 | 0.001 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.silu.default` | 1x50x1024 -> 1x50x1024 float32 | 10 | 0.001 |
| denoise | `model.action_time_mlp_out` | `loader.py:forward:71` | `aten.add.Tensor` bias | 1x50x1024, 1024x1024 -> 1x50x1024 float32 | 10 | 0.001 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.ones.default` | - -> 1x50 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.cat.default` | 1x1x1024, 1x50x1024 -> 1x51x1024 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.cat.default` | 1x1, 1x50 -> 1x51 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 51 -> 1x51 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.expand.default` | 1x51 -> 1x51 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x816 -> 1x1x816 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.expand.default` | 1x1x816 -> 1x51x816 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.cumsum.default` | 1x51 -> 1x51 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x51 -> 1x1x51 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x51 -> 1x51x1 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.le.Tensor` | 1x1x51, 1x51x1 -> 1x51x51 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x51 -> 1x1x51 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x51 -> 1x51x1 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.mul.Tensor` | 1x1x51, 1x51x1 -> 1x51x51 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.__and__.Tensor` | 1x51x51, 1x51x51 -> 1x51x51 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.cat.default` | 1x51x816, 1x51x51 -> 1x51x867 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.sum.dim_IntList` | 1x816 -> 1 int64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1 -> 1x1 int64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.cumsum.default` | 1x51 -> 1x51 int64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.add.Tensor` | 1x1, 1x51 -> 1x51 int64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.sub.Tensor` | 1x51 -> 1x51 int64 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.unsqueeze.default` | 1x51x867 -> 1x1x51x867 bool | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.where.Scalar` | 1x1x51x867 -> 1x1x51x867 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.clone.default` | 1x1x816x256 -> 1x1x816x256 float32 | 360 | 0.000 |
| denoise | `(top)` | `pi_gemma.py:forward:254` | `aten.arange.start` | - -> 51 int64 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.unsqueeze.default` | 128 -> 1x128 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.unsqueeze.default` | 1x128 -> 1x128x1 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.to.dtype` | 1x128x1 -> 1x128x1 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.expand.default` | 1x128x1 -> 1x128x1 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:105` | `aten.to.dtype_layout` | 1x128x1 -> 1x128x1 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:106` | `aten.unsqueeze.default` | 1x51 -> 1x1x51 int64 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:106` | `aten.to.dtype` | 1x1x51 -> 1x1x51 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:110` | `aten.to.dtype` | 1x128x1 -> 1x128x1 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:110` | `aten.to.dtype` | 1x1x51 -> 1x1x51 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:110` | `aten.matmul.default` | 1x128x1, 1x1x51 -> 1x128x51 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:110` | `aten.transpose.int` | 1x128x51 -> 1x51x128 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:111` | `aten.cat.default` | 1x51x128, 1x51x128 -> 1x51x256 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:112` | `aten.cos.default` | 1x51x256 -> 1x51x256 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:112` | `aten.mul.Tensor` | 1x51x256 -> 1x51x256 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:113` | `aten.sin.default` | 1x51x256 -> 1x51x256 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:113` | `aten.mul.Tensor` | 1x51x256 -> 1x51x256 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.rotary_emb` | `modeling_gemma.py:forward:115` | `aten.to.dtype` | 1x51x256 -> 1x51x256 float32 | 20 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.to.dtype` | 1x51x1024 -> 1x51x1024 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.square.default` | 1x51x1024 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.mean.dim` | 1x51x1024 -> 1x51x1 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.add.Tensor` | 1x51x1 -> 1x51x1 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.rsqrt.default` | 1x51x1 -> 1x51x1 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:117` | `aten.mul.Tensor` | 1x51x1024, 1x51x1 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:119` | `aten.to.dtype` | 1024 -> 1024 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:119` | `aten.add.Tensor` | 1024 -> 1024 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:119` | `aten.mul.Tensor` | 1x51x1024, 1024 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.input_layernorm` | `pi_gemma.py:forward:120` | `aten.type_as.default` | 1x51x1024, 1x51x1024 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:229` | `aten.view.default` | 1x51x2048 -> 1x51x8x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:229` | `aten.transpose.int` | 1x51x8x256 -> 1x8x51x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:230` | `aten.view.default` | 1x51x256 -> 1x51x1x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:230` | `aten.transpose.int` | 1x51x1x256 -> 1x1x51x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:231` | `aten.view.default` | 1x51x256 -> 1x51x1x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:231` | `aten.transpose.int` | 1x51x1x256 -> 1x1x51x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.unsqueeze.default` | 1x51x256 -> 1x1x51x256 float32 | 360 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.mul.Tensor` | 1x8x51x256, 1x1x51x256 -> 1x8x51x256 float32 | 360 | 0.038 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.slice.Tensor` | 1x8x51x256 -> 1x8x51x128 float32 | 360 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.neg.default` | 1x8x51x128 -> 1x8x51x128 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.cat.default` | 1x8x51x128, 1x8x51x128 -> 1x8x51x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.add.Tensor` | 1x8x51x256, 1x8x51x256 -> 1x8x51x256 float32 | 180 | 0.019 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.mul.Tensor` | 1x1x51x256, 1x1x51x256 -> 1x1x51x256 float32 | 360 | 0.005 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.slice.Tensor` | 1x1x51x256 -> 1x1x51x128 float32 | 360 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.neg.default` | 1x1x51x128 -> 1x1x51x128 float32 | 180 | 0.001 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.cat.default` | 1x1x51x128, 1x1x51x128 -> 1x1x51x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:234` | `aten.add.Tensor` | 1x1x51x256, 1x1x51x256 -> 1x1x51x256 float32 | 180 | 0.002 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:239` | `aten.cat.default` | 1x1x816x256, 1x1x51x256 -> 1x1x867x256 float32 | 360 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.unsqueeze.default` | 1x1x867x256 -> 1x1x1x867x256 float32 | 360 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.expand.default` | 1x1x1x867x256 -> 1x1x8x867x256 float32 | 360 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.reshape.default` | 1x1x8x867x256 -> 1x8x867x256 float32 | 360 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.transpose.int` | 1x8x867x256 -> 1x8x256x867 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.matmul.default` | 1x8x51x256, 1x8x256x867 -> 1x8x51x867 float32 | 180 | 32.600 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.mul.Tensor` | 1x8x51x867 -> 1x8x51x867 float32 | 180 | 0.064 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.add.Tensor` | 1x8x51x867, 1x1x51x867 -> 1x8x51x867 float32 | 180 | 0.064 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.softmax.int` | 1x8x51x867 -> 1x8x51x867 float32 | 180 | 0.318 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.to.dtype` | 1x8x51x867 -> 1x8x51x867 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.matmul.default` | 1x8x51x867, 1x8x867x256 -> 1x8x51x256 float32 | 180 | 32.600 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.transpose.int` | 1x8x51x256 -> 1x51x8x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:245` | `aten.contiguous.default` | 1x51x8x256 -> 1x51x8x256 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn` | `modeling_gemma.py:forward:256` | `aten.reshape.default` | 1x51x8x256 -> 1x51x2048 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*` | `pi_gemma.py:forward:182` | `aten.add.Tensor` | 1x51x1024, 1x51x1024 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.to.dtype` | 1x51x1024 -> 1x51x1024 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.square.default` | 1x51x1024 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.mean.dim` | 1x51x1024 -> 1x51x1 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.add.Tensor` | 1x51x1 -> 1x51x1 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.rsqrt.default` | 1x51x1 -> 1x51x1 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:117` | `aten.mul.Tensor` | 1x51x1024, 1x51x1 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:119` | `aten.to.dtype` | 1024 -> 1024 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:119` | `aten.add.Tensor` | 1024 -> 1024 float32 | 180 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:119` | `aten.mul.Tensor` | 1x51x1024, 1024 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.post_attention_layernorm` | `pi_gemma.py:forward:120` | `aten.type_as.default` | 1x51x1024, 1x51x1024 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.mlp.act_fn` | `activations.py:forward:51` | `aten.gelu.default` | 1x51x4096 -> 1x51x4096 float32 | 180 | 0.038 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.mlp` | `modeling_gemma.py:forward:78` | `aten.mul.Tensor` | 1x51x4096, 1x51x4096 -> 1x51x4096 float32 | 180 | 0.038 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*` | `pi_gemma.py:forward:187` | `aten.add.Tensor` | 1x51x1024, 1x51x1024 -> 1x51x1024 float32 | 180 | 0.009 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:117` | `aten.to.dtype` | 1x51x1024 -> 1x51x1024 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:117` | `aten.square.default` | 1x51x1024 -> 1x51x1024 float32 | 10 | 0.001 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:117` | `aten.mean.dim` | 1x51x1024 -> 1x51x1 float32 | 10 | 0.001 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:117` | `aten.add.Tensor` | 1x51x1 -> 1x51x1 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:117` | `aten.rsqrt.default` | 1x51x1 -> 1x51x1 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:117` | `aten.mul.Tensor` | 1x51x1024, 1x51x1 -> 1x51x1024 float32 | 10 | 0.001 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:119` | `aten.to.dtype` | 1024 -> 1024 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:119` | `aten.add.Tensor` | 1024 -> 1024 float32 | 10 | 0.000 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:119` | `aten.mul.Tensor` | 1x51x1024, 1024 -> 1x51x1024 float32 | 10 | 0.001 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.norm` | `pi_gemma.py:forward:120` | `aten.type_as.default` | 1x51x1024, 1x51x1024 -> 1x51x1024 float32 | 10 | 0.001 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.slice.Tensor` | 1x51x1024 -> 1x50x1024 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.to.dtype` | 1x50x1024 -> 1x50x1024 float32 | 10 | 0.000 |
| denoise | `model.action_out_proj` | `loader.py:forward:71` | `aten.add.Tensor` bias | 1x50x1024, 32x1024 -> 1x50x32 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.mul.Tensor` | 1x50x32 -> 1x50x32 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.add.Tensor` | 1x50x32, 1x50x32 -> 1x50x32 float32 | 10 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.expand.default` | scalar -> 1 float32 | 9 | 0.000 |
| denoise | `(top)` | `loader.py:forward:71` | `aten.to.dtype` | 1x32 -> 1x32 float32 | 9 | 0.000 |

| where | op instances | GFLOP |
| --- | ---: | ---: |
| `atlas:projection_gemm` | 1,928 | 4,191.248 |

| phase | module | op | shape | count | where | GFLOP |
| --- | --- | --- | --- | ---: | --- | ---: |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding` | `conv2d.padding` | 1x3x224x224, 1152x3x14x14 -> 1x1152x16x16 float32 | 3 | `atlas:projection_gemm` | 1.040 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.q_proj` | `linear.default` | 1x256x1152, 1152x1152 -> 1x256x1152 float32 | 81 | `atlas:projection_gemm` | 55.038 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.k_proj` | `linear.default` | 1x256x1152, 1152x1152 -> 1x256x1152 float32 | 81 | `atlas:projection_gemm` | 55.038 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.v_proj` | `linear.default` | 1x256x1152, 1152x1152 -> 1x256x1152 float32 | 81 | `atlas:projection_gemm` | 55.038 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.self_attn.out_proj` | `linear.default` | 1x256x1152, 1152x1152 -> 1x256x1152 float32 | 81 | `atlas:projection_gemm` | 55.038 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.mlp.fc1` | `linear.default` | 1x256x1152, 4304x1152 -> 1x256x4304 float32 | 81 | `atlas:projection_gemm` | 205.627 |
| vision | `model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.*.mlp.fc2` | `linear.default` | 1x256x4304, 1152x4304 -> 1x256x1152 float32 | 81 | `atlas:projection_gemm` | 205.627 |
| prefix | `model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear` | `linear.default` | 1x256x1152, 2048x1152 -> 1x256x2048 float32 | 3 | `atlas:projection_gemm` | 3.624 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.q_proj` | `linear.default` | 1x816x2048, 2048x2048 -> 1x816x2048 float32 | 18 | `atlas:projection_gemm` | 123.212 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.k_proj` | `linear.default` | 1x816x2048, 256x2048 -> 1x816x256 float32 | 18 | `atlas:projection_gemm` | 15.401 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.v_proj` | `linear.default` | 1x816x2048, 256x2048 -> 1x816x256 float32 | 18 | `atlas:projection_gemm` | 15.401 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.self_attn.o_proj` | `linear.default` | 1x816x2048, 2048x2048 -> 1x816x2048 float32 | 18 | `atlas:projection_gemm` | 123.212 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.mlp.gate_proj` | `linear.default` | 1x816x2048, 16384x2048 -> 1x816x16384 float32 | 18 | `atlas:projection_gemm` | 985.695 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.mlp.up_proj` | `linear.default` | 1x816x2048, 16384x2048 -> 1x816x16384 float32 | 18 | `atlas:projection_gemm` | 985.695 |
| prefix | `model.paligemma_with_expert.paligemma.model.language_model.layers.*.mlp.down_proj` | `linear.default` | 1x816x16384, 2048x16384 -> 1x816x2048 float32 | 18 | `atlas:projection_gemm` | 985.695 |
| denoise | `model.state_proj` | `linear.default` | 1x32, 1024x32 -> 1x1024 float32 | 10 | `atlas:projection_gemm` | 0.001 |
| denoise | `model.action_in_proj` | `linear.default` | 1x50x32, 1024x32 -> 1x50x1024 float32 | 10 | `atlas:projection_gemm` | 0.033 |
| denoise | `model.action_time_mlp_in` | `linear.default` | 1x50x2048, 1024x2048 -> 1x50x1024 float32 | 10 | `atlas:projection_gemm` | 2.097 |
| denoise | `model.action_time_mlp_out` | `linear.default` | 1x50x1024, 1024x1024 -> 1x50x1024 float32 | 10 | `atlas:projection_gemm` | 1.049 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn.q_proj` | `linear.default` | 1x51x1024, 2048x1024 -> 1x51x2048 float32 | 180 | `atlas:projection_gemm` | 38.504 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn.k_proj` | `linear.default` | 1x51x1024, 256x1024 -> 1x51x256 float32 | 180 | `atlas:projection_gemm` | 4.813 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn.v_proj` | `linear.default` | 1x51x1024, 256x1024 -> 1x51x256 float32 | 180 | `atlas:projection_gemm` | 4.813 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.self_attn.o_proj` | `linear.default` | 1x51x2048, 1024x2048 -> 1x51x1024 float32 | 180 | `atlas:projection_gemm` | 38.504 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.mlp.gate_proj` | `linear.default` | 1x51x1024, 4096x1024 -> 1x51x4096 float32 | 180 | `atlas:projection_gemm` | 77.007 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.mlp.up_proj` | `linear.default` | 1x51x1024, 4096x1024 -> 1x51x4096 float32 | 180 | `atlas:projection_gemm` | 77.007 |
| denoise | `model.paligemma_with_expert.gemma_expert.model.layers.*.mlp.down_proj` | `linear.default` | 1x51x4096, 1024x4096 -> 1x51x1024 float32 | 180 | `atlas:projection_gemm` | 77.007 |
| denoise | `model.action_out_proj` | `linear.default` | 1x50x1024, 32x1024 -> 1x50x32 float32 | 10 | `atlas:projection_gemm` | 0.033 |

Zero-FLOP records (views, casts, copies, and index construction):

| phase | where | ops |
| --- | --- | --- |
