"""Resident FP8 attention for 128- and 256-wide heads.

Independent entry point; the pinned dimension-64 generators stay unchanged.
K and V occupy one 256 KiB VMEM bank each. Two MXU accumulators produce
one 64-column output panel at a time, reusing the resident probabilities.
"""
from dataclasses import dataclass
import math
import struct

from atlas_attention_v2_program import _bf16, tile_masks, check_program_v2


def bf16_scale(value):
    """Round a positive FP32 attention scale to nearest-even BF16."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@dataclass(frozen=True)
class AttentionWideConfig:
    query_rows: int
    keys: int
    query_heads: int
    kv_heads: int
    head_dim: int = 128
    mask: str = "none"
    query_start: int | None = None
    scale: float | None = None
    q_exponent: int = 0
    k_exponent: int = 0
    v_exponent: int = 0
    p_exponent: int = -8
    output_fp8: bool = False
    o_exponent: int = 0
    packed_decode: bool = False

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in
               (self.query_rows, self.keys, self.query_heads, self.kv_heads)):
            raise ValueError("attention dimensions must be positive integers")
        if self.head_dim not in (128, 256) or self.query_heads % self.kv_heads:
            raise ValueError("require head dimension 128 or 256 and integral GQA")
        if self.mask not in ("none", "causal"):
            raise ValueError("wide attention supports none and causal masks")
        if self.query_start is not None and (type(self.query_start) is not int or self.query_start < 0):
            raise ValueError("query_start must be a nonnegative integer")
        if self.mask == "causal" and self.query_start is None and self.keys < self.query_rows:
            raise ValueError("causal queries require keys >= rows or explicit query_start")
        scale = self.head_dim ** -0.5 if self.scale is None else self.scale
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("attention scale must be positive and finite")
        object.__setattr__(self, "scale", bf16_scale(scale))
        if self.p_exponent != -8:
            raise ValueError("wide attention uses fixed P scale 256")
        for value in (self.q_exponent, self.k_exponent, self.v_exponent, self.o_exponent):
            if type(value) is not int or not -60 <= value <= 60:
                raise ValueError("FP8 exponent must be an integer in -60..60")
        _bf16(self.scale * 2.0 ** (self.q_exponent + self.k_exponent))
        if self.key_tiles * self.head_dim * 32 > 256 * 1024:
            raise ValueError("resident KV exceeds one 256 KiB VMEM bank per operand")
        if self.query_tiles > 128 or self.query_tiles * self.key_tiles > 2048:
            raise ValueError("query tiles or descriptor table exceeds reserved VMEM")
        if self.packed_decode:
            raise ValueError("wide attention uses ordinary tiled decode, packed_decode must be False")

    @property
    def query_tiles(self):
        return (self.query_rows + 31) // 32

    @property
    def key_tiles(self):
        return (self.keys + 31) // 32

    @property
    def decode_packed(self):
        return False


def pack_tiles_wide(value):
    """BF16 [heads, rows, dim], tiled by row block, head, and 16 columns."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        value = value.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
    value = np.asarray(value)
    if value.dtype != np.uint16 or value.ndim != 3 or value.shape[2] not in (128, 256):
        raise ValueError("expected BF16 words [heads, rows, 128 or 256]")
    heads, rows, dim = value.shape
    padded = np.zeros((heads, (rows + 31) // 32 * 32, dim), dtype="<u2")
    padded[:, :rows] = value
    return b"".join(padded[h, r:r + 32, d:d + 16].tobytes()
                    for r in range(0, padded.shape[1], 32)
                    for h in range(heads) for d in range(0, dim, 16))


def pack_output_wide(config, value):
    import numpy as np
    if not config.output_fp8:
        return pack_tiles_wide(value)
    value = np.asarray(value, dtype=np.uint8)
    if value.shape != (config.query_heads, config.query_rows, config.head_dim):
        raise ValueError("output shape differs from attention configuration")
    padded = np.zeros((config.query_heads, config.query_tiles * 32, config.head_dim), dtype=np.uint8)
    padded[:, :config.query_rows] = value
    return b"".join(padded[h, r:r + 32, d:d + 32].tobytes()
                    for r in range(0, padded.shape[1], 32)
                    for h in range(config.query_heads) for d in range(0, config.head_dim, 32))


def unpack_output_wide(config, data):
    import numpy as np
    width = 32 if config.output_fp8 else 16
    dtype = np.uint8 if config.output_fp8 else "<u2"
    value = np.frombuffer(data, dtype=dtype).reshape(
        config.query_tiles, config.query_heads, config.head_dim // width, 32, width)
    return value.transpose(1, 0, 3, 2, 4).reshape(
        config.query_heads, config.query_tiles * 32, config.head_dim)[:, :config.query_rows].copy()


def stage_wide(config, q, k, v, mask_tiles=None):
    """Deduplicate masks and stage ordinary tiled query heads, including decode."""
    import numpy as np
    import torch

    c = config
    allowed = tile_masks(c, mask_tiles)
    for value, shape in zip((q, k, v), ((c.query_heads, c.query_rows, c.head_dim),
        (c.kv_heads, c.keys, c.head_dim), (c.kv_heads, c.keys, c.head_dim))):
        if tuple(value.shape) != shape:
            raise ValueError("input tensor shapes differ from attention configuration")
    qbytes = pack_tiles_wide(q)
    parts = [qbytes, pack_tiles_wide(k), pack_tiles_wide(v)]
    offsets = {"q": 0, "k": len(parts[0]), "v": len(parts[0]) + len(parts[1])}
    arena = bytearray(b"".join(parts))
    offsets["descriptors"] = len(arena)
    descriptors, masks, unique, counts = [], bytearray(), {}, []
    for row in allowed:
        visible = [index for index, tile in enumerate(row) if tile.any()]
        counts.append(visible[-1] + 1 if visible else 0)
        for tile in row:
            if not tile.any():
                descriptors.append((0, 1))
            elif tile.all():
                descriptors.append((0, 0))
            else:
                mask = np.where(tile, 0, 0x7f80).astype("<u2")
                blob = mask[:, :16].tobytes() + mask[:, 16:].tobytes()
                if blob not in unique:
                    unique[blob] = 0x4400 + len(masks) // 4
                    masks.extend(blob)
                descriptors.append((unique[blob], 0))
    table = b"".join(struct.pack("<II", *entry) for entry in descriptors)
    offsets["descriptor_bytes"] = (len(table) + 31) // 32 * 32
    arena.extend(table + bytes(offsets["descriptor_bytes"] - len(table)))
    offsets["key_counts"] = len(arena)
    counts_blob = struct.pack("<" + "I" * len(counts), *counts)
    offsets["key_count_bytes"] = (len(counts_blob) + 31) // 32 * 32
    arena.extend(counts_blob + bytes(offsets["key_count_bytes"] - len(counts_blob)))
    offsets["masks"] = len(arena)
    offsets["mask_bytes"] = len(masks)
    offsets["unique_masks"] = len(unique)
    arena.extend(masks)
    offsets["output"] = len(arena) + 256
    offsets["output_bytes"] = c.query_tiles * c.query_heads * (c.head_dim * (32 if c.output_fp8 else 64))
    arena.extend(b"\xa5" * 256 + bytes(offsets["output_bytes"]) + b"\x5a" * 256)
    if offsets["descriptor_bytes"] > 16384 or offsets["key_count_bytes"] > 4096 or len(masks) > 192512:
        raise ValueError("resident descriptors or deduplicated masks exceed bank zero")
    return bytes(arena), offsets


def query_key_counts(config):
    """Static visible key-tile bounds for ordinary none/causal query blocks."""
    c = config
    if c.mask == "none":
        return (c.key_tiles,) * c.query_tiles
    start = c.keys - c.query_rows if c.query_start is None else c.query_start
    return tuple(min(c.key_tiles, max(0, start + min((tile + 1) * 32, c.query_rows) + 31) // 32)
                 for tile in range(c.query_tiles))


def assembly_wide(config, offsets, *, dram_base=0x90000000):
    """One KV-head staging pass, followed by all its query heads and blocks."""
    c = config
    if dram_base % 32 or dram_base < 0 or dram_base + offsets["output"] + offsets["output_bytes"] >= 2**32:
        raise ValueError("arena must fit one 32-bit aligned DMA window")
    group = c.query_heads // c.kv_heads
    out_head = c.head_dim * (32 if c.output_fp8 else 64)
    head_bytes = c.head_dim * 64
    resident_words = c.head_dim * 8
    dims = c.head_dim // 32
    lines = []

    def emit(*items):
        lines.extend(items)

    def head_dma(vmem, dram, channel, *, store=False, count=head_bytes):
        # RTL permits at most 4096 bytes per DMA command. Complete every
        # intermediate segment; the caller owns the final segment's wait.
        result = []
        for offset in range(0, count, 4096):
            if offset:
                result.append(f"dma.wait {channel}")
            result.extend((f"li x1, {offset // 4}", f"add x1, {vmem}, x1",
                           f"li x28, {offset}", f"add x28, {dram}, x28",
                           f"li x9, {min(4096, count - offset)}"))
            result.append(f"dma.store x28, x1, x9, {channel}" if store else
                          f"dma.load x1, x28, x9, {channel}")
        return result

    def op(instruction, delay=66):
        emit(instruction, f"delay {delay}")

    def pair_load(reg, base, offset=0):
        op(f"vload m{reg}, {offset}({base})", 34)
        op(f"vload m{reg + 1}, {offset + 8}({base})", 34)

    def pair_store(reg, base, offset=0):
        op(f"vstore m{reg}, {offset}({base})", 34)
        op(f"vstore m{reg + 1}, {offset + 8}({base})", 34)

    def quantize(dest, exponent, base, offset=0):
        pair_load(12, base, offset)
        op(f"vli.all m28, {_bf16(448.0 * 2.0 ** exponent)}")
        op(f"vli.all m30, {_bf16(-448.0 * 2.0 ** exponent)}")
        op("vminimum.bf16 m12, m12, m28")
        op("vmaximum.bf16 m12, m12, m30")
        op("vmatpush.acc.bf16.mxu0 acc0, m12", 32)
        op(f"vmatpop.fp8.acc.mxu0 m{dest}, acc0, e{(c.q_exponent, c.k_exponent, c.v_exponent).index(exponent)}", 34)

    def key_start(label):
        emit("add x11, x18, x0", "add x12, x13, x0", "li x10, 0x10000",
            f"beq x11, x0, {label}_done", "nop", f"{label}:")

    def key_end(label, extra=()):
        emit(*extra, "addi x10, x10, 512", "addi x12, x12, 8", "addi x11, x11, -1",
            f"beq x11, x0, {label}_done", "nop", f"jal x0, {label}", "nop", f"{label}_done:")

    emit("dma.config x0, ch0", "dma.wait ch0", "dma.wait ch1", "li x29, 0x2000",
        f"li x21, {dram_base + offsets['k']}", f"li x22, {dram_base + offsets['v']}",
        f"li x19, {dram_base + offsets['q']}", f"li x20, {dram_base + offsets['output']}",
        f"li x15, {c.kv_heads}", f"li x23, {c.kv_heads * head_bytes}")
    for index, exponent in enumerate((c.q_exponent, c.k_exponent, c.v_exponent)):
        emit(f"seli e{index}, {127 - exponent}")
    emit("seli e3, 127")
    for key, size, target in (("descriptors", "descriptor_bytes", 0x3000),
        ("key_counts", "key_count_bytes", 0x4000), ("masks", "mask_bytes", 0x4400)):
        for offset in range(0, offsets[size], 4096):
            emit(f"li x27, {dram_base + offsets[key] + offset}", f"li x28, {target + offset // 4}",
                f"li x9, {min(4096, offsets[size] - offset)}", "dma.load x28, x27, x9, ch0", "dma.wait ch0")
    op("vli.all m24, 0")
    op(f"vli.all m8, {_bf16(c.scale * 2.0 ** (c.q_exponent + c.k_exponent))}")
    emit("kv_loop:", "li x2, 0x0", "li x3, 0x1000", "add x6, x21, x0", "add x7, x22, x0",
        "li x24, 0x20000", "li x25, 0x30000", f"li x11, {c.key_tiles}",
        *head_dma("x2", "x6", "ch0"), "dma.wait ch0", "stage_loop:", *head_dma("x3", "x7", "ch1"))
    for dim in range(dims):
        quantize(14, c.k_exponent, "x2", dim * 16)
        op(f"vstore m14, {dim * 8}(x24)", 34)
    emit("dma.wait ch1", "add x6, x6, x23", "add x7, x7, x23", "addi x11, x11, -1",
        "beq x11, x0, stage_last", "nop", *head_dma("x2", "x6", "ch1"), "stage_last:")
    for dim in range(dims):
        quantize(14, c.v_exponent, "x3", dim * 16)
        op("vtrpose.xlu m14, m14")
        op(f"vstore m14, {dim * 8}(x25)", 34)
    emit("dma.wait ch1", f"li x28, {resident_words}", "add x24, x24, x28", "add x25, x25, x28",
        "beq x11, x0, stage_done", "nop", "jal x0, stage_loop", "nop", "stage_done:",
        f"li x16, {1 if c.decode_packed else group}", "head_loop:",
        "add x4, x19, x0", "add x8, x20, x0", f"li x14, {c.query_tiles}",
        "li x13, 0xc000", "li x17, 0x10000", "li x2, 0x0", "li x3, 0x1000",
        *head_dma("x2", "x4", "ch1"), "query_loop:", "dma.wait ch1")
    emit("li x28, 1", "beq x14, x28, query_no_prefetch", "nop",
        f"li x28, {c.query_heads * head_bytes}", "add x27, x4, x28", *head_dma("x3", "x27", "ch1"),
        "query_no_prefetch:")
    # Runtime metadata must not bound resident-VMEM iteration. Both supported
    # mask geometries determine the bound statically; x14 is the concrete number
    # of query blocks remaining. Keep the descriptor skip/mask behavior below.
    counts = query_key_counts(c)
    emit(f"li x18, {counts[-1]}")
    for tile, count in enumerate(counts[:-1]):
        if count == counts[-1]:
            continue
        emit(f"li x28, {c.query_tiles - tile}", f"bne x14, x28, query_count_next_{tile}", "nop",
             f"li x18, {count}", "jal x0, query_count_ready", "nop", f"query_count_next_{tile}:")
    emit("query_count_ready:")
    op("vli.all m4, 65408")
    emit("li x6, 0x20000")
    key_start("scores_loop")
    emit("lw x27, 4(x12)", "delay 4", "bne x27, x0, scores_masked", "nop")
    op("vmatpush.acc.bf16.mxu1 acc0, m24", 32)
    for dim in range(dims):
        quantize(0, c.q_exponent, "x2", dim * 16)
        op(f"vload m14, {dim * 8}(x6)", 34)
        op("vmatpush.weight.mxu1 w0, m14", 32)
        op(f"vmatmul.acc.mxu1 acc0, m0, w0", 96)
    op("vmatpop.bf16.acc.mxu1 m16, acc0", 34)
    op("vmul.bf16 m16, m16, m8")
    emit("lw x27, 0(x12)", "delay 4", "beq x27, x0, scores_ready", "nop")
    pair_load(18, "x27")
    op("vsub.bf16 m16, m16, m18")
    emit("jal x0, scores_ready", "nop", "scores_masked:")
    op("vli.all m16, 65408")
    emit("scores_ready:")
    pair_store(16, "x10")
    op("vredmax.row.bf16 m22, m16", 39)
    op("vmaximum.bf16 m4, m4, m22")
    key_end("scores_loop", (f"li x28, {resident_words}", "add x6, x6, x28"))
    op("vli.all m26, 65407")
    op("vmaximum.bf16 m4, m4, m26")
    op("vli.all m26, 14392")  # Two FP8 +1 bytes per 16-bit immediate.
    op("vmatpush.weight.mxu1 w1, m26", 32)
    op("vmatpush.acc.bf16.mxu1 acc0, m24", 32)
    emit("li x24, 0x18000")
    key_start("exp_loop")
    emit("lw x27, 4(x12)", "delay 4", "bne x27, x0, exp_masked", "nop")
    pair_load(16, "x10")
    op("vsub.bf16 m20, m16, m4")
    op(f"vli.all m28, {_bf16(-128.0)}")
    op("vmaximum.bf16 m20, m20, m28")
    emit("lw x27, 0(x12)", "delay 4", "beq x27, x0, exp_unmasked", "nop")
    pair_load(18, "x27")
    op("vsub.bf16 m20, m20, m18")
    emit("exp_unmasked:")
    op("vexp.bf16 m20, m20")
    emit("jal x0, exp_ready", "nop", "exp_masked:")
    op("vli.all m20, 0")
    emit("exp_ready:")
    op(f"vli.all m28, {_bf16(256.0)}")
    op("vmul.bf16 m20, m20, m28")
    op("vmatpush.acc.bf16.mxu0 acc0, m20", 32)
    op("vmatpop.fp8.acc.mxu0 m22, acc0, e3", 34)
    op("vstore m22, 0(x24)", 34)
    op("vmatmul.acc.mxu1 acc0, m22, w1", 96)
    key_end("exp_loop", ("addi x24, x24, 256",))
    op("vmatpop.bf16.acc.mxu1 m6, acc0", 34)
    op("vli.all m28, 16256")
    op("vmaximum.bf16 m6, m6, m28")
    op("vrecip.bf16 m10, m6")
    op(f"vli.all m28, {_bf16(2.0 ** c.v_exponent)}")
    op("vmul.bf16 m10, m10, m28")
    for panel in range(0, dims, 2):
        op("vmatpush.acc.bf16.mxu1 acc0, m24", 32)
        op("vmatpush.acc.bf16.mxu1 acc1, m24", 32)
        emit("li x7, 0x30000", "li x24, 0x18000")
        key_start(f"pv_loop_{panel}")
        emit("lw x27, 4(x12)", "delay 4", f"bne x27, x0, pv_skip_{panel}", "nop")
        op("vload m22, 0(x24)", 34)
        for lane in range(2):
            dim = panel + lane
            op(f"vload m14, {dim * 8}(x7)", 34)
            op("vmatpush.weight.mxu1 w0, m14", 32)
            op(f"vmatmul.acc.mxu1 acc{lane}, m22, w0", 96)
        emit(f"pv_skip_{panel}:")
        key_end(f"pv_loop_{panel}", (f"li x28, {resident_words}", "add x7, x7, x28", "addi x24, x24, 256"))
        for lane in range(2):
            dim = panel + lane
            op(f"vmatpop.bf16.acc.mxu1 m12, acc{lane}", 34)
            op("vmul.bf16 m12, m12, m10")
            if c.output_fp8:
                op(f"vli.all m28, {_bf16(2.0 ** -c.o_exponent)}")
                op("vmul.bf16 m12, m12, m28")
                op(f"vli.all m28, {_bf16(448.0)}")
                op(f"vli.all m30, {_bf16(-448.0)}")
                op("vminimum.bf16 m12, m12, m28")
                op("vmaximum.bf16 m12, m12, m30")
                op("vmatpush.acc.bf16.mxu0 acc0, m12", 32)
                op("vmatpop.fp8.acc.mxu0 m14, acc0, e3", 34)
                op(f"vstore m14, {dim * 8}(x29)", 34)
            else:
                pair_store(12, "x29", dim * 16)
    emit(*head_dma("x29", "x8", "ch0", store=True, count=out_head), "dma.wait ch0")
    emit(f"li x28, {c.query_heads * head_bytes}", "add x4, x4, x28",
        f"li x28, {c.query_heads * out_head}", "add x8, x8, x28",
        "add x28, x2, x0", "add x2, x3, x0", "add x3, x28, x0",
        f"li x28, {c.key_tiles * 8}", "add x13, x13, x28", "addi x17, x17, 4",
        "addi x14, x14, -1", "beq x14, x0, query_done", "nop", "jal x0, query_loop", "nop", "query_done:",
        f"li x28, {head_bytes}", "add x19, x19, x28", f"li x28, {out_head * (group if c.decode_packed else 1)}",
        "add x20, x20, x28", "addi x16, x16, -1", "bne x16, x0, head_loop", "nop",
        f"li x28, {head_bytes}", "add x21, x21, x28", "add x22, x22, x28", "addi x15, x15, -1",
        "beq x15, x0, finished", "nop", "jal x0, kv_loop", "nop", "finished:",
        "dma.wait ch0", "dma.wait ch1", "delay 128", "nop", "ecall")
    return check_program_v2("\n".join(lines) + "\n")
