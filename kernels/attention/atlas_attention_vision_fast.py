"""Opt-in schedule for unmasked, whole-tile BF16 vision attention.

The original generator remains the arithmetic and layout authority. This
schedule hoists two immutable constants, uses the inner-product MXU latency,
and overlaps the first QK/PV compute with the next independent weight load.
"""
import re

from atlas_attention_v2_program import assembly_v2 as _original, check_program_v2


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("original attention schedule differs from the qualified template")
    return source.replace(old, new, 1)


def assembly_v2(config, offsets, *, dram_base=0x90000000):
    """Generate the explicit vision-fast variant with the original arena ABI.

    Padding, masks, grouped heads, packed decode, and FP8 output require the
    original schedule. Reject them rather than silently selecting a fallback.
    """
    c = config
    if (c.mask != "none" or c.query_rows % 32 or c.keys % 32
            or c.query_heads != c.kv_heads or c.output_fp8 or c.decode_packed):
        raise ValueError("vision-fast requires unmasked whole tiles, equal heads, and BF16 output")
    source = _original(c, offsets, dram_base=dram_base)
    # Every key is visible in every whole query tile. Make this bound explicit
    # so the timing verifier can prove all resident VMEM accesses in the loops.
    source = _once(source, "lw x18, 0(x17)", f"li x18, {c.key_tiles}")
    # Whole unmasked tiles have neither skip flags nor mask pointers. Expose
    # these zeros so the verifier excludes the unreachable mask-load paths.
    for offset, count in ((4, 3), (0, 2)):
        instruction = f"lw x27, {offset}(x12)"
        if source.count(instruction) != count:
            raise ValueError("original mask descriptor sites differ from the qualified template")
        source = source.replace(instruction, "li x27, 0")
    source = _once(source, "kv_loop:\n", "vli.all m32, 49920\ndelay 66\n"
                   "vli.all m34, 17280\ndelay 66\nkv_loop:\n")
    source = _once(source, "vli.all m28, 49920\ndelay 66\nvmaximum.bf16 m20, m20, m28",
                   "vmaximum.bf16 m20, m20, m32")
    source = _once(source, "vli.all m28, 17280\ndelay 66\nvmul.bf16 m20, m20, m28",
                   "vmul.bf16 m20, m20, m34")
    # MXU1's inclusive compute latency is 35, compared with MXU0's 95.
    # A DELAY of 34 gives an issue gap of 36, leaving the compute drained.
    source, count = re.subn(r"(vmatmul\.acc\.mxu1[^\n]*\n)delay 96\n", r"\1delay 34\n", source)
    if count != 5:
        raise ValueError("expected exactly five inner-product compute sites")
    # The first dimension's compute reads Q/P and resident w0. Loading the
    # second dimension into m14 touches neither operand nor the accumulator.
    for instruction in ("vmatmul.acc.mxu1 acc0, m0, w0", "vmatmul.acc.mxu1 acc0, m22, w0"):
        source = _once(source, instruction + "\ndelay 34\nvload m14, 8(",
                       instruction + "\nvload m14, 8(")
    # ECALL bypasses DELAY in the pinned model and RTL. End with explicit DMA
    # waits; all vector and MXU commands have already drained at this point.
    source = _once(source, "dma.wait ch0\ndma.wait ch1\ndelay 128\necall\n",
                   "dma.wait ch0\ndma.wait ch1\necall\n")
    return check_program_v2(source)
