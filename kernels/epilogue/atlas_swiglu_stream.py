"""Opt-in contiguous finite-BF16 SiLU times up using the shared epilogue.

The RTL EXP/reciprocal tables and truncating denominator add are approximate
SiLU arithmetic, not the host FP32-expression SiLU. Both input buffers must
be finite BF16, with only the last 1024-element chunk zero padded. Gate uses
VMEM bytes 2048..4095; up and output share bytes 0..2047. No model placement
or default projection generator is changed by this standalone primitive.
"""
from atlas_epilogue import Epilogue, Step, TileInput
from atlas_rne_add_stream import flat_geometry

def layout(elements):
    rows,width=flat_geometry(elements)
    padded=rows*width
    size=2*padded
    return dict(elements=elements,padded_elements=padded,rows=rows,width=width,size=size,
                gate_offset=256,up_offset=size+512,output_offset=2*size+768,
                arena_bytes=3*size+1024,guard_bytes=256,vmem_bytes=4096)

def assembly(elements,*,dram_base=0x90000000):
    spec=layout(elements)
    if type(dram_base) is not int or dram_base<0 or dram_base%32 or dram_base+spec['arena_bytes']>=2**32:
        raise ValueError('guarded arena must fit the aligned low32 DMA window')
    gate=dram_base+spec['gate_offset'];up=dram_base+spec['up_offset'];output=dram_base+spec['output_offset']
    e=Epilogue(spec['rows'],spec['width'],output,
               (Step('silu'),Step('mul',TileInput(up))),scratch_registers=(10,11))
    # Gate occupies VMEM bytes 2048..4095. Up and output share bytes 0..2047.
    lines=['li x30, 0','dma.config x30, ch0','li x1, 512','li x2, 768',
           'li x3, 0','li x9, 256','li x8, 2048',f'li x14, {gate}',f'li x12, {output}',
           f"li x13, {spec['padded_elements']//1024}",'silu_flat_chunk:',
           'dma.load x1, x14, x8, ch0','dma.wait ch0',
           'vload m2, 0(x1)','delay 34','vload m3, 0(x2)','delay 34']
    lines+=e.apply(12)+e.store(12)+['add x14, x14, x8','add x12, x12, x8',
                                 'addi x13, x13, -1','bne x13, x0, silu_flat_chunk','nop','ecall']
    return '\n'.join(lines)+'\n'
