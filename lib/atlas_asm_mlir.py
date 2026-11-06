"""Translate Atlas assembly text into Atlas machine-IR MLIR.

The input language is the one the kernel generators emit and npu_model's assembler
(``npu_model.util.converter``) accepts: one instruction per line, ``#`` comments,
whole-line ``label:`` definitions, the ``li`` and ``nop`` pseudo-instructions, and
branch or jump operands that are labels or signed instruction-word offsets. The
output uses the generic op syntax of the Atlas dialect (``atlas.start`` followed by
one machine op per encoded word), which ``atlas-opt`` verifies and ``atlas-emit``
encodes. This module does not import npu_model, so comparing its words with the
assembler's is an independent check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

__all__ = ['AsmError', 'Instruction', 'parse', 'translate', 'MNEMONICS']


class AsmError(ValueError):
    """The text is not accepted by the Atlas assembler grammar."""


@dataclass
class Instruction:
    op: str                      # Atlas dialect operation name, e.g. "atlas.alu_imm"
    fields: dict                 # attribute name -> int, str, or bool
    line: int                    # 1-based source line
    text: str                    # source text of the line (pseudo expansions share it)
    labels: list = field(default_factory=list)


_ABI = {'zero': 0, 'ra': 1, 'sp': 2, 'gp': 3, 'tp': 4, 't0': 5, 't1': 6, 't2': 7, 's0': 8, 'fp': 8, 's1': 9,
        **{f'a{i}': 10 + i for i in range(8)}, **{f's{i}': 16 + i for i in range(2, 12)},
        **{f't{i}': 25 + i for i in range(3, 7)}}
_LABEL = re.compile(r'^[A-Za-z_][A-Za-z0-9_.]*$')
_OFFSET = re.compile(r'^(-?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+))\(x(\d+)\)')


def _int(token):
    try:
        return int(token, 0)
    except ValueError:
        raise AsmError(f'expected an integer, got {token!r}') from None


def _bounded(value, low, high, what):
    if not low <= value < high:
        raise AsmError(f'{what} {value} is out of range [{low}, {high})')
    return value


def _reg(token, prefix, count, what):
    token = token.lower()
    if prefix == 'x' and token in _ABI:
        return _ABI[token]
    if not token.startswith(prefix):
        raise AsmError(f'expected {what} ({prefix}N), got {token!r}')
    try:
        number = int(token[len(prefix):])
    except ValueError:
        raise AsmError(f'malformed {what} {token!r}') from None
    return _bounded(number, 0, count, what)


def _x(token): return _reg(token, 'x', 32, 'scalar register')
def _m(token): return _reg(token, 'm', 64, 'matrix register')
def _e(token): return _reg(token, 'e', 32, 'exponent register')
def _w(token): return _reg(token, 'w', 2, 'weight slot')
def _acc(token): return _reg(token, 'acc', 2, 'accumulator slot')
def _ch(token): return _reg(token, 'ch', 8, 'DMA channel')


def _signed12(value):
    """npu_model's Imm12 accepts [-2048, 4096) and keeps the low 12 bits; the dialect stores them signed."""
    _bounded(value, -2048, 4096, '12-bit immediate')
    value &= 0xfff
    return value - 0x1000 if value & 0x800 else value


def _unsigned(value, bits, low):
    _bounded(value, low, 1 << bits, f'{bits}-bit immediate')
    return value & ((1 << bits) - 1)


def _shamt(token): return _bounded(_int(token), 0, 32, 'shift amount')


def _based(token):
    match = _OFFSET.match(token)
    if not match:
        raise AsmError(f'expected imm(xN), got {token!r}')
    return _signed12(int(match.group(1), 0)), _bounded(int(match.group(2)), 0, 32, 'scalar register')


_ALU_IMM = ('addi', 'slti', 'sltiu', 'xori', 'ori', 'andi')
_SHIFT_IMM = ('slli', 'srli', 'srai')
_ALU_REG = ('add', 'sub', 'sll', 'slt', 'sltu', 'xor', 'srl', 'sra', 'or', 'and')
_BRANCH = ('beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu')
_LOADS = ('lb', 'lh', 'lw', 'lbu', 'lhu')
_STORES = ('sb', 'sh', 'sw')
_BINARY = {'vadd.bf16': 'add', 'vsub.bf16': 'sub', 'vmul.bf16': 'mul', 'vminimum.bf16': 'min', 'vmaximum.bf16': 'max'}
_REDUCE = {'vredsum.bf16': 'col_sum', 'vredmin.bf16': 'col_min', 'vredmax.bf16': 'col_max',
           'vredsum.row.bf16': 'row_sum', 'vredmin.row.bf16': 'row_min', 'vredmax.row.bf16': 'row_max'}
_UNARY = {'vmov': 'mov', 'vrecip.bf16': 'recip', 'vexp.bf16': 'exp', 'vexp2.bf16': 'exp2', 'vrelu.bf16': 'relu',
          'vsin.bf16': 'sin', 'vcos.bf16': 'cos', 'vtanh.bf16': 'tanh', 'vlog2.bf16': 'log2', 'vsqrt.bf16': 'sqrt',
          'vsquare.bf16': 'square', 'vcube.bf16': 'cube'}
_CSR = {'csrrw': 'rrw', 'csrrs': 'rrs', 'csrrc': 'rrc', 'csrrwi': 'rrwi', 'csrrsi': 'rrsi', 'csrrci': 'rrci'}


def _strip(line):
    index = line.find('#')
    return line[:index].strip() if index >= 0 else line.strip()


def _tokens(line):
    return [t.rstrip(',') for t in re.split(r'[\s,]+', _strip(line)) if t]


def _li(rd, value):
    """The assembler's LI expansion: ADDI for 12-bit values, else LUI with an ADDI only for a nonzero low part."""
    value &= 0xffffffff
    if value >= 0x80000000:
        value -= 0x100000000
    if -2048 <= value <= 2047:
        return [('atlas.alu_imm', dict(kind='addi', dst=rd, src=0, immediate=value))]
    low = value & 0xfff
    if low & 0x800:
        low -= 0x1000
    upper = ('atlas.upper', dict(kind='lui', dst=rd, immediate=((value - low) >> 12) & 0xfffff))
    return [upper] if low == 0 else [upper, ('atlas.alu_imm', dict(kind='addi', dst=rd, src=rd, immediate=low))]


def _width(tokens):
    if tokens[0].lower() == 'li':
        if len(tokens) < 3:
            raise AsmError('malformed li')
        return len(_li(0, _int(tokens[2])))
    return 1


def _arity(tokens, count):
    if len(tokens) != count + 1:
        raise AsmError(f'{tokens[0]!r} expects {count} operands, got {len(tokens) - 1}')


def _control_offset(token, labels, pc, words_limit):
    """Branch and jump operands: a defined label or a signed word offset; returns the byte displacement."""
    if _LABEL.fullmatch(token):
        if token not in labels:
            raise AsmError(f'undefined label {token!r}')
        words = labels[token] - pc
    else:
        words = _int(token)
    _bounded(words, -words_limit, words_limit, 'instruction-word offset')
    return 2 * words


def _instruction(tokens, labels, pc):
    mnemonic = tokens[0].lower()
    a = tokens[1:]
    if mnemonic == 'nop' and not a:
        return [('atlas.alu_imm', dict(kind='addi', dst=0, src=0, immediate=0))]
    if mnemonic == 'li' and len(a) == 2:
        return _li(_x(a[0]), _int(a[1]))
    if mnemonic in _ALU_IMM:
        _arity(tokens, 3)
        return [('atlas.alu_imm', dict(kind=mnemonic, dst=_x(a[0]), src=_x(a[1]), immediate=_signed12(_int(a[2]))))]
    if mnemonic in _SHIFT_IMM:
        _arity(tokens, 3)
        return [('atlas.alu_imm', dict(kind=mnemonic, dst=_x(a[0]), src=_x(a[1]), immediate=_shamt(a[2])))]
    if mnemonic in _ALU_REG:
        _arity(tokens, 3)
        return [('atlas.alu_reg', dict(kind=mnemonic, dst=_x(a[0]), lhs=_x(a[1]), rhs=_x(a[2])))]
    if mnemonic in ('lui', 'auipc'):
        _arity(tokens, 2)
        return [('atlas.upper', dict(kind=mnemonic, dst=_x(a[0]), immediate=_unsigned(_int(a[1]), 20, -(1 << 19))))]
    if mnemonic in _LOADS:
        _arity(tokens, 2)
        offset, base = _based(a[1])
        return [('atlas.scalar_load', dict(kind=mnemonic, dst=_x(a[0]), base=base, offset=offset))]
    if mnemonic == 'seld':
        _arity(tokens, 2)
        offset, base = _based(a[1])
        return [('atlas.scalar_load', dict(kind='seld', dst=_e(a[0]), base=base, offset=offset))]
    if mnemonic == 'seli':
        _arity(tokens, 2)
        return [('atlas.scalar_load', dict(kind='seli', dst=_e(a[0]), base=0, offset=_signed12(_int(a[1]))))]
    if mnemonic in _STORES:
        _arity(tokens, 2)
        offset, base = _based(a[1])
        return [('atlas.scalar_store', dict(kind=mnemonic, src=_x(a[0]), base=base, offset=offset))]
    if mnemonic in ('vload', 'vstore'):
        _arity(tokens, 2)
        offset, base = _based(a[1])
        register = 'dst' if mnemonic == 'vload' else 'src'
        return [('atlas.' + mnemonic, {register: _m(a[0]), 'base': base, 'offset': offset, 'format': 'raw'})]
    if mnemonic in _BINARY:
        _arity(tokens, 3)
        return [('atlas.vpu_binary', dict(kind=_BINARY[mnemonic], dst=_m(a[0]), lhs=_m(a[1]), rhs=_m(a[2])))]
    if mnemonic in _REDUCE:
        _arity(tokens, 2)
        return [('atlas.vpu_reduce', dict(kind=_REDUCE[mnemonic], dst=_m(a[0]), src=_m(a[1])))]
    if mnemonic in _UNARY:
        _arity(tokens, 2)
        return [('atlas.vpu_unary', dict(kind=_UNARY[mnemonic], dst=_m(a[0]), src=_m(a[1])))]
    if mnemonic in ('vpack.bf16.fp8', 'vunpack.fp8.bf16'):
        _arity(tokens, 3)
        direction = 'bf16_to_fp8' if mnemonic.startswith('vpack') else 'fp8_to_bf16'
        return [('atlas.vpu_pack', dict(direction=direction, dst=_m(a[0]), src=_m(a[1]), scale_reg=_e(a[2])))]
    if mnemonic.startswith('vli.') and mnemonic[4:] in ('all', 'row', 'col', 'one'):
        _arity(tokens, 2)
        return [('atlas.vli', dict(mode=mnemonic[4:], dst=_m(a[0]), immediate=_unsigned(_int(a[1]), 16, -(1 << 15))))]
    if mnemonic == 'vtrpose.xlu':
        _arity(tokens, 2)
        return [('atlas.xlu_transpose', dict(dst=_m(a[0]), src=_m(a[1])))]
    if mnemonic in _BRANCH:
        _arity(tokens, 3)
        return [('atlas.branch', dict(kind=mnemonic, lhs=_x(a[0]), rhs=_x(a[1]),
                                      offset_bytes=_control_offset(a[2], labels, pc, 2048)))]
    if mnemonic == 'jal':
        _arity(tokens, 2)
        return [('atlas.jump', dict(kind='jal', dst=_x(a[0]), base=0,
                                    offset=_control_offset(a[1], labels, pc, 524288)))]
    if mnemonic == 'jalr':
        _arity(tokens, 3)
        return [('atlas.jump', dict(kind='jalr', dst=_x(a[0]), base=_x(a[1]), offset=_signed12(_int(a[2]))))]
    if mnemonic == 'delay':
        _arity(tokens, 1)
        return [('atlas.delay', dict(cycles=_unsigned(_int(a[0]), 12, -2048)))]
    if mnemonic in _CSR:
        _arity(tokens, 3)
        source = _shamt(a[2]) if mnemonic.endswith('i') else _x(a[2])
        return [('atlas.csr', dict(kind=_CSR[mnemonic], dst=_x(a[0]), source=source,
                                   address=_unsigned(_int(a[1]), 12, -2048)))]
    if mnemonic in ('ecall', 'ebreak'):
        _arity(tokens, 0)
        return [('atlas.trap', dict(kind=mnemonic))]
    if mnemonic == 'fence':
        _arity(tokens, 0)
        return [('atlas.fence', {})]
    match = re.fullmatch(r'vmatpush\.(weight|acc\.fp8|acc\.bf16)\.mxu([01])', mnemonic)
    if match:
        _arity(tokens, 2)
        kind = {'weight': 'weight_fp8', 'acc.fp8': 'acc_fp8', 'acc.bf16': 'acc_bf16'}[match.group(1)]
        slot = _w(a[0]) if kind == 'weight_fp8' else _acc(a[0])
        return [('atlas.mxu_push', dict(kind=kind, unit=int(match.group(2)), src=_m(a[1]), slot=slot))]
    match = re.fullmatch(r'vmatpop\.(fp8|bf16)\.acc\.mxu([01])', mnemonic)
    if match:
        fp8 = match.group(1) == 'fp8'
        _arity(tokens, 3 if fp8 else 2)
        return [('atlas.mxu_pop', dict(format=match.group(1), unit=int(match.group(2)), dst=_m(a[0]), slot=_acc(a[1]),
                                       scale_reg=_e(a[2]) if fp8 else 0))]
    match = re.fullmatch(r'vmatmul(\.acc)?\.mxu([01])', mnemonic)
    if match:
        _arity(tokens, 3)
        return [('atlas.mxu_matmul', dict(unit=int(match.group(2)), src=_m(a[1]), weight_slot=_w(a[2]),
                                          acc_slot=_acc(a[0]), accumulate=bool(match.group(1))))]
    if mnemonic in ('dma.load', 'dma.store'):
        _arity(tokens, 4)
        rd, rs1, rs2, channel = _x(a[0]), _x(a[1]), _x(a[2]), _ch(a[3])
        if mnemonic == 'dma.load':   # VMEM word x[rd] <- DRAM x[rs1]
            return [('atlas.dma', dict(direction='load', channel=channel, reg=rd, dram=rs1, size=rs2))]
        return [('atlas.dma', dict(direction='store', channel=channel, reg=rs1, dram=rd, size=rs2))]
    if mnemonic == 'dma.config':
        _arity(tokens, 2)
        return [('atlas.dma_config', dict(channel=_ch(a[1]), base_reg=_x(a[0])))]
    if mnemonic == 'dma.wait':
        _arity(tokens, 1)
        return [('atlas.dma_wait', dict(channel=_ch(a[0])))]
    raise AsmError(f'unknown mnemonic {tokens[0]!r}')


MNEMONICS = sorted({'nop', 'li', *_ALU_IMM, *_SHIFT_IMM, *_ALU_REG, 'lui', 'auipc', *_LOADS, 'seld', 'seli', *_STORES,
                    'vload', 'vstore', *_BINARY, *_REDUCE, *_UNARY, 'vpack.bf16.fp8', 'vunpack.fp8.bf16',
                    'vli.all', 'vli.row', 'vli.col', 'vli.one', 'vtrpose.xlu', *_BRANCH, 'jal', 'jalr', 'delay',
                    *_CSR, 'ecall', 'ebreak', 'fence', 'dma.load', 'dma.store', 'dma.config', 'dma.wait',
                    *(f'vmatpush.{k}.mxu{u}' for k in ('weight', 'acc.fp8', 'acc.bf16') for u in (0, 1)),
                    *(f'vmatpop.{k}.acc.mxu{u}' for k in ('fp8', 'bf16') for u in (0, 1)),
                    *(f'vmatmul{a}.mxu{u}' for a in ('', '.acc') for u in (0, 1))})


def parse(text):
    """Return the program's instructions, one per encoded word, in order."""
    lines = text.splitlines()
    labels, pc = {}, 0
    for raw in lines:                        # pass 1: label word addresses
        line = _strip(raw)
        if line.endswith(':'):
            labels[line[:-1].strip()] = pc
        elif (tokens := _tokens(line)):
            try:
                pc += _width(tokens)
            except AsmError as error:
                raise AsmError(f'line {lines.index(raw) + 1}: {error}') from None
    program, pending, pc = [], [], 0
    for number, raw in enumerate(lines, 1):  # pass 2: instructions
        line = _strip(raw)
        if line.endswith(':'):
            pending.append(line[:-1].strip())
            continue
        tokens = _tokens(line)
        if not tokens:
            continue
        try:
            expanded = _instruction(tokens, labels, pc)
        except AsmError as error:
            raise AsmError(f'line {number}: {error}: {raw.strip()}') from None
        for index, (op, fields) in enumerate(expanded):
            program.append(Instruction(op, fields, number, raw.strip(), pending if index == 0 else []))
            pending = []
        pc += len(expanded)
    return program


def _attribute(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return f'{value} : i32'
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def translate(text, *, name='program.S'):
    """Translate assembly text to an Atlas machine-IR module (generic syntax, one op per word)."""
    program = parse(text)
    source = name.replace('\\', '\\\\').replace('"', '\\"')
    out = ['module {', f'  %s0 = "atlas.start"() : () -> !atlas.state loc("{source}":0:0)']
    for index, insn in enumerate(program, 1):
        attributes = ', '.join(f'{key} = {_attribute(value)}' for key, value in insn.fields.items())
        attributes = ' {' + attributes + '}' if attributes else ''
        out.append(f'  %s{index} = "{insn.op}"(%s{index - 1}){attributes} : (!atlas.state) -> !atlas.state'
                   f' loc("{source}":{insn.line}:1)')
    out.append('}')
    return '\n'.join(out) + '\n'
