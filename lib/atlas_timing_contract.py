"""Explicit immutable DRAM inputs for conditional Atlas timing verification.

``ATLAS_TIMING_DRAM_CONTRACTS`` names a JSON manifest with schema
``atlas.timing.dram-contracts.v1`` and ``contracts`` mapping assembly-text SHA-256
(lowercase hex) to ``{name, segments: [{address, bytes_hex}], image: {path,
address}}``. Addresses are unsigned 32-bit byte addresses. Segment starts and
lengths are aligned to 4 bytes, and their little-endian words become verifier
facts only after matching the supplied raw image. Relative image paths resolve
against the manifest directory; direct API contracts resolve against cwd.

The caller promises the exact declared DRAM bytes remain immutable throughout
one invocation, including against host or other external writes. Source hashes
select supplied assumptions; they never establish those assumptions themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from atlas_path_guard import checked, lexical_path

SCHEMA = 'atlas.timing.dram-contracts.v1'
ENVIRONMENT = 'ATLAS_TIMING_DRAM_CONTRACTS'
ASSUMPTIONS = (
    'Declared DRAM words equal the supplied image bytes at invocation entry.',
    'Declared DRAM words remain immutable throughout the invocation, including against host and external writes.',
)


class ContractError(ValueError):
    """An explicit DRAM contract is malformed or does not match its image."""


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ContractError(f'{label} requires exactly {", ".join(keys)}')


def _address(value, label):
    if type(value) is not int or not 0 <= value < 2**32 or value % 4:
        raise ContractError(f'{label} must be an aligned unsigned 32-bit byte address')
    return value


def _physical(value, base):
    if not isinstance(value, str) or not value:
        raise ContractError('image path must be a nonempty string')
    path = lexical_path(value)
    return checked(path if path.is_absolute() else base / path)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError('duplicate JSON key in DRAM contract manifest')
        result[key] = value
    return result


def _mlir_string(value):
    return '"' + ''.join(chr(b) if 32 <= b < 127 and b not in (34, 92) else f'\\{b:02X}'
                         for b in value.encode('utf-8')) + '"'


@dataclass(frozen=True)
class DramContract:
    name: str
    words: tuple
    sha256: str
    image_path: str
    image_address: int
    image_sha256: str

    def record(self):
        return dict(name=self.name, sha256=self.sha256, assumptions=list(ASSUMPTIONS), words=len(self.words),
                    image=dict(path=self.image_path, address=self.image_address, sha256=self.image_sha256))

    def annotate(self, mlir):
        if not mlir.startswith('module {\n'):
            raise ContractError('translator did not produce the expected module header')
        words = ', '.join(f'{{address = {address} : i64, value = {value} : i64}}'
                          for address, value in self.words)
        attribute = f'atlas.dram_contract = {{name = {_mlir_string(self.name)}, words = [{words}]}}'
        return 'module attributes {' + attribute + '} {\n' + mlir[len('module {\n'):]


def validate(contract, *, base=None):
    """Validate a direct contract against its raw image, returning immutable word facts."""
    try:
        _keys(contract, ('name', 'segments', 'image'), 'DRAM contract')
        name = contract['name']
        if not isinstance(name, str) or not name.strip() or any(ord(c) < 32 or ord(c) == 127 for c in name):
            raise ContractError('contract name must be a nonempty string without control characters')
        # Require valid UTF-8 before emitting an MLIR string.
        name.encode('utf-8')
        image = contract['image']
        _keys(image, ('path', 'address'), 'contract image')
        image_address = _address(image['address'], 'image address')
        image_path = _physical(image['path'], Path.cwd() if base is None else Path(base))
        segments = contract['segments']
        if not isinstance(segments, list) or not segments:
            raise ContractError('contract segments must be a nonempty list')
        ranges = []
        for segment in segments:
            _keys(segment, ('address', 'bytes_hex'), 'contract segment')
            address = _address(segment['address'], 'segment address')
            encoded = segment['bytes_hex']
            if not isinstance(encoded, str) or not re.fullmatch(r'(?:[0-9a-fA-F]{8})+', encoded):
                raise ContractError('segment bytes_hex must encode nonempty complete 4-byte words')
            contents = bytes.fromhex(encoded)
            if address + len(contents) > 2**32:
                raise ContractError('segment exceeds the unsigned 32-bit address space')
            ranges.append((address, contents))
        ranges.sort(key=lambda pair: pair[0])
        for (address, contents), (following, _) in zip(ranges, ranges[1:]):
            if address + len(contents) > following:
                raise ContractError('duplicate or overlapping contract segments')
        image_bytes = image_path.read_bytes()
        if not image_bytes or image_address + len(image_bytes) > 2**32:
            raise ContractError('image must be nonempty and fit the unsigned 32-bit address space')
        words = []
        for address, contents in ranges:
            offset = address - image_address
            if offset < 0 or offset + len(contents) > len(image_bytes):
                raise ContractError('contract segment is outside the supplied image')
            if image_bytes[offset:offset + len(contents)] != contents:
                raise ContractError('contract segment bytes differ from the supplied image')
            words.extend((address + offset, int.from_bytes(contents[offset:offset + 4], 'little'))
                         for offset in range(0, len(contents), 4))
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        identity = dict(name=name, words=words, image_address=image_address, image_sha256=image_sha256,
                        assumptions=ASSUMPTIONS)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return DramContract(name, tuple(words), digest, str(image_path), image_address, image_sha256)
    except ContractError:
        raise
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        raise ContractError(f'invalid DRAM contract: {error}') from error


def select(program_text, dram_contract=None):
    """Load before cache lookup so changed image/manifest bytes cannot reuse an old proof.

    An explicit ``dram_contract`` takes precedence over environment selection.
    No matching source key means ordinary verification with no contract.
    """
    if dram_contract is not None:
        return validate(dram_contract)
    manifest_path = os.environ.get(ENVIRONMENT)
    if manifest_path is None:
        return None
    try:
        path = checked(manifest_path)
        manifest = json.loads(path.read_text(), object_pairs_hook=_unique)
        _keys(manifest, ('schema', 'contracts'), 'DRAM contract manifest')
        if manifest['schema'] != SCHEMA or not isinstance(manifest['contracts'], dict):
            raise ContractError(f'DRAM contract manifest requires schema {SCHEMA} and a contracts map')
        for digest in manifest['contracts']:
            if not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ContractError('contract lookup keys must be lowercase source SHA-256 values')
        digest = hashlib.sha256(program_text.encode()).hexdigest()
        if digest not in manifest['contracts']:
            return None
        return validate(manifest['contracts'][digest], base=path.parent)
    except ContractError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise ContractError(f'invalid DRAM contract manifest: {error}') from error
