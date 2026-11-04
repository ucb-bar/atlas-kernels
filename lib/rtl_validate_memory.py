"""Replay DMA CSV requests and accepted responses through the pinned backend."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

from atlas_path_guard import checked
from rtl_memory_config import model_backend

FIELDS = ('event', 'absolute_cycle', 'source', 'address', 'opcode', 'due_cycle', 'accepted_cycle')


def validate(model_root, trace):
    """Validate every accepted A/D event; return counts and source provenance.

    Raise ValueError on malformed traces, incorrect due cycles, early or
    misidentified responses, capacity violations, or unreturned requests.
    """
    data = checked(trace).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8')))
    if reader.fieldnames != list(FIELDS):
        raise ValueError('DMA trace columns must be ' + ','.join(FIELDS))
    requests = responses = peak_pending = last_delivery = 0
    previous_cycle = -1
    with model_backend(model_root) as (module, backend, metadata):
        for number, row in enumerate(reader, 2):
            def fail(message):
                raise ValueError(f'DMA trace row {number}: {message}')

            if None in row or any(value is None for value in row.values()):
                fail('malformed CSV row')
            try:
                values = {key: int(row[key]) for key in FIELDS[1:]}
            except ValueError:
                fail('non-integer event field')
            cycle, source, address, opcode, due, accepted = (values[key] for key in FIELDS[1:])
            if min(cycle, source, address, due, accepted) < 0 or cycle < previous_cycle:
                fail('negative field or decreasing absolute cycle')
            if opcode not in (0, 4):
                fail('opcode must be PutFullData (0) or Get (4)')
            previous_cycle = cycle
            store = opcode != 4
            if row['event'] == 'request':
                if accepted != cycle:
                    fail('request accepted_cycle differs from absolute_cycle')
                if not backend.can_accept(cycle):
                    fail('request accepted when backend full')
                request = module.BeatRequest(source=source, address=address, is_store=store, cycle=cycle)
                backend.issue(request)
                predicted = next(ready for ready, _, pending in backend._pending if pending is request)
                if predicted != due:
                    fail(f'due_cycle is {due}, backend predicts {predicted}')
                requests += 1
            elif row['event'] == 'response':
                request = backend.response(cycle)
                if request is None:
                    fail('response accepted before backend presents it')
                if (request.source, request.address, request.is_store, request.cycle) != (source, address, store, accepted):
                    fail('response identity or accepted_cycle differs from backend')
                if backend._pending[0][0] != due:
                    fail('response due_cycle differs from backend')
                backend.pop_response(cycle)
                responses += 1
                last_delivery = cycle
            else:
                fail('event must be request or response')
            peak_pending = max(peak_pending, backend.outstanding)
        if requests != responses or backend.outstanding:
            raise ValueError(f'DMA trace is incomplete: {requests} requests, {responses} responses')
    return dict(schema='atlas-kernels.rtl-memory-validation.v1', status='PASS',
                requests=requests, responses=responses, peak_pending=peak_pending,
                last_delivery=last_delivery, trace_sha256=hashlib.sha256(data).hexdigest(),
                model_revision=metadata['model_revision'], source_sha256=metadata['source_sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    output = checked(args.out) if args.out else None
    result = validate(args.model_root, args.trace)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        checked(output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
