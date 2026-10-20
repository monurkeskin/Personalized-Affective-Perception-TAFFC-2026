"""Original stdlib protocol-1 worker loop, also compatible with Python 2.7."""
from __future__ import print_function

import hashlib
import json
import sys

try:
    string_type = basestring
except NameError:
    string_type = str


def _reject_constant(value):
    raise ValueError("Nonfinite JSON number: " + value)


def _valid_id(value, limit=200):
    return (isinstance(value, string_type) and bool(value) and len(value) <= limit
            and not any(ord(char) < 32 for char in value))


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object field.")
        value[key] = item
    return value


def _write(transport, response, limit):
    text = json.dumps(response, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False) + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) > limit:
        raise ValueError("Worker response exceeds the message size bound.")
    try:
        transport.write(encoded)
    except TypeError:
        transport.write(text)
    transport.flush()


def serve(handler, max_message_bytes=65536):
    """Dispatch handle(op,payload,sid,rid), or a legacy call(op,payload).

    An identity is reserved before calling a handler. Repeating a failed attempt
    cannot replay its side effects; conflicting content is always rejected.
    """
    if not isinstance(max_message_bytes, int) or not 1024 <= max_message_bytes <= 1000000:
        raise ValueError("Invalid worker message size bound.")
    original_stdout = sys.stdout
    transport = getattr(original_stdout, "buffer", original_stdout)
    source = getattr(sys.stdin, "buffer", sys.stdin)
    sys.stdout = sys.stderr
    attempts = {}
    try:
        while True:
            line = source.readline(max_message_bytes + 1)
            if not line:
                break
            response = None
            try:
                if len(line) > max_message_bytes or not line.endswith(b"\n" if isinstance(line, bytes) else "\n"):
                    raise ValueError("Worker request exceeds the message size bound.")
                if not isinstance(line, string_type) or isinstance(line, bytes):
                    line = line.decode("utf-8")
                message = json.loads(line, parse_constant=_reject_constant,
                                     object_pairs_hook=_unique_object)
                if not isinstance(message, dict):
                    raise ValueError("Worker message must be an object.")
                session_id = message.get("session_id")
                request_id = message.get("request_id")
                if not _valid_id(session_id) or not _valid_id(request_id):
                    raise ValueError("Invalid session/request identity.")
                response = {"protocol": 1, "session_id": session_id, "request_id": request_id}
                if type(message.get("protocol")) is not int or message["protocol"] != 1:
                    raise ValueError("Unsupported worker protocol.")
                operation = message.get("operation")
                payload = message.get("payload")
                if not _valid_id(operation, 100) or not isinstance(payload, dict):
                    raise ValueError("Invalid operation or object payload.")
                if set(message) != {"protocol", "session_id", "request_id", "operation", "payload"}:
                    raise ValueError("Unexpected worker message fields.")
                fingerprint = hashlib.sha256(json.dumps(
                    message, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                    allow_nan=False).encode("utf-8")).hexdigest()
                key = (session_id, request_id)
                if key in attempts:
                    if attempts[key][0] != fingerprint:
                        raise ValueError("Request identity reused with different content.")
                    response = attempts[key][1]
                else:
                    response.update(ok=False, error="Previous delivery is unknown; no automatic retry.")
                    attempts[key] = (fingerprint, dict(response))
                    try:
                        if hasattr(handler, "handle"):
                            result = handler.handle(operation, payload, session_id, request_id)
                        else:
                            result = handler.call(operation, payload)
                        if not isinstance(result, dict):
                            raise ValueError("Worker handler must return an object.")
                        # Validate and bound the response before it becomes the cached acknowledgement.
                        candidate = dict(response)
                        candidate.pop("error", None)
                        candidate.update(ok=True, result=result)
                        serialized = json.dumps(candidate, sort_keys=True, separators=(",", ":"),
                                                ensure_ascii=True, allow_nan=False)
                        if len(serialized.encode("utf-8")) + 1 > max_message_bytes:
                            raise ValueError("Worker result exceeds the message size bound.")
                        response = json.loads(serialized)
                    except Exception as exc:
                        response = dict(response)
                        response.update(ok=False, error=str(exc)[:1000])
                    attempts[key] = (fingerprint, dict(response))
                _write(transport, response, max_message_bytes)
            except (ValueError, TypeError, UnicodeError) as exc:
                if response is None:
                    # An unidentifiable/truncated stream cannot be safely resynchronized.
                    break
                response.update(ok=False, error=str(exc)[:1000])
                _write(transport, response, max_message_bytes)
    finally:
        try:
            close = getattr(handler, "close", None)
            if close is not None:
                close()
        finally:
            sys.stdout = original_stdout
