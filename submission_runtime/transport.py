"""Bounded, original stdlib transport for protocol-1 device workers.

This module owns only its explicitly launched child process. It never generates
an action, selects a fallback, evaluates a shell command, or retries delivery.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid


class BridgeError(RuntimeError):
    """A worker could not acknowledge the selected operation safely."""


class BridgeConflictError(BridgeError):
    """A request identity was reused for different content."""


def _label(value, name, limit=200):
    if (not isinstance(value, str) or not value or len(value) > limit
            or any(ord(char) < 32 for char in value)):
        raise ValueError("%s must be a nonempty bounded string without controls." % name)
    return value


def _encoded(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Worker values must be finite JSON objects.") from exc


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object field.")
        value[key] = item
    return value


@dataclass(frozen=True)
class BridgeConfig:
    name: str
    command: tuple
    runtime: str
    capabilities: tuple = ()
    timeout_seconds: float = 10.0
    max_message_bytes: int = 65536
    family: str | None = None

    def __post_init__(self):
        _label(self.name, "name", 100)
        _label(self.runtime, "runtime", 200)
        if (not isinstance(self.command, (list, tuple)) or not self.command
                or len(self.command) > 64):
            raise ValueError("command must be an argv list with 1-64 arguments.")
        for argument in self.command:
            if (not isinstance(argument, str) or not argument or "\0" in argument
                    or len(argument) > 131072):
                raise ValueError("command arguments must be nonempty strings without NUL.")
        if not isinstance(self.capabilities, (list, tuple)) or len(self.capabilities) > 32:
            raise ValueError("capabilities must be a bounded list.")
        for item in self.capabilities:
            _label(item, "capability", 100)
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities must not contain duplicates.")
        if (isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds)
                or not 0 < self.timeout_seconds <= 60):
            raise ValueError("timeout_seconds must lie in (0, 60].")
        if type(self.max_message_bytes) is not int or not 1024 <= self.max_message_bytes <= 1000000:
            raise ValueError("max_message_bytes must lie in [1024, 1000000].")
        if self.family is not None:
            _label(self.family, "family", 100)
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))

    @classmethod
    def from_dict(cls, name, values=None):
        if values is None:
            if not isinstance(name, dict):
                raise ValueError("Bridge configuration must be an object.")
            values = dict(name)
            name = values.pop("name", None)
        elif isinstance(values, dict):
            values = dict(values)
            if "name" in values and values.pop("name") != name:
                raise ValueError("Bridge names disagree.")
        else:
            raise ValueError("Bridge configuration must be an object.")
        allowed = {"command", "runtime", "capabilities", "timeout_seconds", "max_message_bytes", "family"}
        if set(values) - allowed:
            raise ValueError("Unknown bridge configuration fields: " + ", ".join(sorted(set(values) - allowed)))
        if "command" not in values or "runtime" not in values:
            raise ValueError("Bridge configuration requires command and runtime.")
        return cls(name=name, **values)


class ProcessBridge:
    def __init__(self, config):
        if not isinstance(config, BridgeConfig):
            raise ValueError("ProcessBridge needs a BridgeConfig.")
        self.config = config
        self.process = None
        self.receipt = None
        self._lock = threading.RLock()
        self._responses = queue.Queue(maxsize=4)
        self._attempts = {}
        self._reader_thread = None

    def _read_responses(self, process, responses):
        try:
            while True:
                line = process.stdout.readline(self.config.max_message_bytes + 1)
                if not line:
                    responses.put_nowait(BridgeError("Worker exited without an acknowledgement."))
                    return
                if len(line) > self.config.max_message_bytes or not line.endswith(b"\n"):
                    responses.put_nowait(BridgeError("Worker response exceeds the message size bound."))
                    return
                responses.put_nowait(line)
        except (OSError, ValueError, queue.Full):
            try:
                responses.put_nowait(BridgeError("Worker response stream failed."))
            except queue.Full:
                pass

    def connect(self, session_id):
        _label(session_id, "session_id")
        with self._lock:
            if self.process is not None:
                if self.process.poll() is None and self.receipt is not None:
                    return json.loads(_encoded(self.receipt))
                self.close()
            self._responses = queue.Queue(maxsize=4)
            try:
                self.process = subprocess.Popen(
                    list(self.config.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, shell=False, bufsize=0,
                )
                self._reader_thread = threading.Thread(
                    target=self._read_responses, args=(self.process, self._responses), daemon=True)
                self._reader_thread.start()
                receipt = self.request(session_id, "hello-" + uuid.uuid4().hex, "hello", {})
                if receipt.get("runtime") != self.config.runtime:
                    raise BridgeError("Worker runtime does not match configuration.")
                offered = receipt.get("capabilities")
                if (not isinstance(offered, list) or any(not isinstance(x, str) for x in offered)
                        or not set(self.config.capabilities).issubset(offered)):
                    raise BridgeError("Worker lacks a configured capability.")
                if self.config.family is not None and receipt.get("family") != self.config.family:
                    raise BridgeError("Worker family does not match configuration.")
                self.receipt = receipt
                return json.loads(_encoded(receipt))
            except (OSError, ValueError, BridgeError) as exc:
                self.close()
                if isinstance(exc, BridgeError):
                    raise
                raise BridgeError("Could not prepare the selected worker: %s" % exc) from exc

    @staticmethod
    def _write(process, wire, completed):
        try:
            position = 0
            while position < len(wire):
                written = os.write(process.stdin.fileno(), wire[position:position + 4096])
                if written <= 0:
                    raise OSError("Worker input closed.")
                position += written
            completed.put(None)
        except (OSError, ValueError) as exc:
            completed.put(exc)

    def request(self, session_id, request_id, operation, payload, *, timeout_seconds=None):
        budget = self.config.timeout_seconds if timeout_seconds is None else timeout_seconds
        if (isinstance(budget, bool) or not isinstance(budget, (int, float))
                or not math.isfinite(budget) or not 0 < budget <= 60):
            raise ValueError("Request timeout must lie in (0, 60].")
        _label(session_id, "session_id")
        _label(request_id, "request_id")
        _label(operation, "operation", 100)
        if not isinstance(payload, dict):
            raise ValueError("Worker payload must be an object.")
        message = {"protocol": 1, "session_id": session_id, "request_id": request_id,
                   "operation": operation, "payload": payload}
        content = _encoded(message)
        wire = content + b"\n"
        if len(wire) > self.config.max_message_bytes:
            raise ValueError("Worker request exceeds the message size bound.")
        fingerprint = hashlib.sha256(content).hexdigest()
        key = (session_id, request_id)
        with self._lock:
            previous = self._attempts.get(key)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise BridgeConflictError("Request identity reused with different content.")
                if previous[1] == "acknowledged":
                    return json.loads(previous[2])
                raise BridgeError("Previous delivery failed or is unknown; it will not be retried.")
            process = self.process
            if process is None or process.poll() is not None:
                raise BridgeError("Worker is disconnected; prepare it explicitly before a new request.")
            self._attempts[key] = (fingerprint, "attempted", None)
            deadline = time.monotonic() + budget
            completed = queue.Queue(maxsize=1)
            writer = threading.Thread(target=self._write, args=(process, wire, completed), daemon=True)
            writer.start()
            try:
                write_error = completed.get(timeout=max(0.001, deadline - time.monotonic()))
                if write_error is not None:
                    raise BridgeError("Worker input failed: %s" % write_error)
                response = self._responses.get(timeout=max(0.001, deadline - time.monotonic()))
                if isinstance(response, BridgeError):
                    raise response
                data = json.loads(response.decode("utf-8"), parse_constant=self._invalid_constant,
                                  object_pairs_hook=_unique_object)
                if (not isinstance(data, dict) or type(data.get("protocol")) is not int
                        or data.get("protocol") != 1 or data.get("session_id") != session_id
                        or data.get("request_id") != request_id):
                    raise BridgeError("Worker response identity/protocol mismatch.")
                if data.get("ok") is not True:
                    raise BridgeError("Worker rejected command: %s" % str(data.get("error", "unknown error"))[:1000])
                result = data.get("result")
                if not isinstance(result, dict):
                    raise BridgeError("Worker result must be an object.")
                saved = _encoded(result)
                self._attempts[key] = (fingerprint, "acknowledged", saved)
                return json.loads(saved)
            except queue.Empty as exc:
                self._attempts[key] = (fingerprint, "failed", None)
                self.close()
                raise BridgeError("Worker timed out; delivery is unknown and will not be retried.") from exc
            except (OSError, UnicodeError, ValueError, BridgeError) as exc:
                self._attempts[key] = (fingerprint, "failed", None)
                self.close()
                if isinstance(exc, BridgeError):
                    raise
                raise BridgeError("Invalid worker response: %s" % exc) from exc

    @staticmethod
    def _invalid_constant(value):
        raise ValueError("Nonfinite JSON number: " + value)

    def close(self):
        with self._lock:
            process, self.process = self.process, None
            self.receipt = None
            if process is None:
                return
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
            if process.stdout is not None:
                process.stdout.close()
