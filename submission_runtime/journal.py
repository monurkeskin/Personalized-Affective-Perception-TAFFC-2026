"""A durable sidecar of runtime receipts; this journal never creates offers."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading


class JournalError(RuntimeError):
    """A sidecar event cannot be acknowledged safely."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _text(value, name):
    if (not isinstance(value, str) or not value or len(value) > 200
            or any(ord(c) < 32 for c in value)):
        raise JournalError("Invalid " + name + ".")
    return value


@contextmanager
def _exclusive(path):
    descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+b", buffering=0) as lock:
        if os.name == "nt":
            import msvcrt
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b"0")
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class SidecarJournal:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._failed = False

    def _read(self):
        if self.path.is_symlink():
            raise JournalError("Journal must not be a symbolic link.")
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise JournalError("Incomplete sidecar record; preserve the file for review.")
        rows = []
        previous = "0" * 64
        identities = set()
        try:
            for line in raw.splitlines():
                record = json.loads(line)
                checksum = record.pop("sha256")
                if hashlib.sha256(_json(record)).hexdigest() != checksum:
                    raise JournalError("Sidecar hash mismatch.")
                key = (record["session_id"], record["request_id"], record["kind"])
                if (record["sequence"] != len(rows) + 1 or record["previous_sha256"] != previous
                        or key in identities):
                    raise JournalError("Sidecar sequence or request identity mismatch.")
                identities.add(key)
                record["sha256"] = checksum
                rows.append(record)
                previous = checksum
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise JournalError("Invalid sidecar record: %s" % exc) from exc
        return rows

    def read(self):
        with self._lock:
            if not self.path.parent.exists():
                return []
            with _exclusive(self.path.with_suffix(self.path.suffix + ".lock")):
                return self._read()

    def append(self, kind, session_id, request_id, payload):
        _text(kind, "kind")
        _text(session_id, "session_id")
        _text(request_id, "request_id")
        if not isinstance(payload, dict):
            raise JournalError("Sidecar payload must be an object.")
        try:
            canonical = _json(payload)
        except (ValueError, TypeError) as exc:
            raise JournalError("Sidecar payload is not finite JSON.") from exc
        if len(canonical) > 1000000:
            raise JournalError("Sidecar payload exceeds its size bound.")
        payload = json.loads(canonical)
        with self._lock:
            if self._failed:
                raise JournalError("Sidecar write failed; review it before continuing.")
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with _exclusive(self.path.with_suffix(self.path.suffix + ".lock")):
                rows = self._read()
                for row in rows:
                    if (row["session_id"], row["request_id"], row["kind"]) == (session_id, request_id, kind):
                        if _json(row["payload"]) != canonical:
                            raise JournalError("Sidecar request identity reused with different content.")
                        return row
                record = {"schema_version": 1, "sequence": len(rows) + 1, "kind": kind,
                          "session_id": session_id, "request_id": request_id, "payload": payload,
                          "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                          "previous_sha256": rows[-1]["sha256"] if rows else "0" * 64}
                record["sha256"] = hashlib.sha256(_json(record)).hexdigest()
                wire = _json(record) + b"\n"
                descriptor = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    offset = os.lseek(descriptor, 0, os.SEEK_END)
                    try:
                        position = 0
                        while position < len(wire):
                            count = os.write(descriptor, wire[position:])
                            if count <= 0:
                                raise OSError("Incomplete sidecar write.")
                            position += count
                        os.fsync(descriptor)
                    except OSError as exc:
                        self._failed = True
                        try:
                            os.ftruncate(descriptor, offset)
                            os.fsync(descriptor)
                        except OSError:
                            pass
                        raise JournalError("Sidecar event was not acknowledged: %s" % exc) from exc
                finally:
                    os.close(descriptor)
                return record

    def presentation_attempted(self, session_id, request_id, payload):
        return self.append("presentation.attempted", session_id, request_id, payload)

    def presentation_delivered(self, session_id, request_id, receipt):
        return self.append("presentation.delivered", session_id, request_id, {"receipt": receipt})

    def presentation_failed(self, session_id, request_id, error):
        return self.append("presentation.failed", session_id, request_id,
                           {"error": str(error)[:1000], "delivery_status": "unknown"})

