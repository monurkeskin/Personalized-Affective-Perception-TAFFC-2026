"""Local, explicit launcher for isolated submission workers."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 65536


def _invalid_constant(value):
    raise ValueError("Non-finite JSON constant: " + value)


def _json(text):
    return json.loads(text, parse_constant=_invalid_constant)


def load_config(path):
    source = Path(path).resolve()
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("Runtime configuration exceeds 64 KiB.")
    value = _json(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Runtime configuration needs schema_version 1.")
    if not isinstance(value.get("session_id"), str) or not value["session_id"].strip():
        raise ValueError("An explicit session_id is required.")
    manifests = value.get("asset_manifests", [])
    if not isinstance(manifests, list) or any(not isinstance(p, str) or not p for p in manifests):
        raise ValueError("asset_manifests must be a list of explicit local paths.")
    setup_budget = value.get("policy_setup_timeout_seconds", 60)
    if (isinstance(setup_budget, bool) or not isinstance(setup_budget, (int, float))
            or not math.isfinite(setup_budget) or not 0 < setup_budget <= 60):
        raise ValueError("policy_setup_timeout_seconds must lie in (0, 60].")
    return value, source.parent


def _bridges(config):
    from .transport import BridgeConfig, ProcessBridge
    if not isinstance(config.get("policy"), dict):
        raise ValueError("Configure an isolated policy worker.")
    result = {"policy": ProcessBridge(BridgeConfig.from_dict("policy", config["policy"]))}
    if config.get("output") is not None:
        result["output"] = ProcessBridge(BridgeConfig.from_dict("output", config["output"]))
    return result


def doctor(path=None, connect=False):
    report = {"ok": True, "devices_connected": False, "models_loaded": False,
              "scope": "local software and explicit configuration", "configured_workers": []}
    if not (ROOT / "src_code/solver_cl_agent/Agent/Solver_Agent/Solver_Agent.py").is_file():
        raise ValueError("The submission Solver source is missing from this checkout.")
    if path:
        from .assets import verify_assets
        config, base = load_config(path)
        for manifest in config.get("asset_manifests", []):
            verify_assets(base / manifest)
        bridges = _bridges(config)
        try:
            for name, bridge in bridges.items():
                executable = bridge.config.command[0]
                if not Path(executable).is_file() and shutil.which(executable) is None:
                    raise ValueError("Missing worker interpreter: " + executable)
                item = {"name": name, "runtime": bridge.config.runtime,
                        "connected": False}
                if connect:
                    item["receipt"] = bridge.connect(config["session_id"])
                    item["connected"] = True
                report["configured_workers"].append(item)
            report["devices_connected"] = connect and "output" in bridges
            report["worker_code_loaded"] = connect
        finally:
            for bridge in bridges.values():
                bridge.close()
    return report


def smoke():
    """Exercise actual subprocess I/O with no device or policy substitute in run()."""
    from .transport import BridgeConfig, ProcessBridge
    from .journal import SidecarJournal
    with tempfile.TemporaryDirectory(prefix="solver-transport-") as temporary:
        bridge = ProcessBridge(BridgeConfig(
            name="synthetic", command=(sys.executable, "-m", "submission_runtime.smoke_worker"),
            runtime="synthetic-transport", capabilities=("present",), timeout_seconds=3,
        ))
        try:
            receipt = bridge.connect("SYNTHETIC-SMOKE")
            payload = {"text": "Synthetic delivery", "mood": "Neutral"}
            result = bridge.request("SYNTHETIC-SMOKE", "present-1", "present", payload)
            repeated = bridge.request("SYNTHETIC-SMOKE", "present-1", "present", payload)
            if result != repeated or result.get("delivered") is not True:
                raise ValueError("Synthetic delivery contract failed.")
            journal = SidecarJournal(Path(temporary) / "smoke.runtime.jsonl")
            journal.append("presentation.delivered", "SYNTHETIC-SMOKE", "present-1", result)
        finally:
            bridge.close()
    return {"ok": True, "scope": "synthetic transport only", "hardware_tested": False,
            "models_loaded": False, "runtime": receipt["runtime"], "duplicate_preserved": True}


@contextmanager
def _journal_owner(path):
    """Reserve the complete run, including device calls, before any worker starts."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    claim = target.with_suffix(target.suffix + ".run.lock")
    try:
        descriptor = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("Runtime journal is owned by another or interrupted launch; use a new session.") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        claim.unlink()


def run(path, journal_path):
    if journal_path:
        with _journal_owner(journal_path):
            _run_owned(path, journal_path)
    else:
        _run_owned(path, journal_path)


def _run_owned(path, journal_path):
    """Consume explicit decision packets; never invent perception or speech values."""
    from .assets import verify_assets
    from .journal import SidecarJournal
    config, base = load_config(path)
    for manifest in config.get("asset_manifests", []):
        verify_assets(base / manifest)
    session_id = config["session_id"]
    if journal_path and Path(journal_path).exists() and Path(journal_path).stat().st_size:
        raise ValueError("A previous runtime journal exists; inspect it and start a new session.")
    journal = SidecarJournal(Path(journal_path)) if journal_path else None
    bridges = _bridges(config)
    try:
        if "output" in bridges and journal is None:
            raise ValueError("Robot delivery requires --journal with a new private sidecar path.")
        for bridge in bridges.values():
            bridge.connect(session_id)
        if "prepare" in bridges["policy"].receipt.get("capabilities", []):
            prepared = bridges["policy"].request(
                session_id, "prepare-policy-" + uuid.uuid4().hex, "prepare", {},
                timeout_seconds=config.get("policy_setup_timeout_seconds", 60),
            )
            if prepared.get("prepared") is not True:
                raise ValueError("Policy preparation was not acknowledged.")
        while True:
            line = sys.stdin.buffer.readline(MAX_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_BYTES:
                raise ValueError("Decision packet exceeds 64 KiB.")
            packet = _json(line)
            if not isinstance(packet, dict):
                raise ValueError("A decision packet must be an object.")
            request_id = packet.get("request_id")
            if not isinstance(request_id, str) or not request_id.strip():
                raise ValueError("A decision packet needs request_id.")
            payload = {k: v for k, v in packet.items() if k != "request_id"}
            if "stiffness" in payload:
                raise ValueError("stiffness requires a separate explicit device operation, not presentation.")
            if "output" in bridges and not isinstance(payload.get("text"), str):
                raise ValueError("Robot delivery requires explicit original text in the packet.")
            result = bridges["policy"].request(session_id, request_id, "decide", payload)
            if journal:
                journal.append("decision.returned", session_id, request_id, result)
            if "output" in bridges:
                command = {"text": result.get("text", payload["text"]), "mood": result["mood"],
                           "event_id": request_id}
                # Gestures are supplied by the existing presentation layer, never selected here.
                for key in ("gesture", "face"):
                    if key in payload:
                        command[key] = payload[key]
                delivery_id = "present-" + request_id
                if journal:
                    journal.append("presentation.attempted", session_id, delivery_id, command)
                try:
                    delivered = bridges["output"].request(session_id, delivery_id, "present", command)
                    if delivered.get("delivered") is not True:
                        raise ValueError("Output did not acknowledge delivery.")
                except Exception:
                    if journal:
                        journal.append("presentation.failed", session_id, "failed-" + request_id,
                                       {"delivery_status": "unknown"})
                    raise
                if journal:
                    journal.append("presentation.delivered", session_id, "delivered-" + request_id,
                                   delivered)
            print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        for bridge in bridges.values():
            bridge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Validate local files/config; no devices by default.")
    check.add_argument("--config")
    check.add_argument("--connect", action="store_true", help="Explicitly load/connect configured workers.")
    commands.add_parser("smoke", help="Run a synthetic subprocess and journal check.")
    launch = commands.add_parser("run", help="Read decision packets from stdin using configured workers.")
    launch.add_argument("--config", required=True)
    launch.add_argument("--journal", help="Optional private .runtime.jsonl sidecar.")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor(args.config, args.connect)
        elif args.command == "smoke":
            result = smoke()
        else:
            run(args.config, args.journal)
            return 0
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
