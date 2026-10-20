"""Headless policy worker; imports no hardware, ML model or device SDK on startup."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .contracts import identifier
from .solver import adapter_from_config, validate_config


class PolicyWorker:
    def __init__(self, config, *, adapter_factory=None):
        if not isinstance(config, dict):
            raise ValueError("Policy worker config must be an object")
        self.config = dict(config)
        self.adapter_factory = adapter_factory or adapter_from_config
        self._default_factory = adapter_factory is None
        self.adapters = {}
        self.closed_sessions = set()
        self.failed_preparations = set()
        self._closed = False

    def handle(self, operation, payload, session_id, request_id):
        identifier(session_id, "session_id"); identifier(request_id, "request_id")
        if self._closed:
            raise RuntimeError("Policy worker is closed")
        if operation == "hello":
            if self._default_factory:
                validate_config(self.config)
            return {"family": "solver-cl", "runtime": "solver-cl-policy", "capabilities": ["prepare", "decide"], "policy_loading": "explicit_prepare_or_first_decision", "original_policy_preserved": True, "normalization_inferred": False}
        if operation == "close":
            self.closed_sessions.add(session_id)
            adapter = self.adapters.pop(session_id, None)
            if adapter is not None:
                adapter.close()
            return {"closed": True}
        if operation not in {"prepare", "decide"}:
            raise ValueError("Policy worker supports hello, prepare, decide and close")
        if session_id in self.closed_sessions:
            raise ValueError("This policy session has been closed")
        if session_id in self.failed_preparations:
            raise RuntimeError("Policy preparation previously failed; close this session rather than retrying unknown initialization state")
        if session_id not in self.adapters:
            try:
                adapter = self.adapter_factory(self.config)
                if not callable(getattr(adapter, "decide", None)) or not callable(getattr(adapter, "close", None)):
                    raise ValueError("Policy factory must return an adapter with decide and close methods")
                self.adapters[session_id] = adapter
            except Exception:
                self.failed_preparations.add(session_id)
                raise
        if operation == "prepare":
            return {"prepared": True, "runtime": "solver-cl-policy"}
        return self.adapters[session_id].decide(session_id, request_id, payload)

    def close(self):
        if self._closed:
            return
        self._closed = True
        adapters, self.adapters = list(self.adapters.values()), {}
        errors = []
        for adapter in adapters:
            try:
                adapter.close()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Policy resource close failed: " + "; ".join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--factory", help="Trusted adapter factory module:callable; called with the config object")
    args = parser.parse_args(argv)
    if args.config is None and args.factory is None:
        parser.error("--config or --factory is required")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    factory = None
    if args.factory:
        if args.factory.count(":") != 1:
            parser.error("--factory must be module:callable")
        module, name = args.factory.split(":")
        factory = getattr(importlib.import_module(module), name)
    from .bridges.protocol import serve
    serve(PolicyWorker(config, adapter_factory=factory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
