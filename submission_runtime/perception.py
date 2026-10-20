"""Trusted lab snapshot providers; never infer a missing normalization channel."""
from __future__ import annotations

import importlib
import json
import os
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .contracts import AffectSnapshot, finite


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, source, code, message, headers, new_url):
        return None


def snapshot_from_response(response, *, session_id, offer_id):
    if isinstance(response, AffectSnapshot):
        snapshot = response
    else:
        if not isinstance(response, dict):
            raise ValueError("Affect provider must return an explicit snapshot object")
        data = response.get("snapshot", response)
        if not isinstance(data, dict) or (data.get("normalized") is None and data.get("normalized_predictions") is None):
            raise ValueError("Legacy HTTP mean predictions do not provide the normalized channel; configure a trusted lab snapshot provider. Normalization is never inferred.")
        snapshot = AffectSnapshot.from_dict(data)
    if snapshot.session_id != session_id or snapshot.offer_id != offer_id:
        raise ValueError("Affect provider returned another session or offer snapshot")
    return snapshot


class FactorySnapshotProvider:
    def __init__(self, callback):
        self.callback = callback

    def snapshot(self, *, session_id, offer_id):
        function = getattr(self.callback, "snapshot", self.callback)
        response = function(session_id=session_id, offer_id=offer_id)
        return snapshot_from_response(response, session_id=session_id, offer_id=offer_id)

    def close(self):
        function = getattr(self.callback, "close", None)
        if callable(function):
            function()


class LegacyHTTPProvider:
    """Explicit legacy stop operation; ordinary mean-only responses fail closed."""
    def __init__(self, base_url, experiment_id, *, timeout=10, token=None, token_env=None, allow_remote=False):
        address = urlsplit(base_url)
        if address.scheme not in {"http", "https"} or not address.hostname or address.username or address.password:
            raise ValueError("Affect API requires an explicit HTTP(S) address without embedded credentials")
        if address.hostname not in {"localhost", "127.0.0.1", "::1"} and not allow_remote:
            raise ValueError("Remote affect API requires explicit allow_remote=True")
        if not 0 < finite(timeout, "Affect API timeout") <= 60:
            raise ValueError("Affect API timeout must lie in (0,60] seconds")
        if token_env is not None:
            if token is not None or not isinstance(token_env, str) or not token_env or not os.environ.get(token_env):
                raise ValueError("Configure one token source and ensure token_env names a populated environment variable")
            token = os.environ[token_env]
        if token is not None and (not isinstance(token, str) or not token or "\r" in token or "\n" in token):
            raise ValueError("Affect API token must be a nonempty single-line string")
        self.base_url, self.experiment_id = base_url.rstrip("/"), experiment_id
        self.timeout, self.token = timeout, token
        self.opener = build_opener(_NoRedirect())

    def snapshot(self, *, session_id, offer_id):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Solver-Token"] = self.token
        request = Request(self.base_url + "/stop", data=json.dumps({"id": self.experiment_id}, allow_nan=False).encode(), headers=headers, method="POST")
        with self.opener.open(request, timeout=self.timeout) as source:
            raw = source.read(65537)
        if len(raw) > 65536:
            raise ValueError("Affect API response exceeds the metadata limit")
        return snapshot_from_response(json.loads(raw), session_id=session_id, offer_id=offer_id)


def provider_from_config(config):
    if not isinstance(config, dict):
        raise ValueError("affect_provider must be a trusted local configuration object")
    if config.get("type") == "legacy_http":
        return LegacyHTTPProvider(**{key: value for key, value in config.items() if key != "type"})
    reference = config.get("factory")
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("Lab provider requires factory='module:callable' or type='legacy_http'")
    module, name = reference.split(":")
    factory = getattr(importlib.import_module(module), name)
    callback = factory(**config.get("options", {}))
    if not callable(callback) and not callable(getattr(callback, "snapshot", None)):
        raise ValueError("Lab provider factory must return a snapshot callable/provider")
    return FactorySnapshotProvider(callback)
