"""Request boundary for the legacy local API; no perception/learning changes."""
import hmac
import os
from urllib.parse import urlsplit

from flask import jsonify, request


def normalized_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Experiment IDs must be nonnegative integers.")
    if isinstance(value, str) and (not value or any(c not in "0123456789" for c in value)):
        raise ValueError("Experiment IDs must be nonnegative integers.")
    number = int(value)
    if not 0 <= number <= 2147483647:
        raise ValueError("Experiment ID is out of range.")
    return str(number)


def install_request_guard(app):
    token = os.environ.get("SOLVER_API_TOKEN", "")
    hosts = {"localhost", "127.0.0.1", "::1"}
    configured_hosts = {x.strip().lower() for x in os.environ.get("SOLVER_ALLOWED_HOSTS", "").split(",") if x.strip()}
    if configured_hosts - hosts and not token:
        raise ValueError("Additional API hosts require SOLVER_API_TOKEN.")
    hosts.update(configured_hosts)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

    @app.before_request
    def guard_local_experiment():
        # Host/Origin checks stop a browser from driving a local camera via another site.
        try:
            host = urlsplit("//" + request.host).hostname
        except ValueError:
            host = None
        if host not in hosts:
            return jsonify(status=False, error="Untrusted host"), 400
        if request.path == "/" or request.path.startswith("/static/"):
            return None
        origin = request.headers.get("Origin")
        if origin and origin != request.host_url.rstrip("/"):
            return jsonify(status=False, error="Cross-origin experiment request"), 403
        if request.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}:
            return jsonify(status=False, error="Cross-origin experiment request"), 403
        supplied = request.headers.get("X-Solver-Token", "")
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            supplied = authorization[7:]
        if token:
            if not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
                return jsonify(status=False, error="Unauthorized experiment request"), 401
        elif request.remote_addr not in {"127.0.0.1", "::1"}:
            return jsonify(status=False, error="Remote access requires an API token"), 403
        if request.view_args:
            try:
                for name in ("id", "camera_id"):
                    if name in request.view_args:
                        request.view_args[name] = normalized_id(request.view_args[name])
            except ValueError:
                return jsonify(status=False, error="Invalid experiment or camera ID"), 400
        return None
