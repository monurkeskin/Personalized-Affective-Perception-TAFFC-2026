"""Adapters around the unchanged submission Solver; no alternative bidding policy."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading

from .contracts import AffectSnapshot, LegacyBid, PolicyClock, identifier

_SOURCE_LOCK = threading.RLock()


@contextmanager
def _source_context(root):
    if root is None:
        yield
        return
    # The original relative imports/files are supported in a dedicated worker.
    with _SOURCE_LOCK:
        previous_directory, previous_path = Path.cwd(), list(sys.path)
        try:
            sys.path[:0] = [str(root), str(root / "HATN"), str(root / "Agent/Solver_Agent")]
            os.chdir(root)
            yield
        finally:
            os.chdir(previous_directory)
            sys.path[:] = previous_path


class OriginalUtilityFacade:
    """Delegate original arithmetic/selection; add only the bid API its callers require."""
    def __init__(self, original, issue_order, domain_totals=None):
        self._original = original
        self.issue_order = tuple(issue_order)
        self.domain_totals = dict(domain_totals) if domain_totals is not None else None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_original"), name)

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self._original, memo), self.issue_order, self.domain_totals)

    def _wrap(self, values):
        return LegacyBid(values, self.issue_order, self.domain_totals, perspective="Agent")

    def get_offer_utility(self, offer):
        if hasattr(offer, "get_bid"):
            if isinstance(offer, LegacyBid):
                offer = offer.get_bid(perspective="Agent")
            else:
                offer = offer.get_bid()
        return self._original.get_offer_utility(offer)

    def get_offer_below_utility(self, target):
        return self._wrap(self._original.get_offer_below_utility(target))

    def get_offer_between_utility(self, low, high):
        return self._wrap(self._original.get_offer_between_utility(low, high))

    def get_offer_above_utility(self, low):
        return self._wrap(self._original.get_offer_above_utility(low))


class SolverAdapter:
    def __init__(self, solver, issue_order, domain_totals=None, *, clock=None,
                 provider=None, condition=None, source_root=None):
        self.solver = solver
        self.issue_order = tuple(issue_order)
        self.domain_totals = dict(domain_totals) if domain_totals is not None else None
        self.clock = clock or solver.time_controller
        if not isinstance(self.clock, PolicyClock):
            raise ValueError("SolverAdapter requires an explicitly mapped PolicyClock")
        self.provider, self.condition = provider, condition
        self.source_root = Path(source_root) if source_root is not None else None
        self._cache, self._offers = {}, {}
        self._lock = threading.RLock()
        self._session_id = None
        self._closed = False
        self._policy_started = self._poisoned = False

    def decide(self, session_id, request_id, payload):
        identifier(session_id, "session_id"); identifier(request_id, "request_id")
        if not isinstance(payload, dict):
            raise ValueError("Decision payload must be an object")
        serializable = dict(payload)
        if isinstance(serializable.get("affect"), AffectSnapshot):
            serializable["affect"] = serializable["affect"].to_dict()
        fingerprint = json.dumps(serializable, sort_keys=True, separators=(",", ":"), allow_nan=False)
        key = (session_id, request_id)
        with self._lock:
            if self._closed:
                raise RuntimeError("Policy adapter is closed")
            if self._session_id is not None and self._session_id != session_id:
                raise ValueError("A policy adapter belongs to one session only")
            if key in self._cache:
                old, result = self._cache[key]
                if old != fingerprint:
                    raise ValueError("Request identity conflicts with different content")
                if result is None:
                    raise RuntimeError("Previous policy attempt failed; its state is unknown and cannot be replayed")
                return copy.deepcopy(result)
            if self._poisoned:
                raise RuntimeError("Previous policy attempt failed after entering the core; close this session rather than continuing unknown state")
            self._cache[key] = (fingerprint, None)  # Reserve before provider or policy side effects.
            self._session_id = session_id
            try:
                return self._decide_reserved(key, payload, fingerprint)
            except Exception:
                if self._policy_started:
                    self._poisoned = True
                raise
            finally:
                self._policy_started = False

    def _decide_reserved(self, key, payload, fingerprint):
        session_id, request_id = key
        offer = payload.get("offer")
        if not isinstance(offer, dict):
            raise ValueError("Decision requires an offer object")
        offer_id = identifier(offer.get("offer_id"), "offer_id")
        offer_key = (session_id, offer_id)
        if offer_key in self._offers:
            old, result = self._offers[offer_key]
            if old != fingerprint:
                raise ValueError("Offer identity conflicts with different content")
            if result is None:
                raise RuntimeError("Previous offer attempt has unknown policy state")
            self._cache[key] = (fingerprint, result)
            return copy.deepcopy(result)
        self._offers[offer_key] = (fingerprint, None)
        self.clock.set_elapsed_fraction(payload.get("elapsed_fraction"))
        if self.clock.time_is_up():
            raise ValueError("Decision deadline reached; the caller must end the session")
        human_offer = LegacyBid(offer.get("bid", {}), self.issue_order, self.domain_totals, perspective="Human")
        data = payload.get("affect")
        if data is None and self.provider is not None:
            data = self.provider.snapshot(session_id=session_id, offer_id=offer_id)
        snapshot = data if isinstance(data, AffectSnapshot) else AffectSnapshot.from_dict(data)
        if snapshot.session_id != session_id or snapshot.offer_id != offer_id:
            raise ValueError("Affect snapshot session/offer identity does not match the decision")
        if self.condition is not None and snapshot.condition != self.condition:
            raise ValueError("Affect snapshot condition does not match the configured condition")
        with _source_context(self.source_root):
            self._policy_started = True
            returned = self.solver.receive_offer(human_offer, dict(snapshot.raw), dict(snapshot.normalized))
        if not isinstance(returned, (tuple, list)) or len(returned) not in (2, 3):
            raise ValueError("Submission Solver must return (action, mood) or (action, mood, text)")
        action, mood = returned[:2]
        text = returned[2] if len(returned) == 3 else payload.get("text")
        if mood is not None and not isinstance(mood, str):
            raise ValueError("Original mood must be text or null")
        if text is not None and not isinstance(text, str):
            raise ValueError("Original text must be text or null")
        if hasattr(action, "get_acceptor") or type(action).__name__ == "Accept":
            translated = {"kind": "accept", "actor": "agent", "offer_id": offer_id}
        else:
            if isinstance(action, LegacyBid):
                bid = action
            else:
                values = action.get_bid() if hasattr(action, "get_bid") else action
                bid = LegacyBid(values, self.issue_order, self.domain_totals, perspective="Agent")
            response_id = "solver-" + hashlib.sha256(f"{session_id}\0{request_id}".encode()).hexdigest()[:24]
            translated = {"kind": "offer", "actor": "agent", "offer_id": response_id, "bid": bid.to_named("Human")}
        result = {"action": translated, "mood": mood, "text": text, "affect": snapshot.to_dict()}
        json.dumps(result, allow_nan=False)
        self._cache[key] = (fingerprint, result)
        self._offers[offer_key] = (fingerprint, result)
        return copy.deepcopy(result)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for value in (self.provider, self.solver):
                close = getattr(value, "close", None)
                if callable(close):
                    close()


def _source_paths(source_root, profile_path):
    root = Path(source_root).expanduser().resolve()
    if (root / "src_code/solver_cl_agent").is_dir():
        root = root / "src_code/solver_cl_agent"
    profile = Path(profile_path).expanduser()
    if not profile.is_absolute():
        profile = root / profile
    profile = profile.resolve()
    if not (root / "Agent/Solver_Agent/Solver_Agent.py").is_file() or not profile.is_file():
        raise ValueError("Configured original Solver source or preference profile is missing")
    return root, profile


def validate_config(config):
    """Check local metadata only; no policy/ML/provider module is imported."""
    required = ("source_root", "profile_path", "issue_order", "domain_totals", "time_semantics")
    if not isinstance(config, dict) or any(name not in config for name in required):
        raise ValueError("Policy config requires source_root, profile_path, issue_order, domain_totals and time_semantics")
    if not isinstance(config["source_root"], str) or not isinstance(config["profile_path"], str):
        raise ValueError("Policy source_root and profile_path must be local path strings")
    order, totals = config["issue_order"], config["domain_totals"]
    if not isinstance(order, (list, tuple)) or not order or any(not isinstance(name, str) or not name for name in order) or len(set(order)) != len(order):
        raise ValueError("Policy issue_order must contain unique nonempty issue names")
    if not isinstance(totals, dict):
        raise ValueError("Policy domain_totals must explicitly map every issue to an integer count")
    LegacyBid(dict.fromkeys(order, 0), order, totals, perspective="Human")
    PolicyClock(config["time_semantics"], config.get("duration_seconds", 600))
    _source_paths(config["source_root"], config["profile_path"])
    return {name: config[name] for name in required}


def load_original_solver(source_root, profile_path, issue_order, domain_totals, *,
                         time_semantics, duration_seconds=600):
    """Load unchanged numerical files in an isolated policy process, without TF/robot imports."""
    root, profile = _source_paths(source_root, profile_path)
    with _source_context(root):
        for name in ("HATN", "Agent"):
            existing = sys.modules.get(name)
            location = getattr(existing, "__file__", None)
            if location is not None and root not in Path(location).resolve().parents:
                raise RuntimeError("Legacy packages already loaded from another source; use an isolated worker")
        try:
            actions = importlib.import_module("HATN.negoAction")
            sys.modules["negoAction"] = actions
            comparison = importlib.import_module("Agent.Solver_Agent.ComparisonObject")
            sys.modules["ComparisonObject"] = comparison
            utility_module = importlib.import_module("HATN.utilitySpace")
            module = importlib.import_module("Agent.Solver_Agent.Solver_Agent")
        except ImportError as exc:
            raise RuntimeError(f"Original policy runtime dependency/import is missing: {exc}. Prepare its isolated NumPy/scikit-learn environment; no fallback is used.") from exc
        original = utility_module.UtilitySpace(str(profile))
        if tuple(original.issue_names) != tuple(issue_order):
            raise ValueError("Configured issue_order differs from the original profile order")
        if domain_totals is not None and dict(zip(original.issue_names, original.issue_max_counts)) != dict(domain_totals):
            raise ValueError("Configured domain_totals differ from the original profile")
        utility = OriginalUtilityFacade(original, issue_order, domain_totals)
        clock = PolicyClock(time_semantics, duration_seconds)
        solver = module.SolverAgent(utility, clock)
    return SolverAdapter(solver, issue_order, domain_totals, clock=clock, source_root=root)


def adapter_from_config(config):
    validated = validate_config(config)
    adapter = load_original_solver(**validated, duration_seconds=config.get("duration_seconds", 600))
    adapter.condition = config.get("condition")
    if config.get("affect_provider"):
        from .perception import provider_from_config
        adapter.provider = provider_from_config(config["affect_provider"])
    return adapter
