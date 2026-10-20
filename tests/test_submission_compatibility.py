"""Submission policy boundaries; all observations and profiles are synthetic fixtures."""
import importlib.util
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import BaseHandler
from urllib.response import addinfourl

from submission_runtime.contracts import AffectSnapshot, LegacyBid, PolicyClock
from submission_runtime.perception import LegacyHTTPProvider, provider_from_config, snapshot_from_response
from submission_runtime.solver import OriginalUtilityFacade, SolverAdapter, adapter_from_config
from submission_runtime.worker import PolicyWorker

ROOT = Path(__file__).resolve().parents[1]


def affect(session="S1", offer="h1"):
    return {
        "session_id": session, "offer_id": offer, "window_id": "capture-1",
        "model_id": "frozen-plus-session-revision-1", "normalization_id": "original-state-1",
        "observed_at": "2026-10-20T12:00:00Z",
        "raw": {"Arousal": .1, "Valance": -.2},
        "normalized": {"Arousal": .3, "Valance": -.4},
    }


def payload(session="S1", offer="h1"):
    return {
        "offer": {"offer_id": offer, "bid": {"Zeta": 1, "Alpha": 2}},
        "elapsed_fraction": .25, "affect": affect(session, offer),
    }


class FakeSolver:
    def __init__(self, clock, *, failure=False):
        self.time_controller = clock
        self.calls = []
        self.failure = failure

    def receive_offer(self, offer, raw, normalized):
        self.calls.append((offer, raw, normalized, self.time_controller.get_remaining_time()))
        if self.failure:
            raise ValueError("synthetic failure after policy state advanced")
        return LegacyBid([2, 1], ("Zeta", "Alpha"), {"Zeta": 2, "Alpha": 5}, perspective="Agent"), "Original Mood", "Exact text <as supplied>"


def adapter(*, failure=False):
    clock = PolicyClock("elapsed_fraction")
    solver = FakeSolver(clock, failure=failure)
    return SolverAdapter(solver, ("Zeta", "Alpha"), {"Zeta": 2, "Alpha": 5}, clock=clock), solver


class CompatibilityTests(unittest.TestCase):
    def test_snapshot_is_deeply_immutable_and_does_not_normalize(self):
        data = affect()
        snapshot = AffectSnapshot.from_dict(data)
        data["raw"]["Arousal"] = 100
        self.assertEqual(snapshot.raw["Arousal"], .1)
        self.assertEqual(snapshot.normalized["Arousal"], .3)
        with self.assertRaises(TypeError):
            snapshot.normalized["Arousal"] = 0

    def test_missing_normalized_channel_is_an_explicit_error(self):
        raw_only = {"status": True, "predictions": {"Arousal": .1, "Valance": -.2}}
        with self.assertRaisesRegex(ValueError, "normalized"):
            snapshot_from_response(raw_only, session_id="S1", offer_id="h1")
        missing = affect(); missing.pop("normalized")
        with self.assertRaisesRegex(ValueError, "normalized"):
            AffectSnapshot.from_dict(missing)

    def test_nonfinite_and_wrong_head_names_are_rejected(self):
        for value in [float("nan"), float("inf"), True]:
            data = affect(); data["normalized"]["Arousal"] = value
            with self.assertRaises(ValueError):
                AffectSnapshot.from_dict(data)
        data = affect(); data["raw"] = {"valence": .1, "arousal": .2}
        with self.assertRaises(ValueError):
            AffectSnapshot.from_dict(data)

    def test_snapshot_offer_session_window_time_and_model_identity_required(self):
        for field in ["session_id", "offer_id", "window_id", "model_id", "normalization_id", "observed_at"]:
            data = affect(); data.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                AffectSnapshot.from_dict(data)
        data = affect(); data["observed_at"] = "2026-10-20T12:00:00"
        with self.assertRaises(ValueError):
            AffectSnapshot.from_dict(data)

    def test_named_order_perspective_and_clock_preserve_inputs(self):
        wrapper, core = adapter()
        result = wrapper.decide("S1", "r1", payload())
        offer, raw, normalized, t = core.calls[0]
        self.assertEqual(offer.get_bid(perspective="Human"), [1, 2])
        self.assertEqual(offer.get_bid(perspective="Agent"), [1, 3])
        self.assertEqual(raw, {"Arousal": .1, "Valance": -.2})
        self.assertEqual(normalized, {"Arousal": .3, "Valance": -.4})
        self.assertEqual(t, .25)
        self.assertEqual(result["action"]["bid"], {"Zeta": 0, "Alpha": 4})
        self.assertEqual(result["mood"], "Original Mood")
        self.assertEqual(result["text"], "Exact text <as supplied>")

    def test_request_retry_is_cached_and_conflicts_do_not_advance_core(self):
        wrapper, core = adapter()
        first = wrapper.decide("S1", "r1", payload())
        first["action"]["bid"]["Zeta"] = 99
        self.assertEqual(wrapper.decide("S1", "r1", payload())["action"]["bid"]["Zeta"], 0)
        changed = payload(); changed["elapsed_fraction"] = .4
        with self.assertRaisesRegex(ValueError, "identity|conflict|different"):
            wrapper.decide("S1", "r1", changed)
        self.assertEqual(len(core.calls), 1)

    def test_failed_mutating_call_is_never_automatically_replayed(self):
        wrapper, core = adapter(failure=True)
        with self.assertRaisesRegex(ValueError, "synthetic failure"):
            wrapper.decide("S1", "r1", payload())
        with self.assertRaises(RuntimeError):
            wrapper.decide("S1", "r1", payload())
        with self.assertRaises(RuntimeError):
            wrapper.decide("S1", "r2", payload(offer="h2"))
        self.assertEqual(len(core.calls), 1)

    def test_cross_session_or_offer_snapshot_is_rejected_before_policy(self):
        for field, value in [("session_id", "S2"), ("offer_id", "other")]:
            wrapper, core = adapter(); data = payload(); data["affect"][field] = value
            with self.assertRaises(ValueError):
                wrapper.decide("S1", "r1", data)
            self.assertEqual(core.calls, [])

    def test_clock_mapping_is_explicit_not_remaining_seconds(self):
        clock = PolicyClock("elapsed_fraction", duration_seconds=900)
        for t in [0., .5, 1.]:
            clock.set_elapsed_fraction(t)
            self.assertEqual(clock.get_current_time(), t)
            self.assertEqual(clock.get_remaining_time(), t)
        with self.assertRaises(ValueError):
            PolicyClock("remaining_seconds")

    def test_deadline_rejected_before_core_or_provider_effects(self):
        for value in [1., 1.1]:
            wrapper, core = adapter(); data = payload(); data["elapsed_fraction"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "deadline|fraction"):
                wrapper.decide("S1", "r1", data)
            self.assertEqual(core.calls, [])

    def test_http_redirect_cannot_forward_snapshot_token(self):
        visited = []
        class RedirectResponse(BaseHandler):
            handler_order = 100
            def http_open(self, request):
                visited.append(request.full_url)
                response = addinfourl(io.BytesIO(b""), {"Location": "http://outside.invalid/stop"}, request.full_url, 302)
                response.msg = "Found"
                return response
        provider = LegacyHTTPProvider("http://127.0.0.1:5000", "exp", token="synthetic-token")
        provider.opener.add_handler(RedirectResponse())
        with self.assertRaises(HTTPError):
            provider.snapshot(session_id="S1", offer_id="h1")
        self.assertEqual(visited, ["http://127.0.0.1:5000/stop"])
        for value in [float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                LegacyHTTPProvider("http://localhost", "exp", timeout=value)
        with patch.dict(os.environ, {"SYNTHETIC_AFFECT_TOKEN": "synthetic-secret"}):
            configured = provider_from_config({"type": "legacy_http", "base_url": "http://localhost", "experiment_id": "exp", "token_env": "SYNTHETIC_AFFECT_TOKEN"})
        self.assertEqual(configured.token, "synthetic-secret")

    def test_original_utility_selector_is_delegated_without_reordering(self):
        class Original:
            def get_offer_below_utility(self, target):
                self.target = target
                return [2, 1]
            def get_offer_utility(self, values):
                return round(values[0] * .1234567 + values[1] * .7654321, 6)
        original = Original()
        facade = OriginalUtilityFacade(original, ("Zeta", "Alpha"), {"Zeta": 2, "Alpha": 5})
        chosen = facade.get_offer_below_utility(.678901234)
        self.assertEqual(original.target, .678901234)
        self.assertEqual(chosen, [2, 1])
        self.assertEqual(chosen.get_bid(perspective="Human"), [0, 4])
        self.assertEqual(facade.get_offer_utility(chosen), original.get_offer_utility([2, 1]))

    def test_worker_hello_is_lazy_and_decide_keeps_mood_text(self):
        built = []
        def factory(config):
            wrapper, _ = adapter(); built.append(wrapper); return wrapper
        worker = PolicyWorker({}, adapter_factory=factory)
        self.assertEqual(worker.handle("hello", {}, "S1", "hello")["runtime"], "solver-cl-policy")
        self.assertEqual(built, [])
        result = worker.handle("decide", payload(), "S1", "r1")
        self.assertEqual(result["mood"], "Original Mood")
        self.assertEqual(len(built), 1)
        worker.close(); worker.close()

    def test_default_worker_hello_preflights_metadata_without_loading_core(self):
        with self.assertRaisesRegex(ValueError, "config requires"):
            PolicyWorker({}).handle("hello", {}, "S1", "hello")
        config = self.original_config()
        worker = PolicyWorker(config)
        with patch("submission_runtime.solver.importlib.import_module", side_effect=AssertionError("hello must not import policy")):
            self.assertEqual(worker.handle("hello", {}, "S1", "hello")["runtime"], "solver-cl-policy")
        self.assertEqual(worker.adapters, {})
        for field, value in [("profile_path", "missing.xml"), ("time_semantics", "remaining_seconds"), ("issue_order", ["apple", "apple"])]:
            bad = dict(config); bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                PolicyWorker(bad).handle("hello", {}, "S1", "hello")

    def test_prepare_loads_one_factory_before_any_decision(self):
        built = []
        def factory(config):
            wrapper, core = adapter(); built.append(core); return wrapper
        worker = PolicyWorker({}, adapter_factory=factory)
        self.assertIn("prepare", worker.handle("hello", {}, "S1", "hello")["capabilities"])
        self.assertEqual(built, [])
        self.assertTrue(worker.handle("prepare", {}, "S1", "prepare-1")["prepared"])
        self.assertTrue(worker.handle("prepare", {}, "S1", "prepare-2")["prepared"])
        self.assertEqual(len(built), 1)
        self.assertEqual(built[0].calls, [])
        worker.handle("decide", payload(), "S1", "r1")
        self.assertEqual(len(built), 1)
        self.assertEqual(len(built[0].calls), 1)
        worker.close()

    def test_failed_preparation_propagates_and_is_not_automatically_retried(self):
        calls = []
        def failure(config):
            calls.append(config)
            raise ValueError("synthetic missing runtime dependency")
        worker = PolicyWorker({}, adapter_factory=failure)
        with self.assertRaisesRegex(ValueError, "synthetic missing"):
            worker.handle("prepare", {}, "S1", "prepare-1")
        with self.assertRaisesRegex(RuntimeError, "preparation.*failed"):
            worker.handle("prepare", {}, "S1", "prepare-2")
        with self.assertRaises(RuntimeError):
            worker.handle("decide", payload(), "S1", "r1")
        self.assertEqual(len(calls), 1)
        worker.close()

    def test_closed_session_cannot_be_prepared_again(self):
        built = []
        def factory(config):
            wrapper, _ = adapter(); built.append(wrapper); return wrapper
        worker = PolicyWorker({}, adapter_factory=factory)
        worker.handle("prepare", {}, "S1", "prepare-1")
        worker.handle("close", {}, "S1", "close-1")
        with self.assertRaisesRegex(ValueError, "closed"):
            worker.handle("prepare", {}, "S1", "prepare-2")
        self.assertEqual(len(built), 1)
        worker.close()

    def original_config(self):
        order = ("watermelon", "banana", "orange", "apple")
        return {"source_root": str(ROOT / "src_code/solver_cl_agent"), "profile_path": str(ROOT / "src_code/solver_cl_agent/HATN/Domain/Fruit/Fruits_A_Agent.xml"), "issue_order": list(order), "domain_totals": dict.fromkeys(order, 4), "time_semantics": "elapsed_fraction"}

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Actual frozen Solver requires the isolated scikit-learn worker environment")
    def test_actual_frozen_solver_receives_nine_synthetic_offers_without_core_edits(self):
        order = ("watermelon", "banana", "orange", "apple")
        wrapper = adapter_from_config(self.original_config())
        for index in range(1, 10):
            data = {"offer": {"offer_id": f"h{index}", "bid": dict.fromkeys(order, 4)}, "elapsed_fraction": index / 10, "affect": affect(offer=f"h{index}")}
            result = wrapper.decide("S1", f"r{index}", data)
            self.assertIn(result["action"]["kind"], {"offer", "accept"})
            self.assertEqual(result["affect"]["model_id"], "frozen-plus-session-revision-1")
        self.assertEqual(len(wrapper.solver.opponent_history), 9)
        self.assertEqual(wrapper.solver.previous_arousal, .3)
        self.assertEqual(wrapper.solver.previous_valance, -.4)
        accepting = adapter_from_config(self.original_config())
        data = {"offer": {"offer_id": "generous-human", "bid": dict.fromkeys(order, 0)}, "elapsed_fraction": .1, "affect": affect(offer="generous-human")}
        accepted = accepting.decide("S1", "accept-1", data)
        self.assertEqual(accepted["action"], {"kind": "accept", "actor": "agent", "offer_id": "generous-human"})
        self.assertEqual(accepted["mood"], "Happy")


if __name__ == "__main__":
    unittest.main()
