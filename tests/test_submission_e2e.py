"""Real Solver subprocess integration with inert, explicitly fake device output.

Tests named ``actual`` run the production host and default policy worker,
including the unchanged submission Solver. Their output worker is inert. The
separately named synthetic startup regression uses an artificial policy only to
check independent startup/decision deadlines. No robot, camera, SDK or model is
contacted. No test is evidence about scientific or hardware parity.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from submission_runtime.journal import SidecarJournal


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = "inert-e2e-output"

# This wrapper records ownership only, then replaces itself with the actual
# policy CLI. It provides no substitute policy, adapter, model or SDK.
POLICY_EXEC = """
import os
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii')
os.execv(sys.executable, [sys.executable, '-m', 'submission_runtime.worker',
                         '--config', sys.argv[2]])
"""


INERT_OUTPUT = """
import json
import os
from pathlib import Path
import sys
import time
from submission_runtime.bridges.protocol import serve

events = Path(sys.argv[1])
mode = sys.argv[2]

def record(kind, **values):
    event = dict(kind=kind, pid=os.getpid(), fixture='inert-output-no-hardware', **values)
    with events.open('a', encoding='utf-8') as target:
        target.write(json.dumps(event, allow_nan=False) + '\\n')
        target.flush()
        os.fsync(target.fileno())

class InertOutput:
    def handle(self, operation, payload, session_id, request_id):
        if operation == 'hello':
            record('hello', session_id=session_id)
            return {'runtime': 'inert-e2e-output',
                    'capabilities': [] if mode == 'missing-capability' else ['present'],
                    'hardware_tested': False}
        if operation != 'present':
            raise ValueError('The inert fixture supports only present.')
        # Record before acknowledging: this can stand for an effect whose reply
        # is lost, without issuing any physical or external operation.
        record('present', session_id=session_id, request_id=request_id, payload=payload)
        if mode == 'timeout-after-effect':
            time.sleep(30)
        if mode == 'eof-after-effect':
            os._exit(0)
        if mode == 'protocol-contamination':
            sys.__stdout__.write('non-json fixture contamination\\n')
            sys.__stdout__.flush()
        if mode == 'sdk-print-noise':
            print('fake SDK stdout noise: no real SDK was imported')
            print('fake SDK stderr noise', file=sys.stderr)
        return {'delivered': True, 'hardware_tested': False,
                'fixture': 'inert-output-no-hardware'}

    def close(self):
        record('close')

record('start')
serve(InertOutput())
"""


SYNTHETIC_STARTUP_POLICY = """
import json
import os
from pathlib import Path
import sys
import time
from submission_runtime.bridges.protocol import serve

events = Path(sys.argv[1])
Path(sys.argv[2]).write_text(str(os.getpid()), encoding='ascii')
slow_decision = sys.argv[3] == 'slow'

def record(kind):
    with events.open('a', encoding='utf-8') as target:
        target.write(json.dumps({'kind': kind, 'fixture': 'synthetic-startup-no-solver'}) + '\\n')
        target.flush()
        os.fsync(target.fileno())

class SyntheticStartupPolicy:
    prepared = False

    def handle(self, operation, payload, session_id, request_id):
        if operation == 'hello':
            record('hello')
            return {'runtime': 'synthetic-startup-policy', 'capabilities': ['prepare', 'decide']}
        if operation == 'prepare':
            record('prepare-start')
            time.sleep(1.1)
            self.prepared = True
            record('prepare-complete')
            return {'prepared': True, 'fixture': 'synthetic-startup-no-solver'}
        if operation != 'decide':
            raise ValueError('Unsupported fixture operation.')
        if not self.prepared:
            record('unprepared-decide')
            time.sleep(1.1)  # Reproduce cold startup wrongly charged to decision.
        else:
            record('decide')
        if slow_decision:
            time.sleep(1.1)
        return {'mood': 'Synthetic startup fixture', 'text': payload.get('text'),
                'fixture': 'synthetic-startup-no-solver'}

    def close(self):
        record('close')

serve(SyntheticStartupPolicy())
"""


def _packet(session="E2E-SESSION", request="decision-1", offer="human-1", *, generous=False):
    packet = json.loads((ROOT / "examples/runtime/decision.jsonl").read_text())
    packet["request_id"] = request
    packet["offer"]["offer_id"] = offer
    packet["affect"]["session_id"] = session
    packet["affect"]["offer_id"] = offer
    packet["affect"]["window_id"] = "window-" + offer
    packet["text"] = "Synthetic exact text: çığ \\rspd=80\\ unchanged."
    if generous:
        packet["offer"]["bid"] = dict.fromkeys(packet["offer"]["bid"], 0)
    return packet


def _case(tmp_path, *, session="E2E-SESSION", mode="normal"):
    directory = tmp_path / session
    directory.mkdir()
    events = directory / "inert-events.jsonl"
    policy_pid = directory / "actual-policy.pid"
    policy = ROOT / "examples/runtime/solver.json"
    config = {
        "schema_version": 1,
        "session_id": session,
        "asset_manifests": [],
        "policy": {
            "command": [sys.executable, "-c", POLICY_EXEC, str(policy_pid), str(policy)],
            "runtime": "solver-cl-policy",
            "capabilities": ["decide"],
            "timeout_seconds": 20,
        },
        "output": {
            "command": [sys.executable, "-u", "-c", INERT_OUTPUT, str(events), mode],
            "runtime": RUNTIME,
            "capabilities": ["present"],
            "timeout_seconds": .4 if mode == "timeout-after-effect" else 5,
        },
    }
    path = directory / "host.json"
    path.write_text(json.dumps(config))
    journal = directory / "private.runtime.jsonl"
    command = [sys.executable, "-m", "submission_runtime", "run", "--config", str(path),
               "--journal", str(journal)]
    return command, journal, events, policy_pid


def _invoke(command, packets):
    data = "".join(json.dumps(packet, allow_nan=False) + "\n" for packet in packets)
    return subprocess.run(command, cwd=ROOT, input=data, capture_output=True,
                          text=True, timeout=90)


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _results(result):
    return [json.loads(line) for line in result.stdout.splitlines()]


def _assert_owned_children_exited(events, policy_pid):
    pids = {row["pid"] for row in _events(events)}
    if policy_pid.exists():
        pids.add(int(policy_pid.read_text()))
    assert pids, "The process fixtures must actually have started."
    deadline = time.monotonic() + 2
    while pids and time.monotonic() < deadline:
        for pid in tuple(pids):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pids.remove(pid)
        if pids:
            time.sleep(.01)
    assert not pids, "Owned worker processes survived host cleanup: %r" % sorted(pids)


def test_actual_solver_two_rounds_duplicate_and_exact_inert_delivery(tmp_path):
    """Actual counteroffer/accept policy; only presentation is an inert fixture."""
    command, journal, events, pid = _case(tmp_path)
    first = _packet()
    second = _packet(request="decision-2", offer="human-2", generous=True)
    second["elapsed_fraction"] = .2
    result = _invoke(command, [first, copy.deepcopy(first), second])
    assert result.returncode == 0, result.stderr
    decisions = _results(result)
    assert len(decisions) == 3 and decisions[0] == decisions[1]
    assert decisions[0]["action"]["kind"] == "offer"
    assert decisions[2]["action"] == {"kind": "accept", "actor": "agent", "offer_id": "human-2"}
    assert decisions[2]["mood"] == "Happy"
    rows = SidecarJournal(journal).read()
    assert [row["kind"] for row in rows] == [
        "decision.returned", "presentation.attempted", "presentation.delivered",
        "decision.returned", "presentation.attempted", "presentation.delivered",
    ]
    delivered = [row for row in _events(events) if row["kind"] == "present"]
    assert len(delivered) == 2
    for original, decision, event in zip([first, second], [decisions[0], decisions[2]], delivered):
        assert event["payload"]["text"] == original["text"] == decision["text"]
        assert event["payload"]["mood"] == decision["mood"]
        assert event["fixture"] == "inert-output-no-hardware"
        assert decision["affect"]["raw"] == original["affect"]["raw"]
        assert decision["affect"]["normalized"] == original["affect"]["normalized"]
    _assert_owned_children_exited(events, pid)


def test_actual_policy_request_conflict_stops_before_second_device_effect(tmp_path):
    """The actual policy runs once; a conflicting ID cannot advance delivery."""
    command, journal, events, pid = _case(tmp_path)
    first = _packet()
    conflict = copy.deepcopy(first)
    conflict["text"] = "Conflicting synthetic text must never be presented."
    later = _packet(request="later", offer="later-offer")
    result = _invoke(command, [first, conflict, later])
    assert result.returncode != 0 and "identity" in result.stderr.lower()
    assert len(_results(result)) == 1
    assert len([row for row in _events(events) if row["kind"] == "present"]) == 1
    assert len(SidecarJournal(journal).read()) == 3
    _assert_owned_children_exited(events, pid)


@pytest.mark.parametrize("mode", ["timeout-after-effect", "eof-after-effect", "protocol-contamination"])
def test_actual_policy_unknown_inert_delivery_is_journaled_and_never_replayed(tmp_path, mode):
    """Actual Solver result precedes an inert effect with a lost/invalid reply."""
    command, journal, events, pid = _case(tmp_path, mode=mode)
    packet = _packet(generous=True)
    result = _invoke(command, [packet, packet])
    assert result.returncode != 0
    assert result.stdout == ""
    rows = SidecarJournal(journal).read()
    assert [row["kind"] for row in rows] == [
        "decision.returned", "presentation.attempted", "presentation.failed",
    ]
    assert rows[0]["payload"]["action"]["kind"] == "accept"
    assert rows[0]["payload"]["mood"] == "Happy"
    assert rows[-1]["payload"]["delivery_status"] == "unknown"
    assert len([row for row in _events(events) if row["kind"] == "present"]) == 1
    before = journal.read_bytes()
    prior_events = events.read_bytes()
    retry = _invoke(command, [packet])
    assert retry.returncode != 0 and "previous runtime journal" in retry.stderr
    assert journal.read_bytes() == before and events.read_bytes() == prior_events
    _assert_owned_children_exited(events, pid)


def test_inert_sdk_prints_cannot_contaminate_actual_policy_json_results(tmp_path):
    """Fake SDK prints are noise; the policy process remains the actual Solver."""
    command, journal, events, pid = _case(tmp_path, mode="sdk-print-noise")
    result = _invoke(command, [_packet(generous=True)])
    assert result.returncode == 0, result.stderr
    decisions = _results(result)
    assert len(decisions) == 1 and decisions[0]["mood"] == "Happy"
    assert "fake SDK" not in result.stdout
    assert [row["kind"] for row in SidecarJournal(journal).read()][-1] == "presentation.delivered"
    assert len([row for row in _events(events) if row["kind"] == "present"]) == 1
    _assert_owned_children_exited(events, pid)


def test_missing_inert_output_capability_prevents_actual_policy_decision(tmp_path):
    """Actual policy is configured/preflighted, but no core decision may occur."""
    command, journal, events, pid = _case(tmp_path, mode="missing-capability")
    result = _invoke(command, [_packet()])
    assert result.returncode != 0 and "capability" in result.stderr
    assert result.stdout == "" and not SidecarJournal(journal).read()
    assert [row["kind"] for row in _events(events)] == ["start", "hello", "close"]
    _assert_owned_children_exited(events, pid)


@pytest.mark.parametrize("data,success", [("", True), ('{"request_id":', False)])
def test_input_eof_closes_actual_policy_and_inert_output_without_effects(tmp_path, data, success):
    """EOF never substitutes a decision; both worker lifecycles close."""
    command, journal, events, pid = _case(tmp_path)
    result = subprocess.run(command, cwd=ROOT, input=data, capture_output=True,
                            text=True, timeout=90)
    assert (result.returncode == 0) is success
    assert result.stdout == "" and not SidecarJournal(journal).read()
    assert [row["kind"] for row in _events(events)] == ["start", "hello", "close"]
    assert not journal.with_suffix(journal.suffix + ".run.lock").exists()
    _assert_owned_children_exited(events, pid)


def test_independent_actual_policy_sessions_reuse_request_ids_without_crosstalk(tmp_path):
    """Two actual Solver lifecycles use inert outputs and separate private logs."""
    worker_pids = set()
    for session in ("E2E-FIRST", "E2E-SECOND"):
        command, journal, events, pid = _case(tmp_path, session=session)
        packet = _packet(session=session, generous=True)
        result = _invoke(command, [packet, packet])
        assert result.returncode == 0, result.stderr
        first, duplicate = _results(result)
        assert first == duplicate
        assert first["mood"] == "Happy" and first["affect"]["session_id"] == session
        rows = SidecarJournal(journal).read()
        assert len(rows) == 3 and {row["session_id"] for row in rows} == {session}
        presented = [row for row in _events(events) if row["kind"] == "present"]
        assert len(presented) == 1 and presented[0]["session_id"] == session
        worker_pids.add(int(pid.read_text()))
        _assert_owned_children_exited(events, pid)
    assert len(worker_pids) == 2


@pytest.mark.parametrize("decision_mode", ["fast", "slow"])
def test_synthetic_startup_budget_does_not_replace_decision_deadline(tmp_path, decision_mode):
    """Synthetic policy only: slow prepare is separate from the fast deadline."""
    command, journal, events, _ = _case(tmp_path)
    config_path = Path(command[command.index("--config") + 1])
    phases = config_path.parent / "synthetic-startup-events.jsonl"
    synthetic_pid = config_path.parent / "synthetic-policy.pid"
    config = json.loads(config_path.read_text())
    config["policy_setup_timeout_seconds"] = 5
    config["policy"] = {
        "command": [sys.executable, "-u", "-c", SYNTHETIC_STARTUP_POLICY,
                    str(phases), str(synthetic_pid), decision_mode],
        "runtime": "synthetic-startup-policy",
        "capabilities": ["decide", "prepare"],
        "timeout_seconds": .5,
    }
    config_path.write_text(json.dumps(config))
    result = _invoke(command, [_packet()])
    phase_names = [row["kind"] for row in _events(phases)]
    assert phase_names[:4] == ["hello", "prepare-start", "prepare-complete", "decide"], result.stderr
    assert "unprepared-decide" not in phase_names
    if decision_mode == "fast":
        assert result.returncode == 0, result.stderr
        assert _results(result)[0]["fixture"] == "synthetic-startup-no-solver"
        assert len([row for row in _events(events) if row["kind"] == "present"]) == 1
        assert [row["kind"] for row in SidecarJournal(journal).read()][-1] == "presentation.delivered"
    else:
        assert result.returncode != 0 and "timed out" in result.stderr.lower()
        assert result.stdout == "" and not SidecarJournal(journal).read()
        assert not [row for row in _events(events) if row["kind"] == "present"]
    _assert_owned_children_exited(events, synthetic_pid)
