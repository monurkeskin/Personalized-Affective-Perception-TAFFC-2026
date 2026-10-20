# Solver-CL Runtime Integration

`submission_runtime` runs the study's Solver from explicit decision packets and sends its decisions to separately configured device workers. A trusted provider supplies the study pipeline's raw and normalized affect snapshots through the contract below. The study entry point is `src_code/solver_cl_agent/main.py`.

Start with the software example below, then use the policy/perception contract and robot-worker sections to connect the matching lab components. [Tested environments](tested-environments.md) lists the SDK/model versions and asset identities used in the separate local checks.

## Software check

From the repository root, use Python 3.12 in a separate environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements-test.lock
python -m submission_runtime doctor
python -m submission_runtime smoke
python -m pytest tests -q
```

`doctor` without `--connect` checks local files/configuration without importing a
robot SDK or loading a model. `smoke` runs a real subprocess with an inert device;
it checks transport and journaling, with no policy/model/hardware execution.
The compatibility tests also exercise the original Solver on artificial inputs.
Use these checks to verify the software boundary; live model and robot behavior requires the selected lab environment.

For a complete headless policy round using explicit artificial affect values:

```sh
python -m submission_runtime run --config examples/runtime/host.json < examples/runtime/decision.jsonl
```

The example uses the Fruits profile and artificial affect values as a software fixture.
For a study session, select its matching domain, condition order, calibration, clock
and model configuration. Each configured worker
command is a trusted local argv list; no shell is evaluated. Run these commands
from the repository root. Replace `python` with the full interpreter path in your
private configuration when workers use different environments.

## Policy and perception contract

The policy worker is launched with `python -m submission_runtime.worker --config
<policy.json>`. Its configuration explicitly names the source root, original
profile, issue order, domain totals and `elapsed_fraction` clock semantics.
Compatibility facades preserve the original utility calculation, selector,
acceptance result and mood. Repeated request IDs return the same acknowledged
decision; conflicting reuse fails rather than advancing Solver state twice.

The host explicitly prepares the default policy before consuming decision packets.
Initial imports and centroid setup use a separate bounded preparation timeout
(`policy_setup_timeout_seconds`, default 60 seconds). Per-decision and device
timeouts retain their configured values. `doctor --connect` performs metadata
preflight only; it does not prepare the policy or load a perception model.

Each decision includes the pending offer and **separate raw and normalized**
`Arousal`/`Valance` dictionaries. The snapshot identifies its session, offer,
capture window, model, source observation time and normalization state. Field-name
translation does not change values, reorder heads, clip normalized values or
apply a new normalization rule. The two legacy clock method names expose the
explicitly configured normalized elapsed fraction, not remaining seconds.

The legacy perception HTTP `/stop` response contains a mean pair and cannot by
itself supply both channels. The compatibility boundary rejects an incomplete
snapshot. Connect a trusted lab provider that exposes the existing raw and
normalized values at the original round boundary. It must retain the original
CL/FC selection, training eligibility, model publication and capture lifecycle.
Incomplete snapshots are rejected rather than filled with zero; normalization and
training remain responsibilities of the study perception pipeline.

The original camera/TF/Keras runtime remains separate from the modern host and
policy software-test environment. Installing `requirements-test.lock` does not
install TensorFlow, camera SDKs or robot SDKs. The legacy requirements files are
preserved as source constraints; they are not a combined modern installation
recipe. Existing checkpoints/model manifests must match the selected runtime.

## Optional robot worker

The standalone `submission_runtime/bridges/naoqi_legacy.py` script runs with the
matching Python 2.7/NAOqi SDK. It imports that SDK only when explicitly prepared.
Configure it as a separate output worker; do not import it into the modern host
or install the modern package into the robot's interpreter. Use its `--help` for
the exact address/runtime options. Robot behavior names must already be installed
and explicitly mapped by the existing lab presentation layer.
Stiffness is a separate opt-in device operation; decision packets cannot request
it as part of an acknowledged presentation.

The host's `output` section uses the same argv/runtime/capabilities schema as
`policy`. Run `python -m submission_runtime doctor --config runtime.local.json --connect` only when you intend
to connect the selected workers. A hello/preflight receipt validates the reported
runtime and capabilities. Then verify audible speech, visible gestures and the
selected embodiment on the actual device. Worker selection is explicit; a different
robot, avatar or speech component needs its own compatible worker and lab check.

For robot delivery, pass `--journal runtime-output/<new-session>.runtime.jsonl`
and include the original presentation text and any selected gesture/face in each
decision packet. The host forwards the returned mood and supplied text. It records
attempted, acknowledged and unknown delivery; it does not generate a replacement
bid or automatically repeat an uncertain device operation. A nonempty previous
journal blocks a new launch using that path; inspect the record and create a new
session instead of replaying uncertain delivery.
An exclusive run claim also prevents simultaneous processes from using the same
journal. A crash can leave its `.run.lock` file in place; preserve the record and
use a new session path rather than automatically removing that claim.

Asset manifests are checked before configured workers connect. Files must be
confined to the declared root and match their SHA-256 digests. There is no model
download in the runtime launcher. Keep `runtime.local.json`, runtime journals,
camera captures and learned session state private; the supplied ignore rules
cover their conventional paths.

The transport is an original stdlib implementation compatible with the version-1
JSON-lines process boundary reviewed in [NEGOTIATOR](https://github.com/monurkeskin/NEGOTIATOR-IJCAI-2024/tree/a89a92c4140493b070795d8f136ec936b1ccf253).
No NEGOTIATOR policy/provider/presenter code is copied into this MIT repository.
Actual camera/model/robot/SDK behavior requires a device-specific lab check in
addition to the software tests above.
