# Development and Runtime Setup

Use this guide to prepare the software accompanying the [Solver-CL study](../README.md), connect the matching lab environment, and maintain its dependencies. The public-data verifier, perception model and robot bridge use separate environments. See [runtime setup](runtime.md) for the headless launcher and [tested environments](tested-environments.md) for SDK/model profiles.

## Independent verification

`requirements.lock` provides the hashed Python 3.12 NumPy/SciPy environment for public-data verification. `requirements-test.lock` adds Flask and pytest for API/software checks and scikit-learn for the Solver fixtures. Camera, robot and TensorFlow integrations use inert substitutes in the API tests. `requirements-analysis.lock` provides the optional modelling/plotting stack.

Install a lock with `python -m pip install --require-hashes -r <lockfile>`, using its own virtual environment. To regenerate a lock after a reviewed dependency change, use `uv pip compile <requirements-file> --python-version 3.12 --generate-hashes --output-file <lockfile>`. Run the verifier and relevant tests before accepting it.

The result reference is bound to public commit `bd30cdd79ea05a4d23a1840e05e511dc7e0b9b04`. Numerical comparison uses relative tolerance 1e-8 and absolute tolerance 1e-10. Data identity uses canonical keyed records, so file order and equivalent numeric formatting may change without accepting altered observations. Raw byte hashes are retained in the reference and source-preservation manifest. Update result references only as part of an explicit data or analysis version change.

The Fisher summary includes sessions with at least four observations, nonzero variance in both dimensions, and finite |r| < 1-1e-12. This descriptive summary has 63 CL and 66 FC sessions in the frozen data. It is not a paired Fisher test. Pooled round correlations retain all logged rows and are a different estimand.

## Historical runtime boundaries

The study runtime declarations are in `src_code/solver_cl_agent/requirements.txt` and `src_code/solver_cl_agent/FaceChannel/requirements.txt`. Prepare the robot and perception components in separate, compatible environments rather than combining these files into the modern analysis environment. The NAOqi ALProxy bridge requires its matching legacy SDK/interpreter; modern qi bindings expose a different API. TensorFlow 2.4's tested builds use Python 3.6-3.8.

A compatible runtime requires the original camera/CLI/speech/robot/GUI/logger adapters, their assets, the actual SDK build and firmware identity, and the original model manifest. Historical adapter candidates exist in `umutcakan/Human-Robot-Negotiation` branch `onur-experiment-continual` at `69df5c6cfcd8e396eee3f0b7084b97df64968787`; a matching camera signature does not establish full method, interpreter, device or redistribution compatibility. No code is imported automatically from that repository.

Source-preservation tests bind the Solver, GWR and condition-training implementation to the study code. Changes to numerical methods, model behavior or condition assignment need a separate version and validation; source identity checks alone do not validate a live lab session.

The camera-read maintenance fix keeps `None` as `None` before frame conversion,
so the existing capture loop skips a failed read. The protected-source manifest
retains the original Session hash and records the patched hash; the test reverses
that exact guard and checks the complete original file. Valid-frame conversion,
preprocessing and model/training code are unchanged.

## Local API security

The perception server binds to `127.0.0.1`, with debug disabled. Its local API accepts the study client's GET and POST calls and prediction keys. Browser requests with a foreign Origin or cross-origin Fetch Metadata are rejected before experiment side effects. Unexpected Host headers are rejected. These controls protect the local browser boundary; local processes running as the same user remain within the local trust boundary.

To require a token, set `SOLVER_API_TOKEN` in the process environment and send it as `X-Solver-Token` or `Authorization: Bearer ...`. Never put tokens in URLs, source, logs or committed files. Token-enabled clients must send a header; the legacy test UI does not acquire a token automatically.

Additional Host names must be listed explicitly in comma-separated `SOLVER_ALLOWED_HOSTS` and require a token. Without a token, non-loopback callers are rejected even if an operator starts a different WSGI server. These settings do not provide TLS or justify exposing the development server on the internet. Legacy state-changing GET endpoints remain available to preserve the robot client contract; a future POST-only migration must update the original client and server together.

Experiment IDs are canonical nonnegative integers. A URL ID such as `001` and JSON ID `1` refer to the same existing session, preventing accidental reinitialization through two key types. Request bodies are limited to 16 KiB.

## Model assets and persistence

A model already installed at the expected location is loaded as before. Missing model files never trigger an unverified implicit download. Explicit automatic acquisition requires both `SOLVER_ALLOW_MODEL_DOWNLOAD=1` and an operator-verified `SOLVER_FACECHANNEL_ARCHIVE_SHA256`. The author-hosted archive URL is pinned to commit `ec81e508bbc9e95276740202c26d0cbb5a7a20d3`; this source pin alone does not establish the study's model identity.

Before installation, the archive digest must match. Extraction accepts only regular files/directories under `TrainedNetworks`, rejects traversal/links/special files, bounds sizes, stages content, and refuses to replace differing existing assets. A trusted digest is still essential: the installer does not certify model contents or make arbitrary H5/graph files safe. Keep the original inner model hashes and demonstrate legacy inference parity before any TensorFlow/Keras migration.

GWR state still uses the original local pickle format to preserve compatibility. Accept only locally produced trusted state, with private directory permissions; never load state supplied by a network user or third party. Switching to a non-executable format requires its own state-schema and numerical parity work.

Keep new participant captures, learned GWR state, virtual environments, archives and verification outputs in a private workspace. The repository ignore rules cover their conventional generated paths; study figures and the public derived CSVs remain part of the companion.

## Dependency security and CI

Audit the root verification, test and optional analysis locks separately from the lab TF/NAOqi manifests. Review each legacy patch against interpreter, dependency and model/device constraints before installation. Keep unresolved findings visible until the relevant environment has been validated.

See the dated [dependency review](dependency-review.md) for alert priorities and compatibility constraints, and [tested environments](tested-environments.md) for the separately tested SDK/model profiles.

CI runs the complete source-preservation, software/API/compatibility and public-data verification test suite on Python 3.12 and Ubuntu 24.04. The runner label is fixed to the tested OS release; GitHub still [maintains the image within that release](https://github.com/actions/runner-images#image-releases). It excludes participant imagery, checkpoints and raw logs from checkout and uploads no artifacts or caches. The public-data verifier also remains available as a separate command; these checks do not add new scientific analyses.

This submission repository is maintained as one root commit. Automatic version-update and security-fix PRs are disabled; Dependabot alerts remain enabled. Compatible maintenance is reviewed and incorporated into a new submission snapshot directly. Historical TF/NAOqi constraints remain visible and their unresolved alerts are not dismissed.

The [dataset guide](../final_dataset/README.md) explains the supplied data, and [runtime setup](runtime.md) describes the process adapters and live-session requirements.
