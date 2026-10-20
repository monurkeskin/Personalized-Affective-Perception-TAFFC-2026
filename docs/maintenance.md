# Maintenance and compatibility

## Independent verification

`requirements.lock` is the hashed Python 3.12 NumPy/SciPy environment for the public-data verifier. `requirements-test.lock` adds Flask and pytest solely for inert API tests. `requirements-analysis.lock` provides the optional modelling/plotting stack. These environments are separate from the historical robot and ML runtimes.

Install a lock with `python -m pip install --require-hashes -r <lockfile>`, using its own virtual environment. To regenerate a lock after a reviewed dependency change, use `uv pip compile <requirements-file> --python-version 3.12 --generate-hashes --output-file <lockfile>`. Run the verifier and relevant tests before accepting it.

The result reference is bound to public commit `bd30cdd79ea05a4d23a1840e05e511dc7e0b9b04`. Numerical comparison uses relative tolerance 1e-8 and absolute tolerance 1e-10. Data identity uses canonical keyed records, so file order and equivalent numeric formatting may change without accepting altered observations. Raw byte hashes are retained in the reference and source-preservation manifest. A result-reference change is a scientific-data/version decision, not a way to silence a failed check.

The Fisher summary includes sessions with at least four observations, nonzero variance in both dimensions, and finite |r| < 1-1e-12. This descriptive summary has 63 CL and 66 FC sessions in the frozen data. It is not a paired Fisher test. Pooled round correlations retain all logged rows and are a different estimand.

## Historical runtime boundaries

`src_code/solver_cl_agent/requirements.txt` and `FaceChannel/requirements.txt` preserve the original declarations. They are not a jointly solvable modern environment or a supported installation recipe for arbitrary Python versions. The original NAOqi ALProxy bridge uses a legacy SDK/interpreter; modern qi bindings need a separate migration. TensorFlow 2.4's tested builds use Python 3.6-3.8. Neither runtime should be installed into the analysis environment.

A compatible runtime requires the original camera/CLI/speech/robot/GUI/logger adapters, their assets, the actual SDK build and firmware identity, and the original model manifest. Historical adapter candidates exist in `umutcakan/Human-Robot-Negotiation` branch `onur-experiment-continual` at `69df5c6cfcd8e396eee3f0b7084b97df64968787`; a matching camera signature does not establish full method, interpreter, device or redistribution compatibility. No code is imported automatically from that repository.

The source-preservation tests intentionally retain Solver/GWR/condition-training behavior. This prevents maintenance from changing the scientific implementation accidentally; it does not certify that all historical integration or method contracts are complete. A later correction to numerical/model/condition behavior needs its own version, evidence and review.

## Local API security

The built-in server binds explicitly to `127.0.0.1`, with debug disabled. Existing local GET and POST calls and prediction keys are preserved. Browser requests with a foreign Origin or cross-origin Fetch Metadata are rejected before experiment side effects. Unexpected Host headers are rejected. These controls protect the local browser boundary; local processes running as the same user remain within the local trust boundary.

To require a token, set `SOLVER_API_TOKEN` in the process environment and send it as `X-Solver-Token` or `Authorization: Bearer ...`. Never put tokens in URLs, source, logs or committed files. Token-enabled clients must send a header; the legacy test UI does not acquire a token automatically.

Additional Host names must be listed explicitly in comma-separated `SOLVER_ALLOWED_HOSTS` and require a token. Without a token, non-loopback callers are rejected even if an operator starts a different WSGI server. These settings do not provide TLS or justify exposing the development server on the internet. Legacy state-changing GET endpoints remain available to preserve the robot client contract; a future POST-only migration must update the original client and server together.

Experiment IDs are canonical nonnegative integers. A URL ID such as `001` and JSON ID `1` refer to the same existing session, preventing accidental reinitialization through two key types. Request bodies are limited to 16 KiB.

## Model assets and persistence

A model already installed at the expected location is loaded as before. Missing model files never trigger an unverified implicit download. Explicit automatic acquisition requires both `SOLVER_ALLOW_MODEL_DOWNLOAD=1` and an operator-verified `SOLVER_FACECHANNEL_ARCHIVE_SHA256`. The author-hosted archive URL is pinned to commit `ec81e508bbc9e95276740202c26d0cbb5a7a20d3`; this source pin alone does not establish the study's model identity.

Before installation, the archive digest must match. Extraction accepts only regular files/directories under `TrainedNetworks`, rejects traversal/links/special files, bounds sizes, stages content, and refuses to replace differing existing assets. A trusted digest is still essential: the installer does not certify model contents or make arbitrary H5/graph files safe. Keep the original inner model hashes and demonstrate legacy inference parity before any TensorFlow/Keras migration.

GWR state still uses the original local pickle format to preserve compatibility. Accept only locally produced trusted state, with private directory permissions; never load state supplied by a network user or third party. Switching to a non-executable format requires its own state-schema and numerical parity work.

New sessions, GWR state, virtual environments, archives and verification outputs are ignored by Git. Existing tracked study assets are retained. Keep future participant capture outputs in a private workspace. Removing published images or rewriting history requires a separate provenance and sharing-scope decision.

## Dependency security and CI

Root verification/test/optional analysis locks can be audited independently. A clean audit for these locks says nothing about the historical TF/NAOqi manifests. Do not dismiss unresolved legacy alerts, rename manifests to hide them, or install modern replacements into the original device environment without compatibility evidence.

CI performs only data verification, source-preservation and inert unit tests on Python 3.12. It excludes participant imagery, checkpoints and raw logs from checkout and uploads no artifacts or caches. Monthly version-update PRs are bounded; legacy runtime manifests are excluded from automatic version bumps while security alerts remain enabled.
