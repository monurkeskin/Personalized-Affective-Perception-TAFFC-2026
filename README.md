# Personalized Affective Perception via Continual Learning in Human-Agent Negotiation

Supplementary code and public derived data for the Solver-CL study. The public CSV verification tool runs independently of the historical robot and perception environments.

## Verify the public results

Use a separate Python 3.12 environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python verify_all_results.py
```

The command prints JSON and exits with status 0 only when every declared check passes. To save the report explicitly:

```sh
python verify_all_results.py --output-dir verification-results
```

The verifier uses `final_dataset/experiment_subject_data.csv` and `experiment_round_data.csv`: 66 paired participants, 132 sessions and 2,114 logged offer positions. It checks schemas, subject/session/condition keys, order-independent data identity, paired means and sample SDs, CL-FC differences, paired t intervals, p-values, Cohen's d_z, the stated four-outcome Holm family and descriptive correlations. Reordering rows or columns preserves the result; changing observations fails verification.

`verification/expected-results.json` binds the numeric reference to the original public-data commit and file hashes. Each statistic has its own assertion. The tool neither reconstructs raw logs nor executes a camera, robot or model. Nash distance, questionnaire, mediation, bootstrap and secondary mixed-model analyses are outside its declared coverage; they are not labelled verified.

## Study results represented by these CSVs

| Paired outcome | CL-FC difference | Unadjusted 95% CI | Raw p |
|---|---:|---|---:|
| Agent utility | +0.02235 | [0.00421, 0.04049] | 0.01654 |
| Participant concession rate | +9.35 percentage points | [2.02, 16.69] percentage points | 0.01325 |
| Participant utility | -0.00795 | [-0.03108, 0.01517] | 0.49455 |
| Agreement rounds | -0.95455 | [-2.21132, 0.30223] | 0.13415 |
| Estimated arousal | +0.12509 | [0.08012, 0.17006] | <0.001 |

Agent utility and participant concessions have Holm-adjusted p=0.03974 within the four-outcome family of estimated arousal, concessions, agent utility and agreement rounds. A separate six-move family gives p=0.07948 for participant concessions. The family and exploratory status matter when interpreting these comparisons.

The study compares the complete CL-enabled negotiation system with its FaceChannel baseline. Higher estimated arousal is a model-output difference, without independent affect ground truth. Nonsignificant participant-utility or Nash comparisons do not establish equivalence. Concession frequency alone does not establish cooperation, and exploratory mediation does not establish parallel causal mechanisms. Calibration and subsequent online adaptation are evaluated together.

Pooled round arousal-valence correlations are descriptive: CL r=0.45406 and FC r=-0.15302. They differ from participant-wise Fisher-z summaries. The verifier labels both estimands and reports finite-Fisher inclusion counts without treating them as the same statistic or a paired test.

## Repository layout

- `final_dataset/`: the two unchanged public derived CSVs and data documentation.
- `verify_all_results.py`, `verification/`: offline verification and frozen references.
- `src_code/solver_cl_agent/`: historical negotiation and perception implementation.
- `assets/`: study figures retained from the original companion.
- `tests/`: verification, security and source-preservation checks.
- `docs/maintenance.md`: environment boundaries, model acquisition and compatibility requirements.

## Robot and perception environments

The offline verification environment must remain separate from the original NAO/Pepper SDK and TensorFlow/Keras environments. Do not install all requirements files into one interpreter. The two historical runtime manifests remain unchanged pending validation against the original lab environment and checkpoints.

The current public source requires additional lab adapters and FaceChannel model assets. Installing the root requirements does not supply them. CAAE checkpoint entries exist in the Git tree, while source-focused checkouts may omit binary assets. Restore the original, trusted assets and matching adapters before attempting the historical runtime. The maintained Python 2.7/NAOqi bridge and modern `qi` bindings are different interfaces; switching between them requires device and protocol validation.

Maintenance preserves the original Solver, GWR, normalization/inference paths, thresholds, utility domains, checkpoints and study observations. Existing method/integration gaps are not silently changed by dependency or security work. `verification/protected-sources.json` checks preserved source/data/legacy constraints and the unchanged FaceChannel model methods. In differential feedback, ties in delta_V+delta_A intentionally use the documented +1 convention.

The local perception API retains GET/POST compatibility for existing loopback clients. It rejects unexpected Host and cross-origin browser requests. Remote access requires explicit host configuration and an API token. Model acquisition requires explicit opt-in and a trusted archive digest. See [maintenance guidance](docs/maintenance.md) before using either interface.

## Tests and optional analysis tools

```sh
python -m pip install --require-hashes -r requirements-test.lock
python -m pytest tests -q
```

The API tests use inert camera, robot and TensorFlow substitutes. They do not establish hardware availability, real-time latency or historical model parity. Optional statistical modelling and plotting packages have a separate `requirements-analysis.lock`; they are not required by the verifier.

CI uses one small CPU job, sparse checkout, no uploaded artifacts and no Actions cache. Dependabot version updates cover the independent root environment and Actions. Legacy runtime security alerts remain visible; their remediation requires a validated compatibility environment.

## Related components and citation

The system builds on [CLIFER](https://github.com/nchuramani/CLIFER), [FaceChannel](https://github.com/pablovin/FaceChannel) and the negotiation framework referenced in the study. Upstream model assets have their own provenance and usage terms.

```bibtex
@article{PersonalizedAffectivePerception2025,
  title   = {Personalized Affective Perception via Continual Learning in Human-Agent Negotiation},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2025}
}
```

The repository's existing license is retained in [LICENSE](LICENSE).
