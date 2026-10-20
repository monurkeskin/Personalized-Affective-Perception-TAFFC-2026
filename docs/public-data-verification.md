# Reproducing the Public Data Results

Use the two public derived CSVs to reproduce the declared subject-level comparisons and descriptive affect summaries from the Solver-CL study. This guide explains installation, expected checks and their coverage; the [main README](../README.md) presents the system, method and study figures.

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

`verification/expected-results.json` binds the numeric reference to the original public-data commit and file hashes. Each statistic has its own assertion. The command operates on the derived CSVs. Raw-log reconstruction, camera/robot/model execution, Nash distance, questionnaire, mediation, bootstrap and secondary mixed-model analyses require their corresponding inputs and workflows and are outside this command's coverage.

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

## Tests and optional analysis tools

```sh
python -m pip install --require-hashes -r requirements-test.lock
python -m pytest tests -q
```

The API tests use inert camera, robot and TensorFlow substitutes. They do not establish hardware availability, real-time latency or historical model parity. Optional statistical modelling and plotting packages have a separate `requirements-analysis.lock`; they are not required by the verifier.

CI uses one small CPU job, sparse checkout, no uploaded artifacts and no Actions cache. Dependabot alerts remain enabled; automatic update PRs are disabled for this submission snapshot. Legacy runtime security alerts remain visible; their remediation requires a validated compatibility environment.

For separate SDK/model profiles and hardware boundaries, see [tested environments](tested-environments.md) and [runtime setup](runtime.md).
