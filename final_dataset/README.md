# Public derived study data

Two CSV exports represent a counterbalanced within-participant study: 66 participants, each with one CL and one FC session (132 sessions). `experiment_round_data.csv` contains 2,114 logged offer positions: 980 CL and 1,134 FC. These rows are clustered within participants/sessions.

## Subject summaries

`experiment_subject_data.csv` has one row per `Subject_ID`, `Session_ID` and `Condition`. Outcome fields include `Agent_Utility`, `Human_Utility`, `Agreement_Rounds`, and the six participant move rates. Raw/normalized affect summary columns are retained as exported. Nash-distance and questionnaire fields are not included in this CSV.

The existing subject IDs, condition assignment, observations and numeric values are unchanged. Pair conditions using `Subject_ID`; do not rely on row order. The exported order comprises 34 CL-first and 32 FC-first participants.

## Logged offer positions

`experiment_round_data.csv` uses `Subject_ID`, `Session_ID`, `Round_ID` and `Condition`, with affect, bidder, utility, move and scaled-time fields. The first two positions in each session contain zero affect entries. Missing bidder/utility/time and actor-specific move fields must retain their export semantics; the verifier does not impute or silently drop them.

The `Norm_Arousal` and `Norm_Valence` fields are exported model estimates. This package does not reconstruct their preprocessing or establish accuracy against independently annotated affect. Group participant summaries using both `Subject_ID` and `Condition`; condition/session identity is also validated across the two files.

## Verification

From the repository root, run `python verify_all_results.py` in the independent locked verification environment. It checks the frozen public data and its declared result subset. It does not reconstruct raw logs, calculate missing questionnaire/Nash fields, execute the models, or validate all manuscript analyses. See the root README for installation, scope and result definitions.
