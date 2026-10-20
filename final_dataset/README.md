# Solver-CL Study Data

**TL;DR:** These two anonymized, derived CSVs accompany *Personalized Affective Perception via Continual Learning in Human-Agent Negotiation*. They contain negotiation outcomes, participant moves and model-estimated affect from the study (N=66).

**Keywords:** human-robot negotiation, study data, behavioral moves, valence-arousal estimates

[Study overview](../README.md) · [Public-data verification](../docs/public-data-verification.md)

For offer-by-offer session records, including the demonstration sessions, see the
[raw-log guide](../raw_logs/README.md). It explains the workbook fields and their
relationship to these derived CSVs.

## Experiment Overview

- **Design**: Counterbalanced within-subjects user study (N=66) comparing two conditions:
  1. **CLIFER (CL)**: A continual learning-enabled Solver agent that personalizes affect perception during interaction.
  2. **FaceChannel (FC)**: A static baseline agent using a pre-trained generalized affect model.
- **Sample Size**: 66 paired participants, each with one CL and one FC session (132 sessions in total). The exported order comprises 34 CL-first and 32 FC-first participants.
- **Data Points**: 2,114 logged offer positions: 980 CL and 1,134 FC. These observations are clustered within participants and sessions.
- **Statistical Power**: The study documentation reports power ≥0.80 for detecting effects as small as d=0.35 with N=66. The public CSV verifier does not recalculate this power analysis.

## Dataset Files

### 1. Subject-Level Outcomes (`experiment_subject_data.csv`)

Contains one summary row per `Subject_ID`, `Session_ID` and `Condition`, with affect, behavioral and outcome measures relevant to RQ1–RQ4.

**Key columns:**

- `Subject_ID`: Anonymized participant identifier shared across conditions and the two CSVs. Pair CL and FC observations using this field, rather than row order.
- `Session_ID`: Session position (1 or 2); the condition at each position depends on the participant's assigned order.
- `Condition`: **CL** (CLIFER) vs **FC** (FaceChannel).
- `Agent_Utility`: Normalized utility score (0.0–1.0) achieved by the agent.
- `Human_Utility`: Normalized utility score (0.0–1.0) achieved by the human participant.
- `Agreement_Rounds`: Number of rounds to agreement, used as the negotiation-efficiency measure.
- `Concession_Rate`: Proportion of participant moves classified as concessions in the stepwise analysis. The accompanying study documentation reports p=0.013 and d=0.32 for the condition comparison; see [public CSV verification](../docs/public-data-verification.md) for the frozen CSV estimates and multiple-comparison families.
- `Selfish_Rate`, `Unfortunate_Rate`, `Fortunate_Rate`, `Nice_Rate`, `Silent_Rate`: Proportions for the other participant move categories. All six rate fields use a 0.0–1.0 scale.
- `Raw_Arousal`, `Raw_Valence`, `Norm_Arousal`, `Norm_Valence`: Exported raw and normalized affect summaries.
- `Max_Arousal`, `Min_Arousal`, `Max_Valence`, `Min_Valence`: Exported affect-extrema summary fields.

Nash distance (distance to the Nash Bargaining Solution) is discussed in the paper but is not a column in this CSV. Questionnaire observations are also absent; the verifier does not reconstruct either measure.

### 2. Round-Level Affect (`experiment_round_data.csv`)

Contains logged offer positions with affective predictions and negotiation context, supporting the affect and move analyses in RQ1–RQ2.

**Key columns:**

- `Subject_ID`, `Session_ID`, `Round_ID`: Participant, session and within-session offer-position identifiers. `Round_ID` starts at 1 and is consecutive within each session.
- `Condition`: **CL** vs **FC**; assignments match the corresponding subject-level summaries.
- `Norm_Arousal`, `Norm_Valence`: Normalized model estimates, retained as exported.
- `Raw_Arousal`, `Raw_Valence`: Raw model estimates stored alongside the normalized values.
- `Max_Arousal`, `Min_Arousal`, `Max_Valence`, `Min_Valence`: Exported affect-extrema fields.
- `Agent_Utility`, `Human_Utility`: Agent and participant utility values at the logged offer position, where present.
- `Bidder`: Actor associated with the logged offer, where present.
- `Agent_Move`, `User_Move`: Actor-specific behavioral move categories, where present.
- `Scaled_Time`: Exported scaled-time value, where present.

The first two logged positions in each session contain zero affect entries. Missing bidder, utility, time and actor-specific move fields retain their export semantics; the verifier neither imputes them nor silently drops observations. These rows should not be treated as independent participant observations.

Group participant summaries using both `Subject_ID` and `Condition`, and retain `Session_ID` when joining the two files. The normalized fields are model outputs: the public package does not reconstruct their preprocessing or establish accuracy against independently annotated affect.

## Verification

From the repository root, run `python verify_all_results.py` in the independent locked Python 3.12 verification environment. Installation instructions, checked statistics and comparison families are in [public CSV verification](../docs/public-data-verification.md).

The verifier checks the frozen public data and its declared result subset, including schemas, participant/session/condition identity and paired outcomes. It does not reconstruct raw logs, calculate missing questionnaire or Nash fields, execute models, or validate every manuscript analysis. For the study method and reported results, see the [main README](../README.md).
