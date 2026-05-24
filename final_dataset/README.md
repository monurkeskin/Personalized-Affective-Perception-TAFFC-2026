# Data Repository for "Personalized Affective Perception via Continual Learning in Human-Agent Negotiation"

This repository contains the anonymized experimental data (N=66) used to reproduce the findings in the paper Personalized Affective Perception via Continual Learning in Human-Agent Negotiation.

## Experiment Overview

- **Design**: Within-subjects user study (N=66) comparing two conditions:
  1. **CLIFER (CL)**: A continual learning-enabled Solver agent that personalizes affect perception in real-time.
  2. **FaceChannel (FC)**: A static baseline agent using a pre-trained generalized affect model.
- **Sample Size**: 66 paired sessions (Total 132 sessions).
- **Data Points**: 2,114 round-level affective predictions.
- **Statistical Power**: N=66 provides power ≥0.80 for detecting effects as small as d=0.35.

## Dataset Files

### 1. Subject-Level Outcomes (`experiment_subject_data.csv`)

Contains aggregated metrics for RQ1-RQ4.
**Key columns:**

- `Condition`: **CL** (CLIFER) vs **FC** (FaceChannel).
- `Agent_Utility`: Normalized utility score (0.0 - 1.0) achieved by the agent.
- `User_Utility`: Normalized utility score (0.0 - 1.0) achieved by the human participant.
- `Agreement_Rounds`: Efficiency metric (number of rounds to agreement).
- `Nash_Distance`: Metric for joint optimality (Euclidean distance to the Nash Bargaining Solution on the Pareto frontier).
- `Concession_Rate`: Frequency of concession moves (Stepwise Analysis). (p=0.013, d=0.32).
- `Selfish_Rate`, `Unfortunate_Rate`, `Fortunate_Rate`, `Nice_Rate`, `Silent_Rate`: Other behavioral move categories.

### 2. Round-Level Affect (`experiment_round_data.csv`)

Contains round-by-round affective predictions for RQ5.
**Key columns:**

- `Condition`: **CL** vs **FC**.
- `Norm_Arousal`: Normalized prediction in range **[-1.0, 1.0]**.
- `Norm_Valence`: Normalized prediction in range **[-1.0, 1.0]**.
- `Raw_Arousal`, `Raw_Valence`: Raw affect predictions before session z-scoring.
- `Agent_Move`, `User_Move`: Move categories captured for the agent and user at each turn.
- **Note on Subject IDs**: Subject IDs in this file are assigned alphabetically to ensure round-level anonymity. Aggregation should be performed by grouping `Condition` and `Subject_ID`.
