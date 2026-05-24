# Statistical Analysis Archive

This folder contains all materials for the statistical analysis supporting the paper:

**"Personalized Affective Perception via Continual Learning in Human-Agent Negotiation"**

## Folder Structure

```
statistical_analysis/
├── data/                               # Source data files (4 files)
│   ├── experiment_subject_data.csv     # Subject-level outcomes & moves (N=66)
│   ├── final_evaluation_results.csv    # Round-level affect data (N=2,114)
│   ├── affective_behavioral_merged.csv # Round-level merged data
│   └── move_affect_comparison.csv      # Move × Affect interaction data
├── code/                               # Analysis scripts (12 files)
│   ├── merge_session_logs.py           # Reconstructs clean round-level datasets from raw logs
│   ├── calculate_nash_distance.py      # Computes game-theoretic Pareto and Nash distance metrics
│   ├── comprehensive_lmm_analysis.py   # Main statistical LMM regressions & paired t-tests
│   ├── mixed_effects_analysis.py       # Mixed-effects modeling & random intercept tests
│   ├── within_subject_mediation.py     # Montoya-Hayes within-subject dual-pathway bootstrap mediation
│   ├── generate_outcome_figures.py     # Pre-generates NegotiationOutcomes & BehavioralAnalysis PNGs
│   ├── generate_bid_space.py           # Pre-generates the Negotiation Bid Space diagram PNG
│   ├── mixed_effects_figures.py        # Generates LMM diagnostics and mixed effects coefficients plots
│   ├── move_affect_raincloud.py        # Pre-generates Round-Level & Subject-Level raincloud plots
│   ├── move_affect_causal.py           # Exploratory causal pathway analysis
│   ├── move_affect_coherence.py        # Group-level ANOVA & CL vs FC affect difference comparisons
│   └── raw_data_verification.py        # Raw uncorrupted data integrity checks
└── README.md                           # This file
```

## Data Sources

| File | Description | Level | Key Variables |
|------|-------------|-------|---------------|
| `experiment_subject_data.csv` | Main subject outcomes | Subject (N=66) | Agent_Utility, User_Utility, Agreement_Rounds, Nash_Distance, Move rates |
| `final_evaluation_results.csv` | Affect predictions | Round (N=2,114) | Norm_Arousal, Norm_Valence, Subject, Condition, Round |
| `affective_behavioral_merged.csv` | Merged affect+moves | Round | Move_Type, Arousal, Valence |
| `move_affect_comparison.csv` | Move×Condition stats | Aggregated | Move type means, effect sizes |

## Key Findings (Verified)

| Metric | FC | CL | p | d | Status |
|--------|----|----|---|---|--------|
| **Arousal** (Subject) | 0.02±0.07 | **0.14±0.16** | <.001 | **0.69** | ✅ SIG*** |
| **Agent Utility** | 0.73±0.08 | **0.75±0.06** | .017 | **0.31** | ✅ SIG* |
| Agreement Rounds | 8.56±5.02 | 7.61±3.83 | .13 | -0.19 | - n.s. |
| **Concession Rate** | 0.28±0.20 | **0.37±0.27** | .013 | **0.32** | ✅ SIG* |
| **A-V Correlation** | r = -0.15 | **r = 0.45** | <.001 | -- | ✅ SIG*** |

## Report Contents (11 Analyses)

1. **Statistical Methodology**: Power analysis, normality tests, test selection
2. **Affect Perception**: Arousal and valence by condition
3. **A-V Correlations**: Circumplex coherence analysis
4. **Negotiation Outcomes**: Utility, rounds, Nash distance
5. **Behavioral Moves**: Move type proportions
6. **Move × Condition**: Affect during different moves
7. **Causal Pathway**: Affect -> Behavior -> Outcomes
8. **Temporal Evolution**: Effect across negotiation timeline
9. **Counterbalancing**: Order effects check (p = .167, n.s.)
10. **Individual Differences**: Responder subgroup analysis
11. **Mediation**: Concession as potential mediator (no evidence)

## Statistical Methodology

1. **Normality**: Shapiro-Wilk tests on difference scores
2. **Normal data**: Paired t-tests with Cohen's d
3. **Non-normal data**: Wilcoxon Signed-Rank with Rank-Biserial r
4. **Hierarchical data**: Linear Mixed-Effects Models (ICC = 0.24)
5. **Power**: N=66, α=0.05, power=0.80 → detects d≥0.35
