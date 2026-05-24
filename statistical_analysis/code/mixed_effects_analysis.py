#!/usr/bin/env python3
"""
Mixed-Effects Statistical Analysis
===================================
Addresses reviewer feedback on nested-data independence by using:
1. Linear Mixed-Effects Models (random intercepts per subject)
2. Cluster-robust standard errors
3. Mediation analysis: Arousal → Move Type

This properly accounts for the hierarchical structure:
- Rounds nested within Sessions nested within Subjects
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'affective_behavioral_merged.csv')

print("=" * 80)
print("MIXED-EFFECTS ANALYSIS: Addressing Nested-Data Independence")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

# Exclude Round 1 (no delta)
df = df[df['Round'] != 1]

# Clean Move_Type
df = df[df['Move_Type'].notna() & (df['Move_Type'] != '')].copy()
df['Move_Type'] = df['Move_Type'].str.strip().str.title()

# Focus on main scientific moves
main_moves = ['Concession', 'Selfish', 'Fortunate', 'Unfortunate', 'Nice', 'Silent']
df = df[df['Move_Type'].isin(main_moves)]

# Create session key for nesting
df['Session_Key'] = df['Subject'] + '_' + df['Condition']

print(f"\nData Summary:")
print(f"  Total rounds: {len(df)}")
print(f"  Subjects: {df['Subject'].nunique()}")
print(f"  FC rounds: {len(df[df['Condition'] == 'FC'])}")
print(f"  CL rounds: {len(df[df['Condition'] == 'CL'])}")

# ============================================================================
# ANALYSIS 1: AROUSAL BY CONDITION (Mixed-Effects)
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 1: Arousal by Condition (Mixed-Effects Model)")
print("  Model: Arousal ~ Condition + (1|Subject)")
print("=" * 80)

# Mixed-effects model with random intercept per subject
model_arousal = smf.mixedlm(
    "Norm_Arousal ~ C(Condition, Treatment('FC'))", 
    data=df, 
    groups=df["Subject"]
)
result_arousal = model_arousal.fit()

print("\nMixed-Effects Results (Arousal ~ Condition):")
print("-" * 60)
print(f"{'Parameter':<25} {'Coef':>10} {'SE':>10} {'z':>10} {'p':>10}")
print("-" * 60)
for param in result_arousal.params.index:
    coef = result_arousal.params[param]
    se = result_arousal.bse[param]
    z = result_arousal.tvalues[param]
    p = result_arousal.pvalues[param]
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"{param:<25} {coef:>10.4f} {se:>10.4f} {z:>10.3f} {p:>10.4f} {sig}")

print(f"\nRandom Effect (Subject) Variance: {result_arousal.cov_re.iloc[0,0]:.4f}")
print(f"ICC (Intraclass Correlation): {result_arousal.cov_re.iloc[0,0] / (result_arousal.cov_re.iloc[0,0] + result_arousal.scale):.3f}")

# ============================================================================
# ANALYSIS 2: VALENCE BY CONDITION (Mixed-Effects)
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 2: Valence by Condition (Mixed-Effects Model)")
print("  Model: Valence ~ Condition + (1|Subject)")
print("=" * 80)

model_valence = smf.mixedlm(
    "Norm_Valence ~ C(Condition, Treatment('FC'))", 
    data=df, 
    groups=df["Subject"]
)
result_valence = model_valence.fit()

print("\nMixed-Effects Results (Valence ~ Condition):")
print("-" * 60)
print(f"{'Parameter':<25} {'Coef':>10} {'SE':>10} {'z':>10} {'p':>10}")
print("-" * 60)
for param in result_valence.params.index:
    coef = result_valence.params[param]
    se = result_valence.bse[param]
    z = result_valence.tvalues[param]
    p = result_valence.pvalues[param]
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"{param:<25} {coef:>10.4f} {se:>10.4f} {z:>10.3f} {p:>10.4f} {sig}")

# ============================================================================
# ANALYSIS 3: AROUSAL BY MOVE TYPE × CONDITION (Mixed-Effects)
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 3: Arousal by Move Type × Condition (Mixed-Effects)")
print("  Model: Arousal ~ Move_Type * Condition + (1|Subject)")
print("=" * 80)

# Focus on main moves for cleaner analysis
df_main = df[df['Move_Type'].isin(['Concession', 'Selfish', 'Fortunate', 'Unfortunate'])]

model_move = smf.mixedlm(
    "Norm_Arousal ~ C(Move_Type) * C(Condition)", 
    data=df_main, 
    groups=df_main["Subject"]
)
result_move = model_move.fit()

print("\nMixed-Effects Results (Arousal ~ Move × Condition):")
print("-" * 70)
print(f"{'Parameter':<40} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>10}")
print("-" * 70)
for param in result_move.params.index:
    coef = result_move.params[param]
    se = result_move.bse[param]
    z = result_move.tvalues[param]
    p = result_move.pvalues[param]
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"{param:<40} {coef:>8.4f} {se:>8.4f} {z:>8.3f} {p:>10.4f} {sig}")

# ============================================================================
# ANALYSIS 4: MEDIATION - Does Arousal Predict Subsequent Move?
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 4: Mediation - Arousal → Subsequent Move Type")
print("  Binary model: P(Concession) ~ Prev_Arousal + Round + (1|Subject)")
print("=" * 80)

# Create lagged arousal and binary move outcome
df_med = df.copy()
df_med = df_med.sort_values(['Session_Key', 'Round'])
df_med['Prev_Arousal'] = df_med.groupby('Session_Key')['Norm_Arousal'].shift(1)
df_med['Prev_Valence'] = df_med.groupby('Session_Key')['Norm_Valence'].shift(1)
df_med['Is_Concession'] = (df_med['Move_Type'] == 'Concession').astype(int)
df_med['Is_Selfish'] = (df_med['Move_Type'] == 'Selfish').astype(int)
df_med = df_med.dropna(subset=['Prev_Arousal', 'Prev_Valence'])

print(f"\nValid rounds for mediation: {len(df_med)}")

for condition in ['FC', 'CL']:
    print(f"\n--- {condition} Condition ---")
    cond_df = df_med[df_med['Condition'] == condition]
    
    # Logistic mixed-effects: P(Concession) ~ Prev_Arousal + Round
    # Using GEE (Generalized Estimating Equations) for cluster-robust inference
    try:
        gee_model = smf.gee(
            "Is_Concession ~ Prev_Arousal + Prev_Valence + Round",
            groups="Subject",
            data=cond_df,
            family=sm.families.Binomial()
        )
        gee_result = gee_model.fit()
        
        print(f"\n  GEE Results (cluster-robust): P(Concession) ~ Prev_Affect")
        print("  " + "-" * 60)
        for param in gee_result.params.index:
            coef = gee_result.params[param]
            se = gee_result.bse[param]
            z = gee_result.tvalues[param]
            p = gee_result.pvalues[param]
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"    {param:<20} β={coef:>7.4f}, SE={se:.4f}, z={z:>6.2f}, p={p:.4f} {sig}")
        
        # Odds ratios
        print(f"\n  Odds Ratios (exponentiated coefficients):")
        print(f"    Prev_Arousal: OR = {np.exp(gee_result.params['Prev_Arousal']):.3f}")
        print(f"    Prev_Valence: OR = {np.exp(gee_result.params['Prev_Valence']):.3f}")
        
    except Exception as e:
        print(f"  Error fitting GEE: {e}")

# ============================================================================
# ANALYSIS 5: PARTICIPANT-LEVEL SUMMARY (Paired t-test)
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS 5: Participant-Level Analysis (Aggregated)")
print("  Paired t-test on subject means")
print("=" * 80)

# Aggregate to subject level
subj_means = df.groupby(['Subject', 'Condition']).agg({
    'Norm_Arousal': 'mean',
    'Norm_Valence': 'mean'
}).reset_index()

fc_subj = subj_means[subj_means['Condition'] == 'FC'].set_index('Subject')
cl_subj = subj_means[subj_means['Condition'] == 'CL'].set_index('Subject')

common = set(fc_subj.index) & set(cl_subj.index)
print(f"\nN = {len(common)} subjects")

for measure in ['Norm_Arousal', 'Norm_Valence']:
    fc_vals = [fc_subj.loc[s, measure] for s in common]
    cl_vals = [cl_subj.loc[s, measure] for s in common]
    
    t_stat, p_val = stats.ttest_rel(fc_vals, cl_vals)
    fc_mean = np.mean(fc_vals)
    cl_mean = np.mean(cl_vals)
    
    # Effect size (Cohen's d)
    diff = np.array(cl_vals) - np.array(fc_vals)
    d = np.mean(diff) / np.std(diff)
    
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
    
    name = 'Arousal' if 'Arousal' in measure else 'Valence'
    print(f"\n{name}:")
    print(f"  FC mean: {fc_mean:.4f}")
    print(f"  CL mean: {cl_mean:.4f}")
    print(f"  Paired t({len(common)-1}) = {t_stat:.3f}, p = {p_val:.4f} {sig}")
    print(f"  Cohen's d = {d:.3f}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: Effect Significance with Proper Clustering")
print("=" * 80)

print("""
| Analysis                        | Method              | p-value   | Significant? |
|---------------------------------|---------------------|-----------|--------------|
| Arousal: FC vs CL               | Mixed-Effects LMM   | {:.4f}    | {}           |
| Valence: FC vs CL               | Mixed-Effects LMM   | {:.4f}    | {}           |
| Arousal ~ Subject (paired)      | Paired t-test       | see above | see above    |
| Concession ~ Prev_Arousal (FC)  | GEE (clustered)     | see above | see above    |
| Concession ~ Prev_Arousal (CL)  | GEE (clustered)     | see above | see above    |
""".format(
    result_arousal.pvalues.get("C(Condition, Treatment('FC'))[T.CL]", np.nan),
    '***' if result_arousal.pvalues.get("C(Condition, Treatment('FC'))[T.CL]", 1) < 0.001 else 'No',
    result_valence.pvalues.get("C(Condition, Treatment('FC'))[T.CL]", np.nan),
    '***' if result_valence.pvalues.get("C(Condition, Treatment('FC'))[T.CL]", 1) < 0.001 else 'No',
))

print("\n" + "=" * 80)
print("MIXED-EFFECTS ANALYSIS COMPLETE")
print("=" * 80)
