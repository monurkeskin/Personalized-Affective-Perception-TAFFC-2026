#!/usr/bin/env python3
"""
Move-Affect Coherence Analysis: Phase 1
========================================
Analyzes the relationship between negotiation move behaviors and affective states.

Research Questions:
- Q1: Do different move types exhibit distinct affect signatures?
- Q2: Is the Move-Affect relationship stronger in CL vs FC?

Output:
- Statistical tables (ANOVA, effect sizes, and A-V correlations as CSV files)
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f_oneway, ttest_rel, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 70)
print("MOVE-AFFECT COHERENCE ANALYSIS: PHASE 1")
print("=" * 70)

# Load the merged dataset
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'affective_behavioral_merged.csv')
df = pd.read_csv(DATA_PATH)

print(f"\nLoaded: {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Filter out rows without Move_Type
df_valid = df[df['Move_Type'].notna() & (df['Move_Type'] != '')].copy()
print(f"Valid rows with Move_Type: {len(df_valid)}")

# Standardize Move_Type names
df_valid['Move_Type'] = df_valid['Move_Type'].str.strip().str.title()
print(f"\nMove Types: {df_valid['Move_Type'].unique()}")

# ============================================================================
# 2. ROUND-LEVEL ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("ROUND-LEVEL ANALYSIS (N = {})".format(len(df_valid)))
print("=" * 70)

# 2.1 Descriptive Statistics: Mean Arousal/Valence per Move_Type × Condition
print("\n--- Descriptive Statistics: Mean Arousal/Valence per Move_Type × Condition ---\n")

# Group by Move_Type and Condition
grouped = df_valid.groupby(['Move_Type', 'Condition']).agg({
    'Norm_Arousal': ['mean', 'std', 'count'],
    'Norm_Valence': ['mean', 'std', 'count']
}).round(4)

print(grouped.to_string())

# Create summary table for paper
print("\n--- Summary Table (for paper) ---\n")
summary_rows = []
for move_type in df_valid['Move_Type'].unique():
    for condition in ['FC', 'CL']:
        subset = df_valid[(df_valid['Move_Type'] == move_type) & (df_valid['Condition'] == condition)]
        if len(subset) > 0:
            summary_rows.append({
                'Move_Type': move_type,
                'Condition': condition,
                'N': len(subset),
                'Arousal_Mean': subset['Norm_Arousal'].mean(),
                'Arousal_SD': subset['Norm_Arousal'].std(),
                'Valence_Mean': subset['Norm_Valence'].mean(),
                'Valence_SD': subset['Norm_Valence'].std()
            })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# 2.2 ANOVA: Effect of Move_Type on Arousal/Valence (within each condition)
print("\n--- ANOVA: Move_Type Effect on Arousal (per Condition) ---\n")

for condition in ['FC', 'CL']:
    cond_data = df_valid[df_valid['Condition'] == condition]
    groups = [cond_data[cond_data['Move_Type'] == mt]['Norm_Arousal'].values 
              for mt in cond_data['Move_Type'].unique()]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        f_stat, p_val = f_oneway(*groups)
        # Calculate eta-squared
        ss_between = sum(len(g) * (np.mean(g) - cond_data['Norm_Arousal'].mean())**2 for g in groups)
        ss_total = sum((cond_data['Norm_Arousal'] - cond_data['Norm_Arousal'].mean())**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        print(f"{condition}: F = {f_stat:.3f}, p = {p_val:.4f}, η² = {eta_sq:.4f}")

print("\n--- ANOVA: Move_Type Effect on Valence (per Condition) ---\n")

for condition in ['FC', 'CL']:
    cond_data = df_valid[df_valid['Condition'] == condition]
    groups = [cond_data[cond_data['Move_Type'] == mt]['Norm_Valence'].values 
              for mt in cond_data['Move_Type'].unique()]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        f_stat, p_val = f_oneway(*groups)
        # Calculate eta-squared
        ss_between = sum(len(g) * (np.mean(g) - cond_data['Norm_Valence'].mean())**2 for g in groups)
        ss_total = sum((cond_data['Norm_Valence'] - cond_data['Norm_Valence'].mean())**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        print(f"{condition}: F = {f_stat:.3f}, p = {p_val:.4f}, η² = {eta_sq:.4f}")

# 2.3 CL vs FC comparison per Move_Type
print("\n--- CL vs FC Comparison per Move_Type (Independent t-test) ---\n")
print(f"{'Move_Type':<15} {'Arousal_FC':>10} {'Arousal_CL':>10} {'p':>8} {'d':>8} {'Valence_FC':>10} {'Valence_CL':>10} {'p':>8} {'d':>8}")
print("-" * 100)

comparison_results = []
for move_type in sorted(df_valid['Move_Type'].unique()):
    fc_data = df_valid[(df_valid['Move_Type'] == move_type) & (df_valid['Condition'] == 'FC')]
    cl_data = df_valid[(df_valid['Move_Type'] == move_type) & (df_valid['Condition'] == 'CL')]
    
    if len(fc_data) > 1 and len(cl_data) > 1:
        # Arousal
        t_ar, p_ar = ttest_ind(cl_data['Norm_Arousal'], fc_data['Norm_Arousal'])
        pooled_std_ar = np.sqrt(((len(cl_data)-1)*cl_data['Norm_Arousal'].std()**2 + 
                                  (len(fc_data)-1)*fc_data['Norm_Arousal'].std()**2) / 
                                 (len(cl_data) + len(fc_data) - 2))
        d_ar = (cl_data['Norm_Arousal'].mean() - fc_data['Norm_Arousal'].mean()) / pooled_std_ar if pooled_std_ar > 0 else 0
        
        # Valence
        t_va, p_va = ttest_ind(cl_data['Norm_Valence'], fc_data['Norm_Valence'])
        pooled_std_va = np.sqrt(((len(cl_data)-1)*cl_data['Norm_Valence'].std()**2 + 
                                  (len(fc_data)-1)*fc_data['Norm_Valence'].std()**2) / 
                                 (len(cl_data) + len(fc_data) - 2))
        d_va = (cl_data['Norm_Valence'].mean() - fc_data['Norm_Valence'].mean()) / pooled_std_va if pooled_std_va > 0 else 0
        
        print(f"{move_type:<15} {fc_data['Norm_Arousal'].mean():>10.3f} {cl_data['Norm_Arousal'].mean():>10.3f} {p_ar:>8.4f} {d_ar:>8.3f} "
              f"{fc_data['Norm_Valence'].mean():>10.3f} {cl_data['Norm_Valence'].mean():>10.3f} {p_va:>8.4f} {d_va:>8.3f}")
        
        comparison_results.append({
            'Move_Type': move_type,
            'N_FC': len(fc_data), 'N_CL': len(cl_data),
            'Arousal_FC': fc_data['Norm_Arousal'].mean(),
            'Arousal_CL': cl_data['Norm_Arousal'].mean(),
            'Arousal_p': p_ar, 'Arousal_d': d_ar,
            'Valence_FC': fc_data['Norm_Valence'].mean(),
            'Valence_CL': cl_data['Norm_Valence'].mean(),
            'Valence_p': p_va, 'Valence_d': d_va
        })

# ============================================================================
# 3. SUBJECT-LEVEL ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("SUBJECT-LEVEL ANALYSIS")
print("=" * 70)

# Aggregate per subject: mean Arousal/Valence for each Move_Type
subject_move_agg = df_valid.groupby(['Subject', 'Condition', 'Move_Type']).agg({
    'Norm_Arousal': 'mean',
    'Norm_Valence': 'mean'
}).reset_index()

# For paired analysis, pivot to get CL and FC side by side
print("\n--- Subject-Level Paired Analysis (CL vs FC within-subject) ---\n")

for move_type in ['Concession', 'Selfish', 'Fortunate', 'Unfortunate']:
    move_data = subject_move_agg[subject_move_agg['Move_Type'] == move_type]
    
    # Pivot to get paired data
    pivot_ar = move_data.pivot(index='Subject', columns='Condition', values='Norm_Arousal')
    pivot_va = move_data.pivot(index='Subject', columns='Condition', values='Norm_Valence')
    
    # Get subjects with both conditions
    if 'FC' in pivot_ar.columns and 'CL' in pivot_ar.columns:
        paired_ar = pivot_ar.dropna()
        paired_va = pivot_va.dropna()
        
        if len(paired_ar) >= 5:
            t_ar, p_ar = ttest_rel(paired_ar['CL'], paired_ar['FC'])
            d_ar = (paired_ar['CL'].mean() - paired_ar['FC'].mean()) / paired_ar['CL'].std() if paired_ar['CL'].std() > 0 else 0
            
            t_va, p_va = ttest_rel(paired_va['CL'], paired_va['FC'])
            d_va = (paired_va['CL'].mean() - paired_va['FC'].mean()) / paired_va['CL'].std() if paired_va['CL'].std() > 0 else 0
            
            print(f"{move_type}: N={len(paired_ar)} paired subjects")
            print(f"  Arousal: FC={paired_ar['FC'].mean():.3f}, CL={paired_ar['CL'].mean():.3f}, p={p_ar:.4f}, d={d_ar:.3f}")
            print(f"  Valence: FC={paired_va['FC'].mean():.3f}, CL={paired_va['CL'].mean():.3f}, p={p_va:.4f}, d={d_va:.3f}")
            print()

# ============================================================================
# 4. AROUSAL-VALENCE CORRELATION BY MOVE TYPE
# ============================================================================

print("\n" + "=" * 70)
print("AROUSAL-VALENCE CORRELATION BY MOVE TYPE")
print("=" * 70)

print(f"\n{'Move_Type':<15} {'Condition':>10} {'r':>8} {'p':>8} {'N':>8}")
print("-" * 55)

for move_type in sorted(df_valid['Move_Type'].unique()):
    for condition in ['FC', 'CL']:
        subset = df_valid[(df_valid['Move_Type'] == move_type) & (df_valid['Condition'] == condition)]
        if len(subset) > 10:
            r, p = stats.pearsonr(subset['Norm_Arousal'], subset['Norm_Valence'])
            print(f"{move_type:<15} {condition:>10} {r:>8.3f} {p:>8.4f} {len(subset):>8}")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

output_path = os.path.join(SCRIPT_DIR, 'analysis')
os.makedirs(output_path, exist_ok=True)

# Save comparison results
if comparison_results:
    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(f"{output_path}/move_affect_comparison.csv", index=False)
    print(f"\nSaved: {output_path}/move_affect_comparison.csv")

# Save summary
summary_df.to_csv(f"{output_path}/move_affect_summary.csv", index=False)
print(f"Saved: {output_path}/move_affect_summary.csv")

print("\n" + "=" * 70)
print("PHASE 1 ANALYSIS COMPLETE")
print("=" * 70)
