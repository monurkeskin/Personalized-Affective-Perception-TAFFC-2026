#!/usr/bin/env python3
"""
Move-Affect Causal Analysis: Phase 2
=====================================
Analyzes directional/causal relationships between affect and negotiation moves.

Research Questions:
- Q1: Can Affect at round r predict Move at round r+1?
- Q2: Does Move at round r predict Δ_Affect at round r+1?

Methods:
- Multinomial logistic regression for Move prediction
- Linear regression / ANOVA for Δ_Affect prediction
- Lagged cross-correlation analysis
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 70)
print("MOVE-AFFECT CAUSAL ANALYSIS: PHASE 2")
print("=" * 70)

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'affective_behavioral_merged.csv')
df = pd.read_csv(DATA_PATH)

# Filter valid moves
df_valid = df[df['Move_Type'].notna() & (df['Move_Type'] != '')].copy()
df_valid['Move_Type'] = df_valid['Move_Type'].str.strip().str.title()

# Focus on main 4 moves for classification
main_moves = ['Concession', 'Selfish', 'Fortunate', 'Unfortunate']
df_main = df_valid[df_valid['Move_Type'].isin(main_moves)].copy()

print(f"\nLoaded {len(df_main)} rounds with main move types")
print(f"Move distribution:\n{df_main['Move_Type'].value_counts()}")

# ============================================================================
# 2. CREATE LAGGED FEATURES
# ============================================================================

print("\n" + "=" * 70)
print("CREATING LAGGED FEATURES")
print("=" * 70)

# Sort by Subject, Session, Round
df_main = df_main.sort_values(['Subject', 'Condition', 'Session', 'Round']).reset_index(drop=True)

# Create session key for grouping
df_main['Session_Key'] = df_main['Subject'] + '_' + df_main['Session'].astype(str) + '_' + df_main['Condition']

# Create lagged features (affect at previous round)
df_main['Prev_Arousal'] = df_main.groupby('Session_Key')['Norm_Arousal'].shift(1)
df_main['Prev_Valence'] = df_main.groupby('Session_Key')['Norm_Valence'].shift(1)
df_main['Prev_Move'] = df_main.groupby('Session_Key')['Move_Type'].shift(1)

# Create next round features (for reverse direction)
df_main['Next_Arousal'] = df_main.groupby('Session_Key')['Norm_Arousal'].shift(-1)
df_main['Next_Valence'] = df_main.groupby('Session_Key')['Norm_Valence'].shift(-1)

# Delta affect (change from current to next round)
df_main['Delta_Arousal'] = df_main['Next_Arousal'] - df_main['Norm_Arousal']
df_main['Delta_Valence'] = df_main['Next_Valence'] - df_main['Norm_Valence']

print(f"Created lagged features")
print(f"Valid rows (with previous round): {df_main['Prev_Arousal'].notna().sum()}")

# ============================================================================
# 3. ANALYSIS 1: AFFECT → MOVE PREDICTION
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: Can Affect(r) predict Move(r+1)?")
print("=" * 70)

for condition in ['FC', 'CL']:
    print(f"\n--- {condition} Condition ---")
    
    cond_df = df_main[(df_main['Condition'] == condition) & 
                       df_main['Prev_Arousal'].notna()].copy()
    
    if len(cond_df) < 50:
        print(f"  Insufficient data: {len(cond_df)} rows")
        continue
    
    # Features: Previous Arousal and Valence
    X = cond_df[['Prev_Arousal', 'Prev_Valence']].values
    
    # Target: Current Move Type
    le = LabelEncoder()
    y = le.fit_transform(cond_df['Move_Type'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Multinomial Logistic Regression
    clf = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    
    # Fit final model for coefficients
    clf.fit(X_scaled, y)
    
    # Baseline: predict most common class
    baseline_acc = cond_df['Move_Type'].value_counts(normalize=True).iloc[0]
    
    print(f"  N = {len(cond_df)}")
    print(f"  Baseline accuracy (majority): {baseline_acc:.3f}")
    print(f"  Logistic Regression CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Improvement over baseline: {(cv_scores.mean() - baseline_acc) * 100:.1f}%")
    
    # Feature importance (coefficients)
    print(f"\n  Coefficients (Arousal, Valence) per class:")
    for i, class_name in enumerate(le.classes_):
        coef_ar = clf.coef_[i, 0]
        coef_va = clf.coef_[i, 1]
        print(f"    {class_name:15s}: Arousal={coef_ar:+.3f}, Valence={coef_va:+.3f}")

# ============================================================================
# 4. ANALYSIS 2: MOVE → Δ_AFFECT PREDICTION  
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: Does Move(r) predict Δ_Affect(r+1)?")
print("=" * 70)

for condition in ['FC', 'CL']:
    print(f"\n--- {condition} Condition ---")
    
    cond_df = df_main[(df_main['Condition'] == condition) & 
                       df_main['Delta_Arousal'].notna()].copy()
    
    if len(cond_df) < 50:
        print(f"  Insufficient data: {len(cond_df)} rows")
        continue
    
    # Group by Move Type and compute mean Delta Affect
    delta_stats = cond_df.groupby('Move_Type').agg({
        'Delta_Arousal': ['mean', 'std', 'count'],
        'Delta_Valence': ['mean', 'std']
    }).round(4)
    
    print(f"\n  Mean Δ_Affect by Move Type:")
    print(f"  {'Move_Type':<15} {'Δ_Arousal':>12} {'Δ_Valence':>12} {'N':>8}")
    print(f"  {'-'*50}")
    
    for move in main_moves:
        if move in delta_stats.index:
            da = delta_stats.loc[move, ('Delta_Arousal', 'mean')]
            dv = delta_stats.loc[move, ('Delta_Valence', 'mean')]
            n = delta_stats.loc[move, ('Delta_Arousal', 'count')]
            print(f"  {move:<15} {da:>+12.4f} {dv:>+12.4f} {int(n):>8}")
    
    # ANOVA: Move Type effect on Delta Arousal
    groups = [cond_df[cond_df['Move_Type'] == m]['Delta_Arousal'].values 
              for m in main_moves if m in cond_df['Move_Type'].values]
    
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\n  ANOVA (Move → Δ_Arousal): F={f_stat:.3f}, p={p_val:.4f}")
        
        # Effect size (eta-squared)
        all_data = cond_df['Delta_Arousal']
        ss_between = sum(len(g) * (np.mean(g) - all_data.mean())**2 for g in groups)
        ss_total = ((all_data - all_data.mean())**2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        print(f"  Effect size: η² = {eta_sq:.4f}")

# ============================================================================
# 5. ANALYSIS 3: LAGGED CROSS-CORRELATION
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 3: Lagged Cross-Correlation")
print("=" * 70)

for condition in ['FC', 'CL']:
    print(f"\n--- {condition} Condition ---")
    
    cond_df = df_main[df_main['Condition'] == condition].copy()
    
    # Encode Move_Type numerically (for correlation)
    # Concession = -1 (cooperative), Selfish = +1 (competitive)
    move_scores = {'Concession': -1, 'Fortunate': 0.5, 'Unfortunate': -0.5, 'Selfish': 1}
    cond_df['Move_Score'] = cond_df['Move_Type'].map(move_scores)
    
    # Correlation: Current Arousal → Next Move
    valid = cond_df[['Norm_Arousal', 'Move_Score']].dropna()
    if len(valid) > 10:
        r_ar_move, p = stats.pearsonr(valid['Norm_Arousal'], valid['Move_Score'])
        print(f"  Arousal(r) → Move_Score(r): r = {r_ar_move:+.3f}, p = {p:.4f}")
    
    # Lagged: Previous Arousal → Current Move
    cond_df['Prev_Move_Score'] = cond_df.groupby('Session_Key')['Move_Score'].shift(1)
    valid_lag = cond_df[['Prev_Arousal', 'Move_Score']].dropna()
    if len(valid_lag) > 10:
        r_prev_ar, p = stats.pearsonr(valid_lag['Prev_Arousal'], valid_lag['Move_Score'])
        print(f"  Arousal(r-1) → Move_Score(r): r = {r_prev_ar:+.3f}, p = {p:.4f}")
    
    # Reverse: Move → Next Arousal
    valid_rev = cond_df[['Move_Score', 'Next_Arousal']].dropna()
    if len(valid_rev) > 10:
        r_move_ar, p = stats.pearsonr(valid_rev['Move_Score'], valid_rev['Next_Arousal'])
        print(f"  Move_Score(r) → Arousal(r+1): r = {r_move_ar:+.3f}, p = {p:.4f}")

# ============================================================================
# 6. ANALYSIS 4: TRANSITION MATRIX (Move Sequences)
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS 4: Move Transition Probabilities")
print("=" * 70)

for condition in ['FC', 'CL']:
    print(f"\n--- {condition} Condition ---")
    
    cond_df = df_main[(df_main['Condition'] == condition) & 
                       cond_df['Prev_Move'].notna()].copy()
    
    if len(cond_df) < 50:
        continue
    
    # Transition matrix
    transitions = pd.crosstab(cond_df['Prev_Move'], cond_df['Move_Type'], normalize='index')
    
    print(f"\n  P(Move(r+1) | Move(r)):")
    print(transitions.round(3).to_string())
    
    # Chi-square test for independence
    contingency = pd.crosstab(cond_df['Prev_Move'], cond_df['Move_Type'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    cramers_v = np.sqrt(chi2 / (len(cond_df) * (min(contingency.shape) - 1)))
    print(f"\n  Chi-square test: χ²={chi2:.2f}, p={p:.4f}, Cramér's V={cramers_v:.3f}")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

output_path = os.path.join(SCRIPT_DIR, 'analysis')
os.makedirs(output_path, exist_ok=True)

# Save lagged dataset
df_main.to_csv(os.path.join(output_path, 'move_affect_lagged.csv'), index=False)
print(f"\nSaved: {output_path}/move_affect_lagged.csv")

print("\n" + "=" * 70)
print("PHASE 2 ANALYSIS COMPLETE")
print("=" * 70)
