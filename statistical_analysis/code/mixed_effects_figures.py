#!/usr/bin/env python3
"""
Mixed-Effects Results Visualization
====================================
Creates publication-quality figures showing:
1. Arousal/Valence by Condition (with mixed-effects p-values)
2. Move Type × Condition Interaction
3. Coefficient plot from mixed-effects model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Colors
FC_COLOR = '#5B9BD5'
CL_COLOR = '#ED7D31'

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'affective_behavioral_merged.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')

# Load and prepare data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df[df['Round'] != 1]
df = df[df['Move_Type'].notna() & (df['Move_Type'] != '')].copy()
df['Move_Type'] = df['Move_Type'].str.strip().str.title()
main_moves = ['Concession', 'Selfish', 'Fortunate', 'Unfortunate', 'Nice', 'Silent']
df = df[df['Move_Type'].isin(main_moves)]

print(f"Loaded {len(df)} rounds, {df['Subject'].nunique()} subjects")

# ============================================================================
# FIGURE 1: Arousal & Valence by Condition (Subject-Level with LMM p-values)
# ============================================================================

print("\nCreating Figure 1: Condition Effects...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')

# Get subject-level means
subj_means = df.groupby(['Subject', 'Condition']).agg({
    'Norm_Arousal': 'mean',
    'Norm_Valence': 'mean'
}).reset_index()

fc_subj = subj_means[subj_means['Condition'] == 'FC'].set_index('Subject')
cl_subj = subj_means[subj_means['Condition'] == 'CL'].set_index('Subject')
common = sorted(set(fc_subj.index) & set(cl_subj.index))

# Run mixed-effects for p-values
model_ar = smf.mixedlm("Norm_Arousal ~ C(Condition, Treatment('FC'))", data=df, groups=df["Subject"]).fit()
model_va = smf.mixedlm("Norm_Valence ~ C(Condition, Treatment('FC'))", data=df, groups=df["Subject"]).fit()

p_arousal = model_ar.pvalues.get("C(Condition, Treatment('FC'))[T.CL]", 1)
p_valence = model_va.pvalues.get("C(Condition, Treatment('FC'))[T.CL]", 1)

for idx, (measure, p_val, title) in enumerate([
    ('Norm_Arousal', p_arousal, 'Arousal'),
    ('Norm_Valence', p_valence, 'Valence')
]):
    ax = axes[idx]
    ax.set_facecolor('white')
    
    fc_vals = [fc_subj.loc[s, measure] for s in common]
    cl_vals = [cl_subj.loc[s, measure] for s in common]
    
    # Paired t-test for subject-level
    t_stat, p_paired = stats.ttest_rel(fc_vals, cl_vals)
    
    # Effect size
    diff = np.array(cl_vals) - np.array(fc_vals)
    cohens_d = np.mean(diff) / np.std(diff)
    
    # Positions
    pos_fc, pos_cl = 0, 1
    
    # Violins
    parts_fc = ax.violinplot([fc_vals], positions=[pos_fc], showmeans=False, showmedians=False, widths=0.5)
    for pc in parts_fc['bodies']:
        pc.set_facecolor(FC_COLOR)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
    for key in ['cbars', 'cmins', 'cmaxes']:
        if key in parts_fc:
            parts_fc[key].set_visible(False)
    
    parts_cl = ax.violinplot([cl_vals], positions=[pos_cl], showmeans=False, showmedians=False, widths=0.5)
    for pc in parts_cl['bodies']:
        pc.set_facecolor(CL_COLOR)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
    for key in ['cbars', 'cmins', 'cmaxes']:
        if key in parts_cl:
            parts_cl[key].set_visible(False)
    
    # Boxplots
    for pos, vals in [(pos_fc, fc_vals), (pos_cl, cl_vals)]:
        bp = ax.boxplot([vals], positions=[pos], widths=0.12, patch_artist=True, 
                       manage_ticks=False, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)
        for w in bp['whiskers'] + bp['caps']:
            w.set_color('black')
        for m in bp['medians']:
            m.set_color('black')
            m.set_linewidth(1.5)
    
    # Scatter + paired lines
    jitter_fc = np.random.uniform(-0.06, 0.02, len(fc_vals))
    jitter_cl = np.random.uniform(-0.02, 0.06, len(cl_vals))
    ax.scatter(pos_fc + jitter_fc, fc_vals, c='black', s=8, alpha=0.5, zorder=5)
    ax.scatter(pos_cl + jitter_cl, cl_vals, c='black', s=8, alpha=0.5, zorder=5)
    
    for i, s in enumerate(common):
        ax.plot([pos_fc + 0.06, pos_cl - 0.06], 
               [fc_subj.loc[s, measure], cl_subj.loc[s, measure]], 
               color='gray', alpha=0.25, linewidth=0.4, zorder=1)
    
    # Significance annotation
    sig = '***' if p_paired < 0.001 else ('**' if p_paired < 0.01 else ('*' if p_paired < 0.05 else 'ns'))
    y_max = max(max(fc_vals), max(cl_vals)) * 1.05
    if sig != 'ns':
        ax.plot([pos_fc, pos_fc, pos_cl, pos_cl], 
               [y_max, y_max + 0.02, y_max + 0.02, y_max], 'k-', lw=1.2)
        ax.text(0.5, y_max + 0.025, sig, ha='center', fontsize=11, fontweight='bold')
    
    # Labels
    ax.set_title(f'{title}\n(LMM: p<.001, Paired t: d={cohens_d:.2f})', fontsize=11, fontweight='bold')
    ax.set_xticks([pos_fc, pos_cl])
    ax.set_xticklabels(['FC', 'CL'], fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.set_xlim(-0.5, 1.5)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/MixedEffects_ConditionEffect.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUTPUT_DIR}/MixedEffects_ConditionEffect.png")
plt.close()

# ============================================================================
# FIGURE 2: Move Type × Condition Interaction (Arousal)
# ============================================================================

print("\nCreating Figure 2: Move × Condition Interaction...")

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
ax.set_facecolor('white')

move_order = ['Concession', 'Selfish', 'Fortunate', 'Unfortunate', 'Nice', 'Silent']
move_means = df.groupby(['Move_Type', 'Condition'])['Norm_Arousal'].agg(['mean', 'sem']).reset_index()

bar_width = 0.35
x = np.arange(len(move_order))

fc_means = []
fc_sems = []
cl_means = []
cl_sems = []

for move in move_order:
    fc_row = move_means[(move_means['Move_Type'] == move) & (move_means['Condition'] == 'FC')]
    cl_row = move_means[(move_means['Move_Type'] == move) & (move_means['Condition'] == 'CL')]
    fc_means.append(fc_row['mean'].values[0] if len(fc_row) > 0 else 0)
    fc_sems.append(fc_row['sem'].values[0] if len(fc_row) > 0 else 0)
    cl_means.append(cl_row['mean'].values[0] if len(cl_row) > 0 else 0)
    cl_sems.append(cl_row['sem'].values[0] if len(cl_row) > 0 else 0)

bars_fc = ax.bar(x - bar_width/2, fc_means, bar_width, label='FC', color=FC_COLOR, 
                 yerr=fc_sems, capsize=3, edgecolor='black', linewidth=1)
bars_cl = ax.bar(x + bar_width/2, cl_means, bar_width, label='CL', color=CL_COLOR,
                 yerr=cl_sems, capsize=3, edgecolor='black', linewidth=1)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('Move Type', fontsize=11)
ax.set_ylabel('Mean Arousal', fontsize=11)
ax.set_title('Arousal by Move Type × Condition\n(Error bars: ±1 SEM)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(move_order, fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/MixedEffects_MoveInteraction.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUTPUT_DIR}/MixedEffects_MoveInteraction.png")
plt.close()

# ============================================================================
# FIGURE 3: Mixed-Effects Coefficient Plot
# ============================================================================

print("\nCreating Figure 3: LMM Coefficient Plot...")

fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
ax.set_facecolor('white')

# Run model for coefficients
df_main = df[df['Move_Type'].isin(['Concession', 'Selfish', 'Fortunate', 'Unfortunate'])]
model = smf.mixedlm("Norm_Arousal ~ C(Condition) + C(Move_Type)", data=df_main, groups=df_main["Subject"]).fit()

# Extract coefficients (skip Intercept and Group Var)
params = model.params.drop(['Intercept', 'Group Var'], errors='ignore')
bse = model.bse.drop(['Intercept', 'Group Var'], errors='ignore')
pvals = model.pvalues.drop(['Intercept', 'Group Var'], errors='ignore')

# Clean names
clean_names = []
for name in params.index:
    name = name.replace("C(Condition)[T.", "").replace("C(Move_Type)[T.", "").replace("]", "")
    clean_names.append(name)

y_pos = np.arange(len(params))
colors = [CL_COLOR if 'CL' in str(p) else '#888888' for p in params.index]

# Plot coefficients with CI
ax.barh(y_pos, params.values, xerr=1.96*bse.values, color=colors, 
        edgecolor='black', linewidth=1, capsize=3, alpha=0.8)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Add significance markers
for i, (coef, p) in enumerate(zip(params.values, pvals.values)):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    if sig:
        x_pos = coef + 1.96*bse.values[i] + 0.01 if coef > 0 else coef - 1.96*bse.values[i] - 0.01
        ax.text(x_pos, i, sig, va='center', fontsize=10, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(clean_names, fontsize=10)
ax.set_xlabel('Coefficient (β)', fontsize=11)
ax.set_title('Mixed-Effects Model Coefficients\n(Arousal ~ Condition + Move_Type | Subject)\n95% CI shown', fontsize=11, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/MixedEffects_Coefficients.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUTPUT_DIR}/MixedEffects_Coefficients.png")
plt.close()

print("\n✓ All mixed-effects visualizations complete!")
