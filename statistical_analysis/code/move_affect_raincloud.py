#!/usr/bin/env python3
"""
Ultra-Compact Move-Affect Raincloud: 1x2 layout (Arousal | Valence)
Each panel shows Subject-Level first, then Round-Level.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, mannwhitneyu

# Colors
FC_COLOR = '#5B9BD5'
CL_COLOR = '#ED7D31'

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'final_evaluation_results.csv')

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Condition'] = df['Condition'].replace({'FaceChannel': 'FC'})
    return df

def add_split_violin(ax, data, position, color, side='left'):
    """Add a half-violin (left or right side)."""
    parts = ax.violinplot([data], positions=[position], showmeans=False, 
                          showmedians=False, widths=0.5)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        if side == 'left':
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], -np.inf, m)
        else:
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], m, np.inf)
    for key in ['cbars', 'cmins', 'cmaxes']:
        if key in parts:
            parts[key].set_visible(False)

def add_boxplot(ax, data, position):
    """Add a boxplot with white fill, no outliers."""
    bp = ax.boxplot([data], positions=[position], widths=0.12, 
                   patch_artist=True, manage_ticks=False,
                   showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    for whisker in bp['whiskers']:
        whisker.set_color('black')
    for cap in bp['caps']:
        cap.set_color('black')
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

def generate_ultra_compact(df):
    """
    Ultra-compact 1x2 layout:
    - Panel 1: Arousal (Subject-Level | Round-Level)
    - Panel 2: Valence (Subject-Level | Round-Level)
    Subject-Level comes FIRST.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
    
    # Get counts for labels
    n_fc = len(df[df['Condition'] == 'FC'])
    n_cl = len(df[df['Condition'] == 'CL'])
    
    for ax, measure in zip(axes, ['Norm_Arousal', 'Norm_Valence']):
        ax.set_facecolor('white')
        measure_name = 'Arousal' if 'Arousal' in measure else 'Valence'
        
        # Get data
        fc_data = df[df['Condition'] == 'FC'][['Subject', measure]].dropna()
        cl_data = df[df['Condition'] == 'CL'][['Subject', measure]].dropna()
        
        # Round-level
        fc_round = fc_data[measure].values
        cl_round = cl_data[measure].values
        
        # Subject-level
        fc_subj = fc_data.groupby('Subject')[measure].mean()
        cl_subj = cl_data.groupby('Subject')[measure].mean()
        
        # Positions: Subject FIRST (0,1), Round SECOND (3,4)
        pos_fc_s, pos_cl_s = 0, 1
        pos_fc_r, pos_cl_r = 3, 4
        
        # --- SUBJECT LEVEL (FIRST) ---
        add_split_violin(ax, fc_subj.values, pos_fc_s, FC_COLOR, 'left')
        add_split_violin(ax, cl_subj.values, pos_cl_s, CL_COLOR, 'right')
        add_boxplot(ax, fc_subj.values, pos_fc_s)
        add_boxplot(ax, cl_subj.values, pos_cl_s)
        
        # Subject scatter + paired lines
        jitter_fc = np.random.uniform(-0.06, 0.02, len(fc_subj))
        jitter_cl = np.random.uniform(-0.02, 0.06, len(cl_subj))
        ax.scatter(pos_fc_s + jitter_fc, fc_subj.values, c='black', s=6, alpha=0.5, zorder=5)
        ax.scatter(pos_cl_s + jitter_cl, cl_subj.values, c='black', s=6, alpha=0.5, zorder=5)
        
        common = set(fc_subj.index) & set(cl_subj.index)
        for subj in common:
            ax.plot([pos_fc_s + 0.06, pos_cl_s - 0.06], [fc_subj[subj], cl_subj[subj]], 
                   color='gray', alpha=0.25, linewidth=0.4, zorder=1)
        
        # Subject-level significance
        common_list = list(common)
        fc_paired = [fc_subj[s] for s in common_list]
        cl_paired = [cl_subj[s] for s in common_list]
        _, p_subj = ttest_rel(fc_paired, cl_paired)
        sig_s = '***' if p_subj < 0.001 else ('**' if p_subj < 0.01 else ('*' if p_subj < 0.05 else ''))
        if sig_s:
            y_max = max(fc_subj.max(), cl_subj.max()) * 1.02
            ax.plot([pos_fc_s, pos_fc_s, pos_cl_s, pos_cl_s], 
                   [y_max, y_max + 0.02, y_max + 0.02, y_max], 'k-', lw=1.2)
            ax.text(0.5, y_max + 0.03, sig_s, ha='center', fontsize=10, fontweight='bold')
        
        # --- ROUND LEVEL (SECOND) ---
        add_split_violin(ax, fc_round, pos_fc_r, FC_COLOR, 'left')
        add_split_violin(ax, cl_round, pos_cl_r, CL_COLOR, 'right')
        add_boxplot(ax, fc_round, pos_fc_r)
        add_boxplot(ax, cl_round, pos_cl_r)
        
        # Round-level significance
        _, p_round = mannwhitneyu(fc_round, cl_round, alternative='two-sided')
        sig_r = '***' if p_round < 0.001 else ('**' if p_round < 0.01 else ('*' if p_round < 0.05 else ''))
        if sig_r:
            y_max = max(fc_round.max(), cl_round.max()) * 1.02
            ax.plot([pos_fc_r, pos_fc_r, pos_cl_r, pos_cl_r], 
                   [y_max, y_max + 0.02, y_max + 0.02, y_max], 'k-', lw=1.2)
            ax.text(3.5, y_max + 0.03, sig_r, ha='center', fontsize=10, fontweight='bold')
        
        # --- STYLING ---
        ax.set_title(f'{measure_name}', fontsize=12, fontweight='bold')
        
        # X-axis labels - clear FC/CL under each violin
        ax.set_xticks([pos_fc_s, pos_cl_s, pos_fc_r, pos_cl_r])
        ax.set_xticklabels(['FC', 'CL', 'FC', 'CL'], fontsize=9)
        
        # Add group labels below with proper N_FC, N_CL notation
        ax.text(0.5, -0.12, r'Subject-Level' + '\n' + r'$N$=66', 
                ha='center', fontsize=8, transform=ax.get_xaxis_transform())
        ax.text(3.5, -0.12, r'Round-Level' + '\n' + r'$N_{FC}$=' + str(n_fc) + r', $N_{CL}$=' + str(n_cl), 
                ha='center', fontsize=8, transform=ax.get_xaxis_transform())
        
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylabel(measure_name, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # Make room for group labels
    
    # Save
    output_dir = os.path.join(SCRIPT_DIR, '..', '..', 'assets')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/MoveAffect_Combined_Raincloud.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(f'{output_dir}/MoveAffect_Combined_Raincloud.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_dir}/MoveAffect_Combined_Raincloud.png")
    print(f"Saved: {output_dir}/MoveAffect_Combined_Raincloud.pdf")
    
    plt.close()


def generate_subject_level(df):
    """Generate Subject-Level only raincloud (Arousal | Valence)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), facecolor='white')
    
    for ax, measure in zip(axes, ['Norm_Arousal', 'Norm_Valence']):
        ax.set_facecolor('white')
        measure_name = 'Arousal' if 'Arousal' in measure else 'Valence'
        
        # Get data
        fc_data = df[df['Condition'] == 'FC'][['Subject', measure]].dropna()
        cl_data = df[df['Condition'] == 'CL'][['Subject', measure]].dropna()
        
        # Subject-level
        fc_subj = fc_data.groupby('Subject')[measure].mean()
        cl_subj = cl_data.groupby('Subject')[measure].mean()
        
        # Positions
        pos_fc, pos_cl = 0, 1
        
        # Add violins and boxplots
        add_split_violin(ax, fc_subj.values, pos_fc, FC_COLOR, 'left')
        add_split_violin(ax, cl_subj.values, pos_cl, CL_COLOR, 'right')
        add_boxplot(ax, fc_subj.values, pos_fc)
        add_boxplot(ax, cl_subj.values, pos_cl)
        
        # Subject scatter + paired lines
        jitter_fc = np.random.uniform(-0.06, 0.02, len(fc_subj))
        jitter_cl = np.random.uniform(-0.02, 0.06, len(cl_subj))
        ax.scatter(pos_fc + jitter_fc, fc_subj.values, c='black', s=12, alpha=0.5, zorder=5)
        ax.scatter(pos_cl + jitter_cl, cl_subj.values, c='black', s=12, alpha=0.5, zorder=5)
        
        common = set(fc_subj.index) & set(cl_subj.index)
        for subj in common:
            ax.plot([pos_fc + 0.06, pos_cl - 0.06], [fc_subj[subj], cl_subj[subj]], 
                   color='gray', alpha=0.3, linewidth=0.6, zorder=1)
        
        # Subject-level significance
        common_list = list(common)
        fc_paired = [fc_subj[s] for s in common_list]
        cl_paired = [cl_subj[s] for s in common_list]
        _, p_subj = ttest_rel(fc_paired, cl_paired)
        sig_s = '***' if p_subj < 0.001 else ('**' if p_subj < 0.01 else ('*' if p_subj < 0.05 else ''))
        if sig_s:
            # Use fixed position for bracket (at 0.52)
            y_bracket = 0.52
            ax.plot([pos_fc, pos_fc, pos_cl, pos_cl], 
                   [y_bracket - 0.02, y_bracket, y_bracket, y_bracket - 0.02], 'k-', lw=1.2)
            ax.text(0.5, y_bracket + 0.01, sig_s, ha='center', fontsize=12, fontweight='bold')
        
        # Styling
        ax.set_title(f'{measure_name}', fontsize=14, fontweight='bold')
        ax.set_xticks([pos_fc, pos_cl])
        ax.set_xticklabels(['FC', 'CL'], fontsize=11)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.4, 0.6)
        ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
        ax.set_yticklabels(['-0.4', '-0.2', '0.0', '0.2', '0.4', ''])  # Hide 0.6
        ax.set_ylabel(measure_name, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_dir = os.path.join(SCRIPT_DIR, '..', '..', 'assets')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/MoveAffect_Raincloud_SubjectLevel.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_dir}/MoveAffect_Raincloud_SubjectLevel.png")
    plt.close()


def generate_round_level(df):
    """Generate Round-Level only raincloud (Arousal | Valence)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), facecolor='white')
    
    n_fc = len(df[df['Condition'] == 'FC'])
    n_cl = len(df[df['Condition'] == 'CL'])
    
    for ax, measure in zip(axes, ['Norm_Arousal', 'Norm_Valence']):
        ax.set_facecolor('white')
        measure_name = 'Arousal' if 'Arousal' in measure else 'Valence'
        
        # Get data
        fc_data = df[df['Condition'] == 'FC'][measure].dropna().values
        cl_data = df[df['Condition'] == 'CL'][measure].dropna().values
        
        # Positions
        pos_fc, pos_cl = 0, 1
        
        # Add violins and boxplots
        add_split_violin(ax, fc_data, pos_fc, FC_COLOR, 'left')
        add_split_violin(ax, cl_data, pos_cl, CL_COLOR, 'right')
        add_boxplot(ax, fc_data, pos_fc)
        add_boxplot(ax, cl_data, pos_cl)
        
        # Round-level significance
        _, p_round = mannwhitneyu(fc_data, cl_data, alternative='two-sided')
        sig_r = '***' if p_round < 0.001 else ('**' if p_round < 0.01 else ('*' if p_round < 0.05 else ''))
        if sig_r:
            y_max = max(fc_data.max(), cl_data.max()) * 1.02
            ax.plot([pos_fc, pos_fc, pos_cl, pos_cl], 
                   [y_max, y_max + 0.03, y_max + 0.03, y_max], 'k-', lw=1.2)
            ax.text(0.5, y_max + 0.04, sig_r, ha='center', fontsize=12, fontweight='bold')
        
        # Styling
        ax.set_title(f'{measure_name}', fontsize=14, fontweight='bold')
        ax.set_xticks([pos_fc, pos_cl])
        ax.set_xticklabels(['FC', 'CL'], fontsize=11)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(measure_name, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_dir = os.path.join(SCRIPT_DIR, '..', '..', 'assets')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/MoveAffect_Raincloud_RoundLevel.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_dir}/MoveAffect_Raincloud_RoundLevel.png")
    plt.close()


if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rounds (FC: {len(df[df['Condition']=='FC'])}, CL: {len(df[df['Condition']=='CL'])})")
    print()
    
    # Generate separate figures
    print("\nGenerating separate Subject-Level figure...")
    generate_subject_level(df)
    print("✓ Subject-Level raincloud complete!")
    
    print("\nGenerating separate Round-Level figure...")
    generate_round_level(df)
    print("✓ Round-Level raincloud complete!")

