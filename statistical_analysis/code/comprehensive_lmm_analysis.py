#!/usr/bin/env python3
"""
Comprehensive Mixed-Effects Analysis for Human-Agent Negotiation Study
=======================================================================
This script implements Linear Mixed-Effects Models (LMM) with cluster-robust
standard errors to address reviewer feedback on nested data dependencies.

Key Analyses:
1. Arousal/Valence by Condition (round-level with subject random effects)
2. Agent Utility by Condition (round-level and session-level)
3. Move Type proportions by Condition (subject-level)
4. Arousal-Valence correlations (subject-level)

Author: Analysis Script
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Statistical packages
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data path - Use merged data file
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'affective_behavioral_merged.csv')


def load_and_prepare_data():
    """Load and preprocess the merged affective-behavioral dataset."""
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH)
    
    # Standardize condition names
    df['Condition'] = df['Condition'].replace({'FaceChannel': 'FC'})
    
    # Create numeric condition variable (CL=1, FC=0)
    df['Condition_Num'] = (df['Condition'] == 'CL').astype(int)
    
    # Use Norm_Arousal and Norm_Valence for analysis (aligned with paper methodology)
    df['Arousal'] = df['Norm_Arousal']
    df['Valence'] = df['Norm_Valence']
    
    # Remove duplicates if any (keep first occurrence)
    df = df.drop_duplicates(subset=['Subject', 'Condition', 'Session', 'Round'])
    
    # Print data summary
    print(f"\nTotal rows: {len(df)}")
    print(f"Unique subjects: {df['Subject'].nunique()}")
    print(f"Conditions: {df['Condition'].unique()}")
    print(f"Rounds per subject-condition: ~{len(df) / (df['Subject'].nunique() * 2):.0f}")
    
    # Count by condition
    print(f"\nRows by Condition:")
    print(df.groupby('Condition').size())
    
    return df


def analyze_affect_lmm(df):
    """
    Analyze Arousal and Valence using Linear Mixed-Effects Models.
    Model: Outcome ~ Condition + (1|Subject)
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: AFFECT BY CONDITION (LMM)")
    print("=" * 70)
    
    results = {}
    
    # Round 1 IS valid for affect analysis (only moves need exclusion)
    df_valid = df.copy()
    
    for outcome in ['Arousal', 'Valence']:
        print(f"\n--- {outcome} Analysis ---")
        
        # Prepare data
        model_df = df_valid[['Subject', 'Condition_Num', outcome]].dropna()
        
        if len(model_df) < 100:
            print(f"  Insufficient data for {outcome}")
            continue
        
        try:
            # Fit LMM: Outcome ~ Condition + (1|Subject)
            model = smf.mixedlm(
                f"{outcome} ~ Condition_Num",
                model_df,
                groups=model_df['Subject']
            )
            result = model.fit(reml=True, method='lbfgs')
            
            # Extract key statistics
            coef = result.params['Condition_Num']
            se = result.bse['Condition_Num']
            z = result.tvalues['Condition_Num']
            p = result.pvalues['Condition_Num']
            
            # Effect size (Cohen's d approximation)
            residual_var = result.scale
            random_var = float(result.cov_re.iloc[0, 0])
            total_var = residual_var + random_var
            cohens_d = coef / np.sqrt(total_var)
            
            # ICC (proportion of variance due to subjects)
            icc = random_var / total_var
            
            print(f"\n  LMM Results:")
            print(f"    CL Effect: β = {coef:.4f} (SE = {se:.4f})")
            print(f"    z = {z:.3f}, p = {p:.4f}")
            print(f"    Cohen's d = {cohens_d:.3f}")
            print(f"    ICC (subject) = {icc:.3f}")
            print(f"    N observations = {len(model_df)}")
            print(f"    N subjects = {model_df['Subject'].nunique()}")
            
            results[outcome] = {
                'coef': coef, 'se': se, 'z': z, 'p': p,
                'cohens_d': cohens_d, 'icc': icc,
                'n_obs': len(model_df),
                'n_subjects': model_df['Subject'].nunique()
            }
            
        except Exception as e:
            print(f"  Error fitting LMM for {outcome}: {e}")
    
    # Also compute subject-level means for paired comparison
    print("\n--- Subject-Level Paired t-tests (for comparison) ---")
    
    subject_means = df_valid.groupby(['Subject', 'Condition']).agg({
        'Arousal': 'mean',
        'Valence': 'mean'
    }).reset_index()
    
    for outcome in ['Arousal', 'Valence']:
        fc_vals = subject_means[subject_means['Condition'] == 'FC'][outcome].values
        cl_vals = subject_means[subject_means['Condition'] == 'CL'][outcome].values
        
        if len(fc_vals) > 0 and len(cl_vals) > 0:
            t_stat, p_val = stats.ttest_rel(cl_vals, fc_vals)
            cohens_d_paired = (cl_vals.mean() - fc_vals.mean()) / np.std(cl_vals - fc_vals)
            
            print(f"\n  {outcome} (subject-level):")
            print(f"    CL mean: {np.mean(cl_vals):.4f} ± {np.std(cl_vals):.4f}")
            print(f"    FC mean: {np.mean(fc_vals):.4f} ± {np.std(fc_vals):.4f}")
            print(f"    t({len(fc_vals)-1}) = {t_stat:.3f}, p = {p_val:.4f}")
            print(f"    Cohen's d = {cohens_d_paired:.3f}")
            
            results[f'{outcome}_paired'] = {
                't': t_stat, 'p': p_val, 'd': cohens_d_paired,
                'cl_mean': np.mean(cl_vals), 'fc_mean': np.mean(fc_vals)
            }
    
    return results


def analyze_agent_utility_lmm(df):
    """
    Analyze Agent Utility using LMM at round level.
    Also compute session-level final utilities for paired t-tests.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: AGENT UTILITY BY CONDITION")
    print("=" * 70)
    
    results = {}
    
    # Round-level LMM
    print("\n--- Round-Level LMM ---")
    
    model_df = df[['Subject', 'Condition_Num', 'Agent Utility', 'Round']].dropna()
    model_df = model_df.rename(columns={'Agent Utility': 'AgentUtility'})
    
    try:
        model = smf.mixedlm(
            "AgentUtility ~ Condition_Num",
            model_df,
            groups=model_df['Subject']
        )
        result = model.fit(reml=True, method='lbfgs')
        
        coef = result.params['Condition_Num']
        se = result.bse['Condition_Num']
        z = result.tvalues['Condition_Num']
        p = result.pvalues['Condition_Num']
        
        residual_var = result.scale
        random_var = float(result.cov_re.iloc[0, 0])
        total_var = residual_var + random_var
        cohens_d = coef / np.sqrt(total_var)
        icc = random_var / total_var
        
        print(f"  CL Effect: β = {coef:.4f} (SE = {se:.4f})")
        print(f"  z = {z:.3f}, p = {p:.4f}")
        print(f"  Cohen's d = {cohens_d:.3f}")
        print(f"  ICC = {icc:.3f}")
        
        results['round_level'] = {
            'coef': coef, 'se': se, 'z': z, 'p': p,
            'cohens_d': cohens_d, 'icc': icc
        }
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Session-level (final round utility = agreement outcome)
    print("\n--- Session-Level (Final Utility) ---")
    
    final_utilities = df.groupby(['Subject', 'Condition']).apply(
        lambda x: x.loc[x['Agent Utility'].notna(), 'Agent Utility'].iloc[-1] if x['Agent Utility'].notna().any() else np.nan
    ).reset_index(name='FinalUtility')
    
    fc_final = final_utilities[final_utilities['Condition'] == 'FC']['FinalUtility'].values
    cl_final = final_utilities[final_utilities['Condition'] == 'CL']['FinalUtility'].values
    
    if len(fc_final) > 0 and len(cl_final) > 0:
        t_stat, p_val = stats.ttest_rel(cl_final, fc_final)
        cohens_d = (cl_final.mean() - fc_final.mean()) / np.std(cl_final - fc_final)
        
        print(f"  CL final utility: {np.mean(cl_final):.4f} ± {np.std(cl_final):.4f}")
        print(f"  FC final utility: {np.mean(fc_final):.4f} ± {np.std(fc_final):.4f}")
        print(f"  t({len(fc_final)-1}) = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Cohen's d = {cohens_d:.3f}")
        
        results['session_level'] = {
            't': t_stat, 'p': p_val, 'd': cohens_d,
            'cl_mean': np.mean(cl_final), 'fc_mean': np.mean(fc_final)
        }
    
    return results


def analyze_agreement_rounds(df):
    """Analyze number of rounds to agreement by condition."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: AGREEMENT ROUNDS (Efficiency)")
    print("=" * 70)
    
    # Count rounds per session
    rounds_per_session = df.groupby(['Subject', 'Condition']).agg(
        TotalRounds=('Round', 'max')
    ).reset_index()
    
    fc_rounds = rounds_per_session[rounds_per_session['Condition'] == 'FC']['TotalRounds'].values
    cl_rounds = rounds_per_session[rounds_per_session['Condition'] == 'CL']['TotalRounds'].values
    
    if len(fc_rounds) > 0 and len(cl_rounds) > 0:
        t_stat, p_val = stats.ttest_rel(cl_rounds, fc_rounds)
        cohens_d = (cl_rounds.mean() - fc_rounds.mean()) / np.std(cl_rounds - fc_rounds)
        
        print(f"  CL rounds: {np.mean(cl_rounds):.2f} ± {np.std(cl_rounds):.2f}")
        print(f"  FC rounds: {np.mean(fc_rounds):.2f} ± {np.std(fc_rounds):.2f}")
        print(f"  t({len(fc_rounds)-1}) = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Cohen's d = {cohens_d:.3f}")
        
        return {
            't': t_stat, 'p': p_val, 'd': cohens_d,
            'cl_mean': np.mean(cl_rounds), 'fc_mean': np.mean(fc_rounds)
        }
    
    return {}


def analyze_move_types(df):
    """
    Analyze behavioral move types by condition.
    Subject-level proportions with paired tests.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: BEHAVIORAL MOVES BY CONDITION")
    print("=" * 70)
    
    # Filter valid moves (exclude Round 1 and empty moves)
    df_moves = df[(df['Round'] > 1) & (df['Move_Type'].notna()) & 
                  (df['Move_Type'] != '')].copy()
    df_moves['Move_Type'] = df_moves['Move_Type'].str.strip().str.title()
    
    # Get subject-level move proportions
    move_counts = df_moves.groupby(['Subject', 'Condition', 'Move_Type']).size().reset_index(name='Count')
    total_per_subject_cond = df_moves.groupby(['Subject', 'Condition']).size().reset_index(name='Total')
    
    move_props = move_counts.merge(total_per_subject_cond, on=['Subject', 'Condition'])
    move_props['Proportion'] = move_props['Count'] / move_props['Total']
    
    # Analyze key move types
    key_moves = ['Concession', 'Selfish', 'Fortunate', 'Nice', 'Unfortunate']
    results = {}
    
    print("\n--- Subject-Level Move Proportions (Paired t-tests) ---")
    
    for move in key_moves:
        move_data = move_props[move_props['Move_Type'] == move]
        
        # Pivot to get FC and CL side by side
        pivot_data = move_data.pivot(index='Subject', columns='Condition', values='Proportion').fillna(0)
        
        if 'FC' in pivot_data.columns and 'CL' in pivot_data.columns:
            fc_vals = pivot_data['FC'].values
            cl_vals = pivot_data['CL'].values
            
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(cl_vals, fc_vals)
            diff = cl_vals - fc_vals
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            
            print(f"\n  {move}:")
            print(f"    CL: {np.mean(cl_vals)*100:.1f}% ± {np.std(cl_vals)*100:.1f}%")
            print(f"    FC: {np.mean(fc_vals)*100:.1f}% ± {np.std(fc_vals)*100:.1f}%")
            print(f"    t = {t_stat:.3f}, p = {p_val:.4f}, d = {cohens_d:.3f}")
            
            results[move] = {
                't': t_stat, 'p': p_val, 'd': cohens_d,
                'cl_mean': np.mean(cl_vals), 'fc_mean': np.mean(fc_vals)
            }
    
    return results


def analyze_arousal_valence_correlation(df):
    """
    Analyze Arousal-Valence correlations within subjects.
    Report subject-level correlations (proper unit of analysis).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 5: AROUSAL-VALENCE CORRELATIONS")
    print("=" * 70)
    
    # Round 1 IS valid for A-V correlation
    df_valid = df[df['Arousal'].notna() & df['Valence'].notna()].copy()
    
    results = {}
    
    # Subject-level correlations
    print("\n--- Subject-Level Correlations ---")
    
    subject_corrs = {}
    for condition in ['FC', 'CL']:
        df_cond = df_valid[df_valid['Condition'] == condition]
        
        corrs = {}  # keyed by subject
        for subject in df_cond['Subject'].unique():
            subj_data = df_cond[df_cond['Subject'] == subject]
            if len(subj_data) >= 3:  # Minimum data points for correlation
                r, _ = stats.pearsonr(subj_data['Arousal'], subj_data['Valence'])
                corrs[subject] = r
        
        corrs_list = list(corrs.values())
        # Fisher z-transform for mean
        z_corrs = np.arctanh(np.clip(corrs_list, -0.999, 0.999))
        mean_z = np.mean(z_corrs)
        mean_r = np.tanh(mean_z)
        
        subject_corrs[condition] = {
            'correlations': corrs,  # dict keyed by subject
            'mean_r': mean_r,
            'se': np.std(corrs_list) / np.sqrt(len(corrs_list)),
            'n': len(corrs_list),
            'arith_mean_r': np.mean(corrs_list)
        }
        
        print(f"\n  {condition}:")
        print(f"    Mean r (Fisher-z) = {mean_r:.3f}, Arithmetic mean r = {np.mean(corrs_list):.3f}")
        print(f"    SE = {np.std(corrs_list)/np.sqrt(len(corrs_list)):.3f}")
        print(f"    N subjects = {len(corrs_list)}")
        print(f"    Range: [{np.min(corrs_list):.3f}, {np.max(corrs_list):.3f}]")
    
    # Paired comparison of correlations — match by subject ID
    fc_dict = subject_corrs['FC']['correlations']
    cl_dict = subject_corrs['CL']['correlations']
    common_subjects = sorted(set(fc_dict.keys()) & set(cl_dict.keys()))
    
    fc_corrs = [fc_dict[s] for s in common_subjects]
    cl_corrs = [cl_dict[s] for s in common_subjects]
    
    # Fisher z-transform before t-test
    z_fc = np.arctanh(np.clip(fc_corrs, -0.999, 0.999))
    z_cl = np.arctanh(np.clip(cl_corrs, -0.999, 0.999))
    
    t_stat, p_val = stats.ttest_rel(z_cl, z_fc)
    d_z = np.mean(z_cl - z_fc) / np.std(z_cl - z_fc, ddof=1)
    
    print(f"\n  Comparison (paired on Fisher-z, N={len(common_subjects)}):")
    print(f"    t({len(common_subjects)-1}) = {t_stat:.3f}, p = {p_val:.4f}, d_z = {d_z:.2f}")
    
    results['subject_correlations'] = {
        'fc_mean_r': subject_corrs['FC']['mean_r'],
        'cl_mean_r': subject_corrs['CL']['mean_r'],
        'fc_arith_mean_r': subject_corrs['FC']['arith_mean_r'],
        'cl_arith_mean_r': subject_corrs['CL']['arith_mean_r'],
        't': t_stat,
        'p': p_val,
        'd_z': d_z,
        'n_paired': len(common_subjects)
    }
    
    return results


def create_summary_table(all_results):
    """Create a summary table of all robust effects."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE: ROBUST EFFECTS (LMM + Subject-Level)")
    print("=" * 70)
    
    rows = []
    
    # Affect results
    if 'affect' in all_results:
        for var in ['Arousal', 'Valence']:
            if var in all_results['affect']:
                r = all_results['affect'][var]
                rows.append({
                    'Variable': f'{var} (LMM)',
                    'Type': 'Round-Level LMM',
                    'CL_Effect': r['coef'],
                    'SE': r['se'],
                    'Test_Stat': r['z'],
                    'p_value': r['p'],
                    'Effect_Size': r['cohens_d'],
                    'Significant': r['p'] < 0.05
                })
            
            paired_key = f'{var}_paired'
            if paired_key in all_results['affect']:
                r = all_results['affect'][paired_key]
                rows.append({
                    'Variable': f'{var} (Subject)',
                    'Type': 'Paired t-test',
                    'CL_Effect': r['cl_mean'] - r['fc_mean'],
                    'SE': np.nan,
                    'Test_Stat': r['t'],
                    'p_value': r['p'],
                    'Effect_Size': r['d'],
                    'Significant': r['p'] < 0.05
                })
    
    # Agent Utility
    if 'utility' in all_results:
        if 'round_level' in all_results['utility']:
            r = all_results['utility']['round_level']
            rows.append({
                'Variable': 'Agent Utility (LMM)',
                'Type': 'Round-Level LMM',
                'CL_Effect': r['coef'],
                'SE': r['se'],
                'Test_Stat': r['z'],
                'p_value': r['p'],
                'Effect_Size': r['cohens_d'],
                'Significant': r['p'] < 0.05
            })
        
        if 'session_level' in all_results['utility']:
            r = all_results['utility']['session_level']
            rows.append({
                'Variable': 'Final Agent Utility',
                'Type': 'Paired t-test',
                'CL_Effect': r['cl_mean'] - r['fc_mean'],
                'SE': np.nan,
                'Test_Stat': r['t'],
                'p_value': r['p'],
                'Effect_Size': r['d'],
                'Significant': r['p'] < 0.05
            })
    
    # Agreement Rounds
    if 'rounds' in all_results and all_results['rounds']:
        r = all_results['rounds']
        rows.append({
            'Variable': 'Agreement Rounds',
            'Type': 'Paired t-test',
            'CL_Effect': r['cl_mean'] - r['fc_mean'],
            'SE': np.nan,
            'Test_Stat': r['t'],
            'p_value': r['p'],
            'Effect_Size': r['d'],
            'Significant': r['p'] < 0.05
        })
    
    # Moves
    if 'moves' in all_results:
        for move, r in all_results['moves'].items():
            rows.append({
                'Variable': f'{move} Move',
                'Type': 'Paired t-test',
                'CL_Effect': r['cl_mean'] - r['fc_mean'],
                'SE': np.nan,
                'Test_Stat': r['t'],
                'p_value': r['p'],
                'Effect_Size': r['d'],
                'Significant': r['p'] < 0.05
            })
    
    # Create DataFrame
    summary_df = pd.DataFrame(rows)
    
    # Print table
    print("\n")
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'lmm_summary_table.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    
    return summary_df


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MIXED-EFFECTS ANALYSIS")
    print("Human-Agent Negotiation with Continual Learning")
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Load data
    df = load_and_prepare_data()
    
    # Store all results
    all_results = {}
    
    # 1. Affect Analysis (LMM)
    all_results['affect'] = analyze_affect_lmm(df)
    
    # 2. Agent Utility Analysis
    all_results['utility'] = analyze_agent_utility_lmm(df)
    
    # 3. Agreement Rounds
    all_results['rounds'] = analyze_agreement_rounds(df)
    
    # 4. Move Type Analysis
    all_results['moves'] = analyze_move_types(df)
    
    # 5. Arousal-Valence Correlations
    all_results['correlations'] = analyze_arousal_valence_correlation(df)
    
    # Create summary table
    summary_df = create_summary_table(all_results)
    
    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    sig_count = summary_df['Significant'].sum()
    total_count = len(summary_df)
    
    print(f"\n  Significant effects: {sig_count}/{total_count}")
    print(f"  Output saved to: {OUTPUT_DIR}")
    
    return all_results, summary_df


if __name__ == "__main__":
    results, summary = main()
