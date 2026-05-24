#!/usr/bin/env python3
"""
Raw Data Verification Script
Computes all paper statistics directly from session logs to verify accuracy.
Uses the repository's raw_logs/ directory and experiment_subject_data.csv for conditions.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# Paths - relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
RAW_LOGS_BASE = os.path.join(REPO_ROOT, 'raw_logs')
SUBJECT_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'experiment_subject_data.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis')

def load_condition_map():
    """Load subject-session-condition mapping from experiment_subject_data.csv"""
    df = pd.read_csv(SUBJECT_DATA_PATH)
    condition_map = {}
    for _, row in df.iterrows():
        # Convert integer Subject_ID (1) to SXXX format (S001) for file matching
        subject_id = f"S{int(row['Subject_ID']):03d}"
        key = (subject_id, row['Session_ID'])
        condition_map[key] = row['Condition']
    return condition_map

def load_all_sessions():
    """Load all session logs from raw_logs directory."""
    condition_map = load_condition_map()
    all_data = []
    loaded_count = 0
    
    for session_num in [1, 2]:
        session_dir = os.path.join(RAW_LOGS_BASE, f'Session_{session_num}')
        if not os.path.exists(session_dir):
            print(f"  Warning: {session_dir} not found")
            continue
        
        for file in os.listdir(session_dir):
            if file.endswith('.xlsx') and file.startswith('negotiation_logs_'):
                # Extract subject name: negotiation_logs_S001_Session_1.xlsx -> S001
                parts = file.replace('.xlsx', '').split('_')
                if len(parts) >= 3:
                    subject = parts[2]  # S001
                else:
                    continue
                
                # Lookup condition from subject data
                condition = condition_map.get((subject, session_num), None)
                if condition is None:
                    continue
                
                # Load file
                try:
                    filepath = os.path.join(session_dir, file)
                    df = pd.read_excel(filepath)
                    df['Subject'] = subject
                    df['Session'] = session_num
                    df['Condition'] = condition
                    all_data.append(df)
                    loaded_count += 1
                except Exception as e:
                    print(f"  Error loading {file}: {e}")
    
    print(f"  Loaded {loaded_count} session files")
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined

def analyze_negotiation_outcomes(df):
    """Analyze Agent Utility by condition."""
    print("\n" + "="*60)
    print("NEGOTIATION OUTCOMES (from raw logs)")
    print("="*60)
    
    # Get final round per session (last offer = agreement)
    final_outcomes = df.groupby(['Subject', 'Condition', 'Session']).agg({
        'Agent Utility': 'last',
        'Human Utility': 'last'
    }).reset_index()
    
    # Subject-level means
    subj_means = final_outcomes.groupby(['Subject', 'Condition']).agg({
        'Agent Utility': 'mean',
        'Human Utility': 'mean'
    }).reset_index()
    
    print("\n--- Final Agent Utility ---")
    for cond in ['FC', 'CL']:
        vals = subj_means[subj_means['Condition'] == cond]['Agent Utility'].dropna()
        print(f"  {cond}: {vals.mean():.3f} ± {vals.std():.3f} (N={len(vals)})")
    
    # Paired t-test
    fc_subj = subj_means[subj_means['Condition'] == 'FC'].set_index('Subject')
    cl_subj = subj_means[subj_means['Condition'] == 'CL'].set_index('Subject')
    common = fc_subj.index.intersection(cl_subj.index)
    
    fc = fc_subj.loc[common, 'Agent Utility'].values
    cl = cl_subj.loc[common, 'Agent Utility'].values
    
    if len(common) > 0:
        t, p = stats.ttest_rel(cl, fc)
        d = (cl.mean() - fc.mean()) / np.std(cl - fc) if np.std(cl - fc) > 0 else 0
        print(f"  Paired t-test (N={len(common)}): t = {t:.3f}, p = {p:.4f}, d = {d:.3f}")
    
    return final_outcomes

def analyze_move_types(df):
    """Analyze move type proportions by condition."""
    print("\n" + "="*60)
    print("BEHAVIORAL MOVES (from raw logs)")
    print("="*60)
    
    # Check column names
    move_col = 'Move' if 'Move' in df.columns else 'Agent_Move' if 'Agent_Move' in df.columns else None
    if move_col is None:
        print("  Warning: Move column not found in raw logs")
        return {}
    
    # Filter human moves only (agent rows don't have moves from human)
    df_human = df[df['Bidder'] == 'Human'].copy() if 'Bidder' in df.columns else df.copy()
    df_human['Move'] = df_human[move_col].astype(str).str.lower().str.strip()
    
    # Get move proportions per subject-condition
    move_counts = df_human.groupby(['Subject', 'Condition', 'Move']).size().reset_index(name='Count')
    totals = df_human.groupby(['Subject', 'Condition']).size().reset_index(name='Total')
    
    move_props = move_counts.merge(totals, on=['Subject', 'Condition'])
    move_props['Proportion'] = move_props['Count'] / move_props['Total']
    
    print("\n--- Move Proportions ---")
    
    moves = ['concession', 'fortunate', 'selfish', 'nice', 'unfortunate']
    results = {}
    
    for move in moves:
        move_data = move_props[move_props['Move'] == move]
        pivot = move_data.pivot_table(index='Subject', columns='Condition', values='Proportion', fill_value=0)
        
        if 'FC' in pivot.columns and 'CL' in pivot.columns:
            fc = pivot['FC'].values
            cl = pivot['CL'].values
            
            t, p = stats.ttest_rel(cl, fc) if len(cl) == len(fc) and len(cl) > 0 else (0, 1)
            d = (cl.mean() - fc.mean()) / np.std(cl - fc) if np.std(cl - fc) > 0 else 0
            
            print(f"\n  {move.title()}:")
            print(f"    FC: {fc.mean()*100:.1f}% ± {fc.std()*100:.1f}%")
            print(f"    CL: {cl.mean()*100:.1f}% ± {cl.std()*100:.1f}%")
            print(f"    t = {t:.2f}, p = {p:.4f}, d = {d:.2f}")
            
            results[move] = {'fc': fc.mean(), 'cl': cl.mean(), 't': t, 'p': p, 'd': d}
    
    return results

def analyze_rounds_to_agreement(df):
    """Count rounds per session."""
    print("\n" + "="*60)
    print("ROUNDS TO AGREEMENT")
    print("="*60)
    
    # Count unique rounds per session (each round has 2 rows: Human + Agent)
    if 'Round' in df.columns:
        rounds = df.groupby(['Subject', 'Condition', 'Session'])['Round'].max().reset_index()
        rounds.columns = ['Subject', 'Condition', 'Session', 'Rounds']
    else:
        # Count rows / 2 (since each round has Human and Agent bid)
        rounds = df.groupby(['Subject', 'Condition', 'Session']).size().reset_index(name='RowCount')
        rounds['Rounds'] = rounds['RowCount'] // 2
    
    # Subject means
    subj_rounds = rounds.groupby(['Subject', 'Condition'])['Rounds'].mean().reset_index()
    
    print("\n--- Rounds per Session ---")
    for cond in ['FC', 'CL']:
        vals = subj_rounds[subj_rounds['Condition'] == cond]['Rounds'].values
        print(f"  {cond}: {vals.mean():.1f} ± {vals.std():.1f}")
    
    fc_subj = subj_rounds[subj_rounds['Condition'] == 'FC'].set_index('Subject')
    cl_subj = subj_rounds[subj_rounds['Condition'] == 'CL'].set_index('Subject')
    common = fc_subj.index.intersection(cl_subj.index)
    
    fc = fc_subj.loc[common, 'Rounds'].values
    cl = cl_subj.loc[common, 'Rounds'].values
    
    if len(common) > 0:
        t, p = stats.ttest_rel(cl, fc)
        d = (cl.mean() - fc.mean()) / np.std(cl - fc) if np.std(cl - fc) > 0 else 0
        print(f"  Paired t-test (N={len(common)}): t = {t:.3f}, p = {p:.4f}, d = {d:.3f}")

def main():
    print("="*60)
    print("RAW DATA VERIFICATION")
    print("Processing session logs from raw_logs/")
    print("="*60)
    
    # Load all session data
    print("\nLoading session logs...")
    df = load_all_sessions()
    
    if df is None or len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    print(f"\n  Total rows: {len(df)}")
    print(f"  Unique subjects: {df['Subject'].nunique()}")
    print(f"  Conditions: {df['Condition'].value_counts().to_dict()}")
    
    # Analysis 1: Negotiation Outcomes
    analyze_negotiation_outcomes(df)
    
    # Analysis 2: Move Types
    analyze_move_types(df)
    
    # Analysis 3: Rounds to Agreement
    analyze_rounds_to_agreement(df)
    
    print("\n" + "="*60)
    print("RAW DATA VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
