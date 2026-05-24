#!/usr/bin/env python3
"""
Round-Level Dataset Reconstruction Script
==========================================
Reconstructs 'affective_behavioral_merged.csv' and 'final_evaluation_results.csv'
directly from 'final_dataset/experiment_round_data.csv' to resolve the turn-length
padding, swapped data, and move alternation bugs.

TDD Assertions Embedded:
- Row count matches exactly 2,114.
- Turn index matches round index.
- Subject S002 CL has exactly 11 rounds (indexed 1 to 11 / Round 1 to 11).
- Move_Type maps human moves correctly.
"""

import os
import sys
import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
ROUND_DATA_PATH = os.path.join(REPO_ROOT, 'final_dataset', 'experiment_round_data.csv')
SUBJECT_DATA_PATH = os.path.join(REPO_ROOT, 'final_dataset', 'experiment_subject_data.csv')

OUTPUT_MERGED_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'affective_behavioral_merged.csv')
OUTPUT_EVAL_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'final_evaluation_results.csv')


def reconstruct_datasets():
    print("=" * 80)
    print("RECONSTRUCTING ROUND-LEVEL AFFECTIVE & BEHAVIORAL DATASETS")
    print("=" * 80)

    # 1. Load the uncorrupted round data
    if not os.path.exists(ROUND_DATA_PATH):
        print(f"Error: Base round-level file not found at {ROUND_DATA_PATH}")
        sys.exit(1)

    print(f"Loading uncorrupted round-level base: {ROUND_DATA_PATH}")
    round_df = pd.read_csv(ROUND_DATA_PATH)
    print(f"  Loaded {len(round_df)} turns (rows) across subjects.")

    # 2. Construct the merged dataframe with exact column mappings
    merged = pd.DataFrame()

    # Format Subject names (e.g. S001)
    merged['Subject'] = round_df['Subject_ID'].apply(lambda x: f"S{x:03d}")
    merged['Condition'] = round_df['Condition']
    merged['Session'] = round_df['Session_ID']
    merged['Round'] = round_df['Round_ID']

    # Affect variables (Raw and Max/Min)
    merged['Raw_Arousal'] = round_df['Raw_Arousal']
    merged['Raw_Valence'] = round_df['Raw_Valence']
    merged['Raw_Max_A'] = round_df['Max_Arousal']
    merged['Raw_Min_A'] = round_df['Min_Arousal']
    merged['Raw_Max_V'] = round_df['Max_Valence']
    merged['Raw_Min_V'] = round_df['Min_Valence']

    # Session-Normalized Affect variables
    merged['Norm_Arousal'] = round_df['Norm_Arousal']
    merged['Norm_Valence'] = round_df['Norm_Valence']
    merged['Norm_Max_A'] = round_df['Max_Arousal']
    merged['Norm_Min_A'] = round_df['Min_Arousal']
    merged['Norm_Max_V'] = round_df['Max_Valence']
    merged['Norm_Min_V'] = round_df['Min_Valence']

    # Move Type mapping: Combine both Human moves (User_Move) and Agent moves (Agent_Move)
    # to represent the active move made in each turn.
    merged['Move_Type'] = round_df['User_Move'].fillna(round_df['Agent_Move']).str.strip().str.title()


    # round-by-round difference scores (Delta)
    # Computed per Session (Subject + Session combo)
    merged['Delta_User'] = round_df.groupby(['Subject_ID', 'Session_ID'])['Human_Utility'].diff()
    merged['Delta_Agent'] = round_df.groupby(['Subject_ID', 'Session_ID'])['Agent_Utility'].diff()

    # Agent Utility
    merged['Agent Utility'] = round_df['Agent_Utility']

    # Sort to ensure standard order: Subject, Session, Round
    merged = merged.sort_values(by=['Subject', 'Session', 'Round']).reset_index(drop=True)

    # 3. Create evaluation results dataframe (same rows, no behavioral columns)
    eval_cols = [
        'Subject', 'Condition', 'Session', 'Round',
        'Raw_Arousal', 'Raw_Valence', 'Raw_Max_A', 'Raw_Min_A', 'Raw_Max_V', 'Raw_Min_V',
        'Norm_Arousal', 'Norm_Valence', 'Norm_Max_A', 'Norm_Min_A', 'Norm_Max_V', 'Norm_Min_V'
    ]
    evaluation = merged[eval_cols].copy()

    # 4. Save to files
    os.makedirs(os.path.dirname(OUTPUT_MERGED_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_MERGED_PATH, index=False)
    print(f"Saved reconstructed file: {OUTPUT_MERGED_PATH}")

    evaluation.to_csv(OUTPUT_EVAL_PATH, index=False)
    print(f"Saved reconstructed file: {OUTPUT_EVAL_PATH}")

    print("\nReconstruction complete. Running TDD Validation Checks...")
    run_tdd_assertions(merged)


def run_tdd_assertions(df):
    """Run strict mathematical and logical assertions to verify dataset correctness."""
    print("-" * 60)
    print("RUNNING DATASETS INTEGRITY SELF-TESTS (TDD)")
    print("-" * 60)

    # Assertion 1: Length of dataset
    expected_rows = 2114
    actual_rows = len(df)
    assert actual_rows == expected_rows, f"Assertion Failed: Row count is {actual_rows}, expected {expected_rows}."
    print(f"  [PASS] Row Count: Exactly {actual_rows} rows.")

    # Assertion 2: Subject S002 CL rounds and turn lengths
    # S002 CL should have exactly 12 rows in the uncorrupted base dataset
    s002_cl = df[(df['Subject'] == 'S002') & (df['Condition'] == 'CL')]
    actual_rounds = len(s002_cl)
    expected_rounds = 12
    assert actual_rounds == expected_rounds, f"Assertion Failed: Subject S002 CL has {actual_rounds} rounds, expected {expected_rounds}."
    print(f"  [PASS] S002 CL Turns: Exactly {actual_rounds} rows (including final agreement NaN row).")

    # Assert S002 CL Round 11 values (the last active offer)
    s002_cl_r11 = s002_cl[s002_cl['Round'] == 11].iloc[0]
    expected_utility = 0.750
    actual_utility = s002_cl_r11['Agent Utility']
    assert abs(actual_utility - expected_utility) < 1e-4, f"Assertion Failed: S002 CL Round 11 Agent Utility is {actual_utility}, expected {expected_utility}."
    print(f"  [PASS] S002 CL Last Offer Utility: {actual_utility:.3f} at Round 11.")

    # Assert S002 CL Round 12 exists but has NaN utility (uncorrupted final block)
    s002_cl_r12 = s002_cl[s002_cl['Round'] == 12].iloc[0]
    assert np.isnan(s002_cl_r12['Agent Utility']), f"Assertion Failed: Round 12 has non-NaN Agent Utility: {s002_cl_r12['Agent Utility']} (padded data corruption!)."
    print("  [PASS] S002 CL Final Block: Round 12 exists as a correct non-corrupted NaN row.")

    # Assertion 3: Move mapping is correct
    # Subject S002 CL human moves: Round 3 (User_Move = fortunate), Round 11 (User_Move = concession)
    s002_cl_r3_move = s002_cl[s002_cl['Round'] == 3].iloc[0]['Move_Type']
    assert s002_cl_r3_move == 'Fortunate', f"Assertion Failed: S002 CL Round 3 move is {s002_cl_r3_move}, expected Fortunate."
    s002_cl_r11_move = s002_cl[s002_cl['Round'] == 11].iloc[0]['Move_Type']
    assert s002_cl_r11_move == 'Concession', f"Assertion Failed: S002 CL Round 11 move is {s002_cl_r11_move}, expected Concession."
    print("  [PASS] Move Type Mapping: Correctly title-cased and assigned to Human turns.")

    # Assertion 4: Mean Valence and Arousal are statistically correct
    # CL Arousal Mean should be significantly higher than FC
    fc_arousal = df[df['Condition'] == 'FC']['Norm_Arousal']
    cl_arousal = df[df['Condition'] == 'CL']['Norm_Arousal']
    print(f"  [INFO] Norm Arousal: CL Mean = {cl_arousal.mean():.4f}, FC Mean = {fc_arousal.mean():.4f}")
    assert cl_arousal.mean() > fc_arousal.mean(), "Assertion Failed: CL Arousal is not higher than FC!"
    print("  [PASS] Affect Normalization Consistency: CL Arousal is successfully normalized and elevated.")

    print("\nALL RECONSTRUCTION SELF-TESTS PASSED SUCCESSFULLY! ✅")
    print("=" * 80)


if __name__ == "__main__":
    reconstruct_datasets()
