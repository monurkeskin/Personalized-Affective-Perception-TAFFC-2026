#!/usr/bin/env python3
"""
Calculate Nash Distance for each negotiation session.
Nash distance = Euclidean distance from agreed bid to Nash bargaining point.
"""

import pandas as pd
import numpy as np
import os
from itertools import product

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'experiment_subject_data.csv')
ROUND_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'experiment_round_data.csv')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'experiment_subject_data.csv')

# ============================================================================
# UTILITY DEFINITIONS FROM XMLs
# ============================================================================

# Human utility (from Holiday_A/Human.xml)
HUMAN_WEIGHTS = {
    'Accommodation': 0.4,
    'Season': 0.3,
    'Destination': 0.2,
    'Events': 0.1
}

HUMAN_EVALS = {
    'Accommodation': {'House': 1.0, 'Hotel': 0.75, 'Caravan': 0.5, 'Boat': 0.25},
    'Season': {'Spring': 1.0, 'Fall': 0.75, 'Winter': 0.5, 'Summer': 0.25},
    'Destination': {'Barcelona': 1.0, 'London': 0.75, 'Boston': 0.5, 'Rome': 0.25},
    'Events': {'Show': 1.0, 'Shopping': 0.75, 'Sports': 0.5, 'Museum': 0.25}
}

# Agent utility (from Holiday_A/Agent.xml)
AGENT_WEIGHTS = {
    'Accommodation': 0.3,
    'Season': 0.4,
    'Destination': 0.1,
    'Events': 0.2
}

AGENT_EVALS = {
    'Accommodation': {'House': 0.5, 'Hotel': 0.25, 'Caravan': 1.0, 'Boat': 0.75},
    'Season': {'Spring': 0.5, 'Fall': 0.25, 'Winter': 1.0, 'Summer': 0.75},
    'Destination': {'Barcelona': 0.5, 'London': 0.25, 'Boston': 1.0, 'Rome': 0.75},
    'Events': {'Show': 0.5, 'Shopping': 0.25, 'Sports': 1.0, 'Museum': 0.75}
}

# Issue options
ISSUES = {
    'Accommodation': ['House', 'Hotel', 'Caravan', 'Boat'],
    'Season': ['Spring', 'Fall', 'Winter', 'Summer'],
    'Destination': ['Barcelona', 'London', 'Boston', 'Rome'],
    'Events': ['Show', 'Shopping', 'Sports', 'Museum']
}


def calculate_human_utility(bid):
    """Calculate human utility for a bid (dict of issue -> value)"""
    utility = 0.0
    for issue, value in bid.items():
        weight = HUMAN_WEIGHTS[issue]
        eval_val = HUMAN_EVALS[issue][value]
        utility += weight * eval_val
    return utility


def calculate_agent_utility(bid):
    """Calculate agent utility for a bid (dict of issue -> value)"""
    utility = 0.0
    for issue, value in bid.items():
        weight = AGENT_WEIGHTS[issue]
        eval_val = AGENT_EVALS[issue][value]
        utility += weight * eval_val
    return utility


def generate_all_bids():
    """Generate all possible bids in the domain"""
    issue_names = list(ISSUES.keys())
    all_options = [ISSUES[issue] for issue in issue_names]
    
    bids = []
    for combo in product(*all_options):
        bid = dict(zip(issue_names, combo))
        bids.append(bid)
    
    return bids


def find_nash_point():
    """
    Find the Nash bargaining point.
    Nash solution maximizes the product of utilities (assuming disagreement = 0,0).
    """
    all_bids = generate_all_bids()
    
    best_product = -1
    nash_point = None
    nash_bid = None
    
    for bid in all_bids:
        human_u = calculate_human_utility(bid)
        agent_u = calculate_agent_utility(bid)
        
        # Nash product (assuming disagreement point at origin)
        nash_product = human_u * agent_u
        
        if nash_product > best_product:
            best_product = nash_product
            nash_point = (agent_u, human_u)  # (Agent, Human)
            nash_bid = bid
    
    return nash_point, nash_bid, best_product


def calculate_nash_distance(agent_utility, human_utility, nash_point):
    """
    Calculate Euclidean distance from agreed point to Nash point.
    """
    nash_agent, nash_human = nash_point
    distance = np.sqrt((agent_utility - nash_agent)**2 + (human_utility - nash_human)**2)
    return distance


def main():
    print("=" * 60)
    print("NASH DISTANCE CALCULATION")
    print("=" * 60)
    
    # Step 1: Find Nash bargaining point
    print("\nStep 1: Finding Nash bargaining point...")
    nash_point, nash_bid, nash_product = find_nash_point()
    print(f"  Nash Bid: {nash_bid}")
    print(f"  Nash Point: Agent U = {nash_point[0]:.4f}, Human U = {nash_point[1]:.4f}")
    print(f"  Nash Product: {nash_product:.4f}")
    
    # Verify bid space size
    all_bids = generate_all_bids()
    print(f"  Total bids in domain: {len(all_bids)}")
    
    # Step 2: Load subject data
    print("\nStep 2: Loading subject data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows")
    
    # Step 3: Calculate Nash distance for each session
    print("\nStep 3: Calculating Nash distance for each session...")
    
    nash_distances = []
    for idx, row in df.iterrows():
        agent_u = row['Agent_Utility']
        human_u = row['Human_Utility']
        
        if pd.isna(agent_u) or pd.isna(human_u):
            nash_distances.append(np.nan)
        else:
            dist = calculate_nash_distance(agent_u, human_u, nash_point)
            nash_distances.append(dist)
    
    df['Nash_Distance'] = nash_distances
    
    # Step 4: Summary statistics
    print("\nStep 4: Summary statistics...")
    fc = df[df['Condition'] == 'FC']['Nash_Distance'].dropna()
    cl = df[df['Condition'] == 'CL']['Nash_Distance'].dropna()
    
    print(f"  FC: {fc.mean():.4f} ± {fc.std():.4f} (N={len(fc)})")
    print(f"  CL: {cl.mean():.4f} ± {cl.std():.4f} (N={len(cl)})")
    
    from scipy import stats
    t, p = stats.ttest_rel(
        df[df['Condition'] == 'CL'].set_index('Subject_ID')['Nash_Distance'],
        df[df['Condition'] == 'FC'].set_index('Subject_ID')['Nash_Distance']
    )
    print(f"  Paired t-test: t = {t:.3f}, p = {p:.4f}")
    
    # Step 5: Save updated data
    print("\nStep 5: Saving updated data...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Saved to: {OUTPUT_PATH}")
    
    # Show column order
    print(f"\n  Columns: {list(df.columns)}")
    
    print("\n" + "=" * 60)
    print("NASH DISTANCE CALCULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
