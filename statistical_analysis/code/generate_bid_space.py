#!/usr/bin/env python3
"""
Generate Bid Space visualization showing:
- All possible bids as blue dots
- Pareto frontier as black line with markers
- Nash bargaining point as red triangle
"""

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from itertools import product
import os

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'assets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Domain XML paths (anonymized utility profiles)
DOMAIN_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'domain')
AGENT_XML = os.path.join(DOMAIN_DIR, "agent_utility.xml")
HUMAN_XML = os.path.join(DOMAIN_DIR, "participant_utility.xml")

def parse_utility_xml(xml_path):
    """Parse utility XML file to extract issue weights and value utilities.
    
    XML Structure:
    <negotiation_domain>
      <utility_space>
        <weight index="1" value="0.3"/>
        <issue index="1" name="Accommodation">
          <item index="1" value="House" evaluation="0.5"/>
          ...
        </issue>
        ...
      </utility_space>
    </negotiation_domain>
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    utility_space = root.find('.//utility_space')
    
    # Collect weights by index
    weights = {}
    for weight_elem in utility_space.findall('weight'):
        idx = int(weight_elem.get('index'))
        weights[idx] = float(weight_elem.get('value'))
    
    # Parse issues
    issues = {}
    for issue in utility_space.findall('issue'):
        issue_name = issue.get('name')
        issue_index = int(issue.get('index'))
        
        # Get weight for this issue
        weight = weights.get(issue_index, 0.0)
        
        # Get value utilities
        values = {}
        for item in issue.findall('item'):
            value_name = item.get('value')
            evaluation = float(item.get('evaluation', 0.0))
            values[value_name] = evaluation
        
        issues[issue_name] = {
            'index': issue_index,
            'weight': weight,
            'values': values
        }
    
    return issues

def calculate_utility(bid, issues):
    """Calculate utility for a bid given issue structure."""
    total = 0.0
    for issue_name, value in bid.items():
        if issue_name in issues:
            issue = issues[issue_name]
            if value in issue['values']:
                total += issue['weight'] * issue['values'][value]
    return total

def generate_all_bids(issues):
    """Generate all possible bids from issue structure."""
    issue_names = list(issues.keys())
    value_lists = [list(issues[name]['values'].keys()) for name in issue_names]
    
    all_bids = []
    for combo in product(*value_lists):
        bid = dict(zip(issue_names, combo))
        all_bids.append(bid)
    
    return all_bids

def find_pareto_frontier(points):
    """Find Pareto-optimal points (maximizing both utilities)."""
    pareto_points = []
    
    for i, (x1, y1) in enumerate(points):
        dominated = False
        for j, (x2, y2) in enumerate(points):
            if i != j:
                # Check if point j dominates point i
                if x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                    dominated = True
                    break
        if not dominated:
            pareto_points.append((x1, y1))
    
    # Sort by user utility for plotting
    pareto_points.sort(key=lambda p: p[0])
    return pareto_points

def find_nash_point(points):
    """Find Nash bargaining point (maximizes product of utilities)."""
    best_product = -1
    nash_point = None
    
    for x, y in points:
        product = x * y  # Assuming disagreement point at (0, 0)
        if product > best_product:
            best_product = product
            nash_point = (x, y)
    
    return nash_point

def generate_bid_space_figure():
    """Generate the bid space visualization."""
    print("Parsing utility XMLs...")
    agent_issues = parse_utility_xml(AGENT_XML)
    human_issues = parse_utility_xml(HUMAN_XML)
    
    print(f"Agent issues: {list(agent_issues.keys())}")
    print(f"Human issues: {list(human_issues.keys())}")
    
    print("Generating all possible bids...")
    all_bids = generate_all_bids(agent_issues)
    print(f"Total bids: {len(all_bids)}")
    
    # Calculate utilities for all bids
    points = []
    for bid in all_bids:
        agent_u = calculate_utility(bid, agent_issues)
        human_u = calculate_utility(bid, human_issues)
        points.append((human_u, agent_u))  # (User, Agent)
    
    # Find Pareto frontier
    pareto_points = find_pareto_frontier(points)
    print(f"Pareto-optimal bids: {len(pareto_points)}")
    
    # Find Nash point
    nash_point = find_nash_point(points)
    print(f"Nash point: User={nash_point[0]:.3f}, Agent={nash_point[1]:.3f}")
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Colors matching paper
    WONG_BLUE = '#5B9BD5'
    WONG_ORANGE = '#ED7D31'
    
    # Plot all bids (Wong Blue)
    user_utils = [p[0] for p in points]
    agent_utils = [p[1] for p in points]
    plt.scatter(user_utils, agent_utils, c=WONG_BLUE, s=40, alpha=0.7, label='Point', zorder=2)
    
    # Plot Pareto frontier (Black)
    pareto_x = [p[0] for p in pareto_points]
    pareto_y = [p[1] for p in pareto_points]
    plt.plot(pareto_x, pareto_y, 'k-', linewidth=2, zorder=3)
    plt.scatter(pareto_x, pareto_y, c='black', s=120, marker='*', label='Pareto', zorder=4)
    
    # Plot Nash point (Orange)
    plt.scatter([nash_point[0]], [nash_point[1]], c=WONG_ORANGE, s=400, marker='^', 
                label='Nash', zorder=5, edgecolors='black', linewidths=1)
    
    # Formatting
    plt.xlabel('Participant Utility', fontsize=14, fontweight='bold')
    plt.ylabel('Agent Utility', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=12)
    
    # Equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'BidSpace.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: BidSpace.png")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING BID SPACE VISUALIZATION")
    print("=" * 60)
    generate_bid_space_figure()
    print("=" * 60)
