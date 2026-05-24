# coding=utf-8
"""
CLIFER Negotiation Experiment Main Entry Point

This script initializes and runs a human-agent negotiation session using the
HAT (Human Agent Tool) framework. It supports multiple configurations for
different experimental conditions.

Usage:
    python main.py

Configuration:
    Modify the HAT initialization parameters below to configure:
    - Participant information
    - Domain files (utility spaces)
    - Interaction modes (CLI, Robot, etc.)
    - Session parameters

For more details, see the README_CODE.md in this directory.
"""

import os
from pathlib import Path

from HATN.human_agent_tool import HAT


# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths - use relative paths for reproducibility
BASE_DIR = Path(__file__).parent.resolve()
DOMAIN_PATH = BASE_DIR / "HATN" / "Domain"
DEFAULT_LOG_PATH = BASE_DIR / "logs"

# Ensure log directory exists
DEFAULT_LOG_PATH.mkdir(exist_ok=True)


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

# Example: Domain A with CLI interaction (for testing/development)
tool = HAT(
    participant_name="",
    human_interaction_type="Human-CLI",
    agent_interaction_type="Agent-CLI",
    agent_type="Hybrid",
    agent_preference_file=str(DOMAIN_PATH / "Fruit" / "Fruits_A_Agent.xml"),
    human_preference_file=str(DOMAIN_PATH / "Fruit" / "Fruits_A_Human.xml"),
    domain_file=str(DOMAIN_PATH / "Fruit" / "Fruits.xml"),
    deadline=900,
    session_number=1,
    log_file_path=str(DEFAULT_LOG_PATH) + "/",
    domain="A"
)

# =============================================================================
# ALTERNATIVE CONFIGURATIONS (Uncomment as needed)
# =============================================================================

# --- Demo Setup ---
# tool = HAT(
#     participant_name="Demo-Participant",
#     human_interaction_type="Microphone",
#     agent_interaction_type="Robot-Mobile",
#     agent_type="Solver",
#     agent_preference_file=str(DOMAIN_PATH / "Demo_Fruit" / "Demo_Agent_Fruits.xml"),
#     human_preference_file=str(DOMAIN_PATH / "Demo_Fruit" / "Demo_Human_Fruits.xml"),
#     domain_file=str(DOMAIN_PATH / "Demo_Fruit" / "Demo_Fruits.xml"),
#     deadline=300,
#     session_number=1,
#     log_file_path=str(DEFAULT_LOG_PATH / "Demo_Logs") + "/",
#     domain="Demo"
# )

# --- Domain B (Robot Interaction) ---
# tool = HAT(
#     participant_name="",
#     human_interaction_type="Microphone",
#     agent_interaction_type="Robot-Mobile",
#     agent_type="Solver",
#     agent_preference_file=str(DOMAIN_PATH / "Fruit" / "Fruits_B_Agent.xml"),
#     human_preference_file=str(DOMAIN_PATH / "Fruit" / "Fruits_B_Human.xml"),
#     domain_file=str(DOMAIN_PATH / "Fruit" / "Fruits.xml"),
#     deadline=900,
#     session_number=2,
#     log_file_path=str(DEFAULT_LOG_PATH) + "/",
#     domain="B"
# )


# =============================================================================
# RUN NEGOTIATION
# =============================================================================

if __name__ == "__main__":
    print("Starting CLIFER Negotiation Session...")
    print(f"Log directory: {DEFAULT_LOG_PATH}")
    tool.negotiate()
