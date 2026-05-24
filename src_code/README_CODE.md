# Solver-CL Agent: Source Code

This directory contains the core source code for the **Personalized Affective Perception via Continual Learning** negotiation framework. The implementation integrates the CLIFER module with the Solver Agent for real-time affective adaptation during human-agent negotiation.

## Architecture Mapping

| Paper Component | Source Directory | Description |
|---|---|---|
| **Negotiator Core** | `solver_cl_agent/HATN/` | Handles the negotiation protocol, utility space, offer history, and speech-to-offer mapping. Implements the Alternating Offers Protocol. |
| **Solver Agent** | `solver_cl_agent/Agent/Solver_Agent/` | Implements the decision-making logic: Time-Based Strategy (Eq. 2), Behavior-Based Strategy (Eq. 3), and Differential Affective Feedback (P_E). |
| **Continual Learning Module** | `solver_cl_agent/FaceChannel/CLModel/` | CLIFER implementation with Episodic/Semantic Memory (GDM-E/GDM-S) and CAAE generator for personalized affect perception. |
| **Strategy Components** | `solver_cl_agent/Agent/Solver_Agent/*.py` | `time_based.py` (Bezier curve), `behavior_based.py` (Tit-for-Tat), emotion controller, and sensitivity calculator. |

## Dependencies

- `requirements.txt`: Python dependencies
- **Naoqi SDK**: SoftBank Robotics SDK for robot control (NAO/Pepper)
- **TensorFlow**: CAAE model for generative replay
- **OpenCV**: Facial frame capture and processing

## Reproducibility Note

The logic for Offer Generation, Sensitivity Classification, and Personalized Affect Perception is fully contained in these Python scripts and can be audited against the equations in the paper.

See `ARCHITECTURE.md` for detailed paper-to-code mapping.

