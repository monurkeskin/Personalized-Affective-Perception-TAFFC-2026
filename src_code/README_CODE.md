# Solver-CL: Source Code

**TL;DR:** This directory implements the Solver agent and its CLIFER components for the accompanying study. The guide below maps paper components to their source directories.

**Keywords:** Solver, CLIFER, automated negotiation, continual learning, affective computing

[Study overview](../README.md) · [Architecture](ARCHITECTURE.md) · [Runtime setup](../docs/runtime.md)

## Architecture Mapping

| Paper Component | Source Directory | Description |
|---|---|---|
| **Negotiator Core** | `solver_cl_agent/HATN/` | Handles the negotiation protocol, utility space, offer history, and speech-to-offer mapping. Implements the Alternating Offers Protocol. |
| **Solver Agent** | `solver_cl_agent/Agent/Solver_Agent/` | Implements the decision-making logic: Time-Based Strategy (Eq. 2), Behavior-Based Strategy (Eq. 3), and Differential Affective Feedback (P_E). |
| **Continual Learning Module** | `solver_cl_agent/FaceChannel/CLModel/` | CLIFER implementation with Episodic/Semantic Memory (GDM-E/GDM-S) and a separate CAAE generator for personalized affect perception. |
| **Strategy Components** | `solver_cl_agent/Agent/Solver_Agent/*.py` | `time_based.py` (Bezier curve), `behavior_based.py` (Tit-for-Tat), emotion controller, and sensitivity calculator. |

See [Architecture Documentation](ARCHITECTURE.md) for the paper-to-code mapping,
active CLIFER imports, and key functions.

## Dependencies

- **Python dependencies**: the [negotiator requirements](solver_cl_agent/requirements.txt) and [perception requirements](solver_cl_agent/FaceChannel/requirements.txt) retain the original runtime declarations.
- **NAOqi SDK**: the matching SoftBank Robotics SDK and legacy `naoqi.ALProxy` bridge for robot control.
- **TensorFlow**: the graph/session runtime compatible with the CAAE checkpoint used for generative replay.
- **FaceChannel**: the pre-trained dimensional model and detector assets used to encode facial affect.
- **OpenCV**: facial frame capture and processing.

The robot, perception, and modern software-check environments are separate.
[Runtime setup](../docs/runtime.md) explains the headless Solver launcher and its
input contracts; [tested environments](../docs/tested-environments.md) lists the
SDK/model versions, hashes, and checks completed for each profile. The historical
requirements files are source constraints; use the documented environment for
the selected task instead of combining them into one installation.

## Reproducibility Note

The scripts provide the offer-generation, sensitivity-classification, and
personalized affect-perception components for comparison with the paper. Running
the complete negotiation system also requires the matching lab interaction
adapters, model assets, and study configuration. The original `main.py` currently
selects `Hybrid`; `SolverAgent` contains the differential-feedback implementation
mapped in the architecture guide.

The public-data verifier and headless software checks can run independently of
the robot and perception stack. Their declared scope does not establish live
camera capture, online learning, model publication, or physical robot operation.
[Maintenance guidance](../docs/maintenance.md) explains source preservation and
runtime compatibility, including the distinction between the original NAOqi
bridge and modern `qi` bindings.
