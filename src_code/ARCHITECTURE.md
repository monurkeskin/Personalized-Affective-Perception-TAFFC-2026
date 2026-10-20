# Solver-CL Agent: Architecture Documentation

This document locates the historical Solver/CLIFER components. The original numerical source is retained for provenance. Running the robot/perception system additionally requires the matching lab adapters, model assets and separate runtime environments; this map does not certify a complete runnable reproduction. See [maintenance guidance](../docs/maintenance.md).

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MAIN ENTRY POINT                                  │
│                              main.py                                         │
│                     HAT (Human Agent Tool) initialization                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEGOTIATOR CORE                                    │
│                     HATN/human_agent_tool.py                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │ utilitySpace   │  │  negoHistory   │  │   negoTime     │                │
│  │  Preferences   │  │  Offer Logs    │  │   Deadline     │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                                │
          ▼                                                ▼
┌──────────────────────────────┐        ┌──────────────────────────────────────┐
│      FACIAL EXPRESSION       │        │           SOLVER AGENT               │
│      ANALYSIS MODULE         │        │   Agent/Solver_Agent/Solver_Agent.py │
│  FaceChannel/CLModel/        │        │  ┌────────────────────────────────┐  │
│  ┌────────────────────────┐  │        │  │   TimeBasedStrategy            │  │
│  │   cl_model_c3.py       │  │        │  │   (time_based.py)              │  │
│  │   - train_GDM()        │  │        │  └────────────────────────────────┘  │
│  │   - get_arousal_valence│  │        │  ┌────────────────────────────────┐  │
│  └────────────────────────┘  │        │  │   BehaviorBasedStrategy        │  │
│  ┌────────────────────────┐  │        │  │   (behavior_based.py)          │  │
│  │   GDM-E (Episodic)     │──┼───────►│  └────────────────────────────────┘  │
│  │   GDM-S (Semantic)     │  │  V,A   │  ┌────────────────────────────────┐  │
│  └────────────────────────┘  │        │  │   EmotionController            │  │
│  ┌────────────────────────┐  │        │  │   (Agent_Emotion/)             │  │
│  │   CAAE Generator       │  │        │  └────────────────────────────────┘  │
│  │   (checkpoint/)        │  │        │  ┌────────────────────────────────┐  │
│  └────────────────────────┘  │        │  │   UncertaintyModule            │  │
└──────────────────────────────┘        │  │   (uncertainty_module.py)      │  │
                                        │  └────────────────────────────────┘  │
                                        └──────────────────────────────────────┘
```

---

## Paper-to-Code Mapping

### Section 4.1: Generative Replay (CLIFER Module)

| Paper Concept | Code Location | Function |
|---------------|---------------|----------|
| Episodic Memory (GDM-E) | `CLModel/GDM_Imagine_Dimensional/episodic_gwr.py` | `EpisodicGWR` class |
| Semantic Memory (GDM-S) | Same file, different params | Lower max_nodes, higher age |
| CAAE Generator | `CLModel/checkpoint/` | Pre-trained TensorFlow model |
| Imagined Samples | `cl_model_c3.py:generate_images()` | Lines 160-206 |
| GDM training with real/imagined features | `cl_model_c3_faster.py:train_GDM()` | Historical GWR update path; no separate weighted-loss optimizer is asserted |

### Section 4.2: Differential Affective Feedback (P_E)

| Paper Concept | Code Location | Line Numbers |
|---------------|---------------|---------------|
| P_sign (Directional) | `Solver_Agent.py` | L476-477 |
| P_rot (Rotation) | `Solver_Agent.py` | L479-486 |
| P_rad (Radial) | `Solver_Agent.py` | L488-489 |
| Normalization | `Solver_Agent.py` | L491-493 |
| Final P_E | `Solver_Agent.py` | L495-499 |
| Emotion Mappings | `Solver_Agent.py:emotion_evaluations` | L287-295 |
| Normalized affect input | `Solver_Agent.py:receive_offer()` | Requires the externally supplied `normalized_predictions` contract |

### Section 4.3: Asynchronous Architecture (Algorithm 1)

| Algorithm Step | Code Location | Description |
|----------------|---------------|-------------|
| Thread 1: Inference | `HATN/human_agent_tool.py:do_normal_nego()` | Main negotiation loop |
| CaptureFrame | `FaceChannel/SessionManager/` | Camera capture |
| Inference(θ_curr) | `cl_model_c3.py:get_arousal_valence()` | Lines 596-662 |
| Thread 2: Learning | `cl_model_c3.py:train_GDM()` | Background CL update |
| GenerativeReplay | `cl_model_c3.py:apply_network_to_images_of_dir()` | Lines 444-475 |

---

## Key Functions Reference

### `Solver_Agent.receive_offer()`
Main decision function. Processes human offers and generates counter-offers using:
1. Time-based target utility (Bezier curve)
2. Behavior-based adjustments (opponent modeling)
3. Emotion feedback modulation

### `cl_model_c3.get_arousal_valence()`
End-to-end CLIFER pipeline:
1. Generate imagined faces using CAAE
2. Encode real + imagined faces via FaceChannel
3. Train/update GDM-E and GDM-S
4. Annotate current frames with personalized arousal/valence

### `cl_model_c3.train_GDM()`
Implements the dual-memory continual learning:
- GDM-E: Fast adaptation (max_nodes=len(data), age=600)
- GDM-S: Slow consolidation (max_nodes=len(data)//2, age=1200)

---

## Dependencies

```
Core:           HATN (negotiation protocol)
ML Framework:   Historical TF graph/session environment (requires original model compatibility)
Affect Model:   FaceChannel (pre-trained dimensional model)
CL Memory:      GWR (Growing When Required networks)
```

## Runtime selection and preservation

`main.py` currently selects Hybrid; the differential-feedback implementation lives in `SolverAgent`. Their integration must be validated with the original adapters before substituting one for the other. The active session/training code imports `cl_model_c3_faster.py`. The analysis verifier runs independently of this runtime and does not certify camera/robot operation, normalization, condition isolation or model publication. Maintenance changes do not alter the original Solver/GWR/condition implementation.
