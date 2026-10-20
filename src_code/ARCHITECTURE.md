# Solver-CL: Architecture

This guide maps the [paper's methods](../README.md#4-proposed-method-implementation-details) to the Solver-CL source, from CLIFER's dual memories to differential feedback and the asynchronous execution paths.

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

The diagram shows the component relationships. The source retains both
`cl_model_c3.py` and `cl_model_c3_faster.py`; `Session.py` and
`CLTrainManager.py` import the faster variant. The references below follow those
active imports. CLIFER paths are relative to `solver_cl_agent/FaceChannel/`, and
Solver paths are relative to `solver_cl_agent/Agent/Solver_Agent/`.

## Paper-to-Code Mapping

### Section 4.1: Generative Replay (CLIFER Module)

| Paper Concept | Code Location | Function |
|---------------|---------------|----------|
| Episodic Memory (GDM-E) | `CLModel/GDM_Imagine_Dimensional/episodic_gwr.py` | `EpisodicGWR` class |
| Semantic Memory (GDM-S) | `CLModel/cl_model_c3_faster.py:train_GDM()` | A second `EpisodicGWR` instance with lower `max_nodes` and higher `age` |
| CAAE Generator | `CLModel/checkpoint/` and `CLModel/cl_model_c3_faster.py:generate_images()` | Pre-trained TensorFlow generator, separate from GDM-S |
| Imagined Samples | `CLModel/cl_model_c3_faster.py:apply_network_to_images_of_dir()` | Generates and encodes imagined faces |
| GDM Training with Real and Imagined Features | `CLModel/cl_model_c3_faster.py:get_arousal_valence()` and `train_GDM()` | Combines the feature arrays and updates the two GWR memories |

The replay path combines observed and imagined features before the GWR updates.
The conceptual `L_total` notation in the paper overview does not denote a
separate weighted-loss optimizer with a tunable λ in `train_GDM()`.

### Section 4.2: Differential Affective Feedback (P_E)

| Paper Concept | Code Location | Function |
|---------------|---------------|----------|
| P_sign (Directional) | `Solver_Agent.py:SolverAgent.receive_offer()` | Sign of `delta_v + alpha * delta_a`, with `alpha = 1.0` and zero mapped to +1 |
| P_rot (Rotation) | `Solver_Agent.py:SolverAgent.receive_offer()` | `atan2(delta_det, delta_dot)` |
| P_rad (Radial) | `Solver_Agent.py:SolverAgent.receive_offer()` | Absolute change in affect-vector magnitude |
| Component Normalization | `Solver_Agent.py:SolverAgent.receive_offer()` | Absolute rotation divided by π; radial change divided by √2 |
| Final P_E | `Solver_Agent.py:SolverAgent.receive_offer()` | Directional sign times the combined magnitude, clipped at 1.0 |
| Emotion Mappings | `Solver_Agent.py:SolverAgent.mood_evaluations` | Affect-to-emotion mapping used by the agent |
| Session-Normalized Affect Input | `Solver_Agent.py:SolverAgent.receive_offer()` | Consumes the externally supplied `normalized_predictions` channel |

The session-normalized input and the component normalization are separate steps.
The lab perception integration must supply separate raw and normalized affect values;
the [compatibility runtime](../docs/runtime.md#policy-and-perception-contract)
passes both channels to the Solver without fitting a replacement normalizer.

### Section 4.3: Asynchronous Architecture (Algorithm 1)

| Algorithm Step | Code Location | Description |
|----------------|---------------|-------------|
| Thread 1: Interaction | `solver_cl_agent/HATN/human_agent_tool.py:HAT.do_normal_nego()` | Main negotiation loop |
| CaptureFrame | `SessionManager/Capturing.py` and `SessionManager/Session.py` | Camera capture and round-level frame collection |
| Inference(θ_curr) | `SessionManager/Session.py:predict_cl()` and `CLModel/cl_model_c3_faster.py:predict()` | Applies the saved GWR memories to current frames |
| Thread 2: Learning | `SessionManager/CLTrainManager.py` and `CLModel/cl_model_c3_faster.py:get_arousal_valence()` | Queues background CL updates |
| GenerativeReplay | `CLModel/cl_model_c3_faster.py:apply_network_to_images_of_dir()` | Produces imagined samples for the training path |

---

## Key Functions Reference

### `SolverAgent.receive_offer()`

Main decision function in `Solver_Agent.py`. It processes human offers and
generates counter-offers using:

1. Time-based target utility (Bezier curve)
2. Behavior-based adjustments (opponent modeling)
3. Emotion feedback modulation

### `cl_model_c3_faster.get_arousal_valence()`

The CLIFER training path:

1. Generates imagined faces using the CAAE when `use_imagined` is enabled
2. Encodes observed and imagined faces via FaceChannel
3. Trains or updates GDM-E and GDM-S
4. Saves the updated GWR memories

`cl_model_c3_faster.predict()` loads the saved memories and annotates the current
frames with personalized arousal and valence. The corresponding functions in
`cl_model_c3.py` provide an alternate source path.

### `cl_model_c3_faster.train_GDM()`

Implements the dual-memory continual learning:

- GDM-E: Fast adaptation (`max_nodes=len(data)`, `age=600`)
- GDM-S: Slow consolidation (`max_nodes=len(data)//2`, `age=1200`)

Both memories use `EpisodicGWR`; the semantic memory learns from representations
produced by the episodic memory. The pre-trained CAAE supplies imagined faces.

---

## Dependencies

```
Core:           HATN (negotiation protocol)
ML Framework:   TensorFlow graph/session runtime compatible with the CAAE model
Affect Model:   FaceChannel (pre-trained dimensional model)
CL Memory:      GWR (Growing When Required networks)
```

## Runtime Setup

The original entry point, `solver_cl_agent/main.py`, currently selects `Hybrid`;
the differential-feedback implementation mapped above is `SolverAgent`.
Selecting and connecting the study configuration requires the matching lab
interaction adapters, trusted model assets, and separate perception and robot
environments. The [source guide](README_CODE.md) identifies these dependencies;
[runtime setup](../docs/runtime.md) describes the optional headless Solver
launcher and its process and affect contracts.

[Tested environments](../docs/tested-environments.md) records the bounded SDK and
model checks. Those checks and the independent public-CSV verifier do not
establish live capture, online learning, model publication, or physical robot delivery.
See [maintenance guidance](../docs/maintenance.md) for environment separation and
source-preservation details.
