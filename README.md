<h1 align="center">
Personalized Affective Perception via Continual Learning in Human-Agent Negotiation
</h1>

<p align="center">
  <a href="#2-abstract">Abstract</a> ·
  <a href="#4-proposed-method-implementation-details">Method</a> ·
  <a href="#5-experimental-dataset">Study Data</a> ·
  <a href="#10-reproducibility">Reproducibility</a> ·
  <a href="#8-citation">Citation</a>
</p>

**TL;DR:** Continual learning personalizes the agent’s facial affect model to the person it is negotiating with. The model estimates valence and arousal. Changes in the direction and intensity of these estimates adjust the value the agent seeks for itself in its next offer.

**Keywords:** Human-Agent Interaction, Human-Robot Interaction, Automated Negotiation, Affective Computing, Continual Learning, Facial Expression Recognition, Personalization, Emotion Recognition

---

## 1. System Architecture

The study architecture combines **CLIFER** affect perception with the Solver negotiation agent:

<p align="center">
  <img src="assets/HAIFramework.png" width="850" alt="System Architecture">
</p>

**Key Components:**

- **Speech to Offer Module**: Converts human verbal offers to structured bids
- **Facial Expression Analysis Module**: Real-time affect perception via CLIFER
- **Negotiator Core**: Manages protocol state, utility space, and offer history
- **Solver Agent**: Decision-making logic (Bidding Strategy, Opponent Model, Acceptance Strategy)
- **CLIFER**: Dual-memory architecture with Episodic (GDM-E) and Semantic (GDM-S) memories

---

## 2. Abstract

Self-interested intelligent agents that negotiate with humans should interpret nonverbal expressions and adapt their behavior accordingly. However, generic facial-affect models degrade across individuals and contexts, creating a generalization gap. We introduce an affect-adaptive negotiation strategy that incorporates Continual Learning with Imagination for Facial Expression Recognition (CLIFER) to personalize valence-arousal perception in real time. Differential feedback converts changes in valence-arousal direction and intensity into adjustments of the agent's target utility. An asynchronous architecture updates CLIFER in the background during live interaction. In a counterbalanced within-subject study in which 66 participants negotiated with a humanoid robot, the CL-enabled agent achieved significantly higher individual utility and elicited higher participant concession rates than the FaceChannel baseline. The CL condition yielded significantly higher mean arousal estimates, alongside a broader output range. Participant utility and Nash distance did not change significantly between conditions. These outcomes support the practical value of coupling continual affective personalization with differential feedback for real-time negotiation.

---

## 3. Prerequisites

### Hardware Requirements

- **Study platform**: the NAO robot used in the paper; live camera, speech and robot operation require the matching lab adapters and assets.
- **Software-only checks**: no camera, robot or GPU is required for public CSV verification and the headless Solver fixtures.
- **Perception model**: use the selected legacy runtime and trusted model assets. A bounded CPU inference profile was tested separately; this does not validate a complete live session.

### Software Requirements

- **Public-data verification and software tests**: Python 3.12, with the separate hashed root lockfiles.
- **Robot bridge**: the matching legacy Python 2.7/NAOqi environment, isolated from the modern host. Modern `qi` is a different interface.
- **Perception stack**: the tested Linux/amd64 Intel TensorFlow 2.4.0 profile and its dependencies, isolated from both environments above.

### Trained Models

The pre-trained CAAE (Conditional Adversarial Auto-Encoder) checkpoint entries are included in `src_code/solver_cl_agent/FaceChannel/CLModel/checkpoint/` through Git LFS. A regular LFS-enabled clone retrieves the three checkpoint files; source-focused checkouts may omit them.

The FaceChannel H5 model and matching lab assets must be supplied separately from trusted sources. For the original CLIFER trained models, see the [CLIFER Repository](https://github.com/nchuramani/CLIFER). Model identities and acquisition controls are documented in [maintenance guidance](docs/maintenance.md).

<a id="5-proposed-method-implementation-details"></a>

## 4. Proposed Method: Implementation Details

The paper couples CLIFER with the Solver Agent to personalize affect perception during real-time negotiation. The components below implement this approach; the [architecture guide](src_code/ARCHITECTURE.md) maps them to the source.

### 4.1. Generative Replay (CLIFER Module)

Located in `src_code/solver_cl_agent/FaceChannel/CLModel/`:

- **Episodic Memory (GDM-E)**: Dynamic buffer for rapid adaptation to user-specific frames.
- **Semantic Memory (GDM-S)**: The semantic GWR memory for longer-term representations. The pretrained CAAE supplies imagined samples; it is distinct from GDM-S.
- **Replay objective (conceptual)**: Learning from observed and imagined samples can be summarized schematically as:

    ```
    L_total = L_real(F_r) + λ · L_imagined(F_gen)
    ```

  This is a conceptual replay schematic. The retained GWR update path does not implement a separate weighted-loss optimizer with a tunable λ; see `train_GDM()` in the architecture map.

### 4.2. Differential Affective Feedback (P_E)

Located in `src_code/solver_cl_agent/Agent/Solver_Agent/Solver_Agent.py` (Lines 467-503):

The agent calculates a **Differential Affective Feedback** signal ($P_E$) that decomposes emotional shift into directional and intensity components:

```
# Session-Adaptive Normalization
Ẽ_t = (E_t - μ_t) / (σ_t + ε)

# Directional Sign (handles "Anger Paradox")
P_sign = sgn(ΔV + α · ΔA)

# Rotation Component (qualitative change)
P_rot = atan2(δ_det, δ_dot)   # normalized by π

# Radial Component (intensity change)
P_rad = |‖Ẽ_t‖ - ‖Ẽ_{t-1}‖|  # normalized by √2

# Final Signal
P_E = P_sign × min(√(P_rot² + P_rad²), 1.0)
```

where `δ_det = det(Ẽ_{t-1}, Ẽ_t)` and `δ_dot = Ẽ_{t-1} · Ẽ_t`. In the retained implementation, α=1.0 and a zero directional sum gives +1. The normalization notation describes the session-level affect representation; the lab perception pipeline supplies the operational raw and normalized values. The compatibility launcher consumes its existing raw and normalized channels without fitting a replacement normalizer.

<p align="center">
  <img src="assets/differentialAffectiveFeedback.png" width="450" alt="Differential Feedback">
</p>

### 4.3. Asynchronous Architecture

The paper describes two cooperating execution paths, coordinated through `src_code/solver_cl_agent/main.py` and the perception modules:

- **Thread 1 (Interaction)**: Captures frames and bids using the *frozen* model (θ_curr).
- **Thread 2 (Learning)**: Asynchronously updates weights (θ_train) using Generative Replay and syncs them back.

The live capture, training/publication and robot lifecycle requires the matching lab integration. The headless software checks exercise the Solver and transport boundaries rather than this full live pipeline.

---

<a id="6-experimental-dataset"></a>

## 5. Experimental Dataset

The folder `final_dataset/` contains the anonymized data from our within-subjects study (N=66, representing 132 negotiation sessions).

The paper reports a power assessment ($N=66$ paired samples provides power $\ge 0.80$ to detect an effect size of $d=0.35$), Shapiro-Wilk normality checks, and Linear Mixed-Effects Models (LMMs) with subject-level random intercepts for the nested round data. These manuscript-level analyses are distinct from the declared public CSV checks. [Verification scope](docs/public-data-verification.md) identifies the checks provided with this repository.


<a id="7-key-results-summary"></a>

## 6. Key Results Summary

The following tables and figures summarize the study's five research questions. The [public-data verification guide](docs/public-data-verification.md) identifies which results can be checked from the accompanying CSVs and distinguishes those checks from questionnaire, Nash, mediation and mixed-model analyses.

### Experimental Procedure

<p align="center">
  <img src="assets/ExperimentProcedure.png" width="500" alt="Experiment Procedure">
</p>

Each participant negotiated twice with the NAO robot "Solver" in a counterbalanced within-subjects design:

- **Demo Session**: Grocery domain (calibration phase for CL initialization)
- **Session 1 & 2**: Holiday domain with FaceChannel (FC) or CLIFER (CL) conditions

<p align="center">
  <img src="assets/BidSpace.png" width="450" alt="Utility Space">
</p>

### 6.1. Affective Perception (RQ1)

**Subject-Level Affect Signal (Paper Fig. 6a):** The CL condition has higher estimated arousal at the subject level (reported d=0.69).

<p align="center">
  <img src="assets/MoveAffect_Raincloud_SubjectLevel.png" width="550" alt="Subject-Level Raincloud">
</p>

**Round-Level Affect Signal (Paper Fig. 6b):** The reported round-level distribution difference is d=0.88; repeated rounds are nested within participants.

<p align="center">
  <img src="assets/MoveAffect_Raincloud_RoundLevel.png" width="550" alt="Round-Level Raincloud">
</p>

<div align="center">

| Metric | FC | CL | *p* | *d* |
|--------|----|----|-----|-----|
| **Arousal (Round)** | 0.01 ± 0.09 | **0.15 ± 0.22** | **<0.001** | **0.88** |
| Valence (Round) | 0.03 ± 0.18 | 0.07 ± 0.20 | <0.001 | 0.18 |
| **Arousal (Subject)** | 0.02 ± 0.07 | **0.14 ± 0.16** | **<0.001** | **0.69** |
| Valence (Subject) | 0.03 ± 0.13 | 0.05 ± 0.13 | 0.317 | 0.13 |

</div>

- **Estimated Arousal**: The study reports a subject-level difference of $d_z=0.69$, with **75.8%** of participants showing higher estimated arousal under CL. This describes model outputs; independent affect ground truth is not supplied by these CSVs.

<p align="center">
  <img src="assets/Responder_Analysis.png" width="550" alt="Responder Analysis">
</p>

- **Temporal Pattern (Paper Fig. 6d)**: The study reports estimated-arousal differences of Δ=0.05 (early) → Δ=0.19 (mid) → Δ=0.15 (late). This descriptive pattern does not isolate online adaptation from calibration or other system components.

<p align="center">
  <img src="assets/Temporal_Evolution.png" width="600" alt="Temporal Evolution">
</p>

- **Signal Coupling (Paper Fig. 6e-g)**: The study reports a positive pooled arousal-valence correlation under CL ($r=0.45$) and a negative correlation under FaceChannel ($r=-0.15$), with within-session summaries of FC: $r=-0.27$ vs. CL: $r=0.88$. These are descriptive model-output associations. The original overview also reported a paired Fisher-z statistic ($t(65)=7.73$, $p<0.001$, $d_z=0.95$); that historical statistic is outside the current verifier's claims. Its finite within-session Fisher summaries include 63 CL and 66 FC sessions and are not treated as a paired test.

<p align="center">
  <img src="assets/Correlation_Scatter.png" width="350" alt="Round-Level Correlation">
  <img src="assets/Balanced_Scatter.png" width="350" alt="Balanced Correlation">
</p>
<p align="center">
  <img src="assets/Subject_Scatter.png" width="350" alt="Subject-Level Correlation">
</p>

### 6.2. Behavioral Influence (RQ2)

<p align="center">
  <img src="assets/BehavioralAnalysis.png" width="500" alt="Behavioral Analysis">
</p>

<div align="center">

| Move Type | FC | CL | *t* | *p* | *d* |
|-----------|----|----|-----|-----|-----|
| **Concession** | 0.28 ± 0.20 | **0.37 ± 0.27** | 2.55 | **0.013** | 0.32 |
| Nice | 0.04 ± 0.08 | 0.04 ± 0.09 | 0.23 | 0.82 | 0.03 |
| Fortunate | 0.18 ± 0.16 | 0.15 ± 0.14 | -1.11 | 0.27 | -0.14 |
| Unfortunate | 0.21 ± 0.17 | 0.21 ± 0.18 | -0.25 | 0.80 | -0.03 |
| Selfish | 0.23 ± 0.19 | 0.19 ± 0.17 | -1.38 | 0.17 | -0.17 |
| Silent | 0.06 ± 0.14 | 0.04 ± 0.11 | -0.87 | 0.39 | -0.11 |

</div>

- **Concession Rate**: The reported participant concession rate increased **from 28.0% to 37.3%** (a relative increase of 33.3%) with the CL agent (raw p=0.013, d=0.32). The multiple-comparison family affects the adjusted inference; see [public CSV verification](docs/public-data-verification.md).
- **Move × Condition Pattern**: The study reports higher estimated arousal across move types, with the largest reported differences for Concession (d=1.05) and Selfish (d=1.10).

### 6.3. Negotiation Outcomes (RQ3)

<p align="center">
  <img src="assets/NegotiationOutcomes.png" width="600" alt="Negotiation Outcomes">
</p>

<div align="center">

| Metric | FC (Mean±SD) | CL (Mean±SD) | *t* | *p* | *d* |
|--------|--------------|--------------|-----|-----|-----|
| **Agent Utility** | 0.73 ± 0.08 | **0.75 ± 0.06** | 2.46 | **0.017** | 0.31 |
| Participant Utility | 0.76 ± 0.09 | 0.75 ± 0.10 | -0.69 | 0.49 | -0.09 |
| Nash Distance | 0.12 ± 0.08 | 0.11 ± 0.08 | -0.89 | 0.378 | -0.11 |
| Agreement Rounds | 8.56 ± 5.02 | 7.61 ± 3.83 | -1.52 | 0.13 | -0.19 |

</div>

- **Agent Utility**: The reported difference favors CL: **CL (0.75) > FC (0.73)**, p=0.017, d=0.31.
- **Participant and Joint Outcomes**: Participant Utility and Nash Distance showed no significant difference. Nonsignificance does not establish equivalence or prove non-exploitation.

### 6.4. Negotiation Efficiency (RQ4)

- **Efficiency**: CL required 11% fewer rounds (7.6 vs 8.6 rounds), but the difference was not significant (p=0.13, d=-0.19).
- **Nash Distance**: The reported difference was nonsignificant (FC: 0.12, CL: 0.11, p=0.378). This comparison does not establish equal joint outcomes.

### 6.5. Subjective Perception (RQ5)

<p align="center">
  <img src="assets/QuestionnaireBoxChart.png" width="500" alt="Subjective Questionnaire">
</p>

<div align="center">

| Question | *p* | *d* |
|----------|-----|-----|
| **Q1 (Preference Modeling)**: Solver modelled my preferences accurately | **0.013** | 0.32 |
| **Q2 (Affective Awareness)**: Solver considered my emotional state | **0.015** | 0.30 |
| **Q3 (Persuasiveness)**: Solver was convincing during the negotiation | **0.043** | 0.26 |
| Q4 (Participant Decisions): Solver's gestures affected my offer behavior | 0.058 | -0.21 |
| Q5 (Time Awareness): While Solver was making its offers, it considered the time | 0.121 | -0.19 |

</div>

The reported questionnaire comparisons favored the CL agent for preference modeling, affective awareness and persuasiveness. These ratings do not identify a causal social-halo mechanism; questionnaire analyses are outside the public CSV verifier's declared coverage.

---

<a id="8-related-repositories"></a>

## 7. Related Repositories

Related software includes the maintained NEGOTIATOR framework and the foundational implementations for affective perception below:

<div align="center">

| Component | Repository |
|-----------|------------|
| **CLIFER** (Continual Learning) | [nchuramani/CLIFER](https://github.com/nchuramani/CLIFER) |
| **NEGOTIATOR** (Human–Agent and Human–Robot Negotiation Framework) | [monurkeskin/NEGOTIATOR-IJCAI-2024](https://github.com/monurkeskin/NEGOTIATOR-IJCAI-2024) |
| **FaceChannel** | [pablovin/FaceChannel](https://github.com/pablovin/FaceChannel) |

</div>

---

<a id="9-citation"></a>

## 8. Citation

Use the manuscript citation below. [CITATION.cff](CITATION.cff), [reference.bib](reference.bib) and [CodeMeta](codemeta.json) provide citation and software metadata. The manuscript remains under review.

```bibtex
@article{PersonalizedAffectivePerception2025,
  title   = {Personalized Affective Perception via Continual Learning in Human-Agent Negotiation},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2025}
}
```

---

<a id="10-license"></a>

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

<a id="11-verification-and-runtime-checks"></a>

## 10. Reproducibility

The public CSVs let you examine participant-level outcomes and affect summaries from the study. Run the verification script with Python 3.12:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python verify_all_results.py
```

The script compares its calculations with the recorded reference results. The [public-data guide](docs/public-data-verification.md) explains the analyses and their coverage.

Run the software tests in the same environment:

```sh
python -m pip install --require-hashes -r requirements-test.lock
python -m pytest tests -q
python -m submission_runtime doctor
python -m submission_runtime smoke
```

The tests exercise the agent’s decision logic and runtime without robot hardware. The [runtime guide](docs/runtime.md) provides a headless negotiation example and instructions for connecting the perception models, camera and robot for a live session.
