# 🧠 CerebroSpheroid V4: Project Roadmap & Deliberation Plan

This document tracks the architectural roadmap, training milestones, and the transition from **Proto-Reasoning** to **Formal Chain-of-Thought Deduction** via explicit `<think>` deliberation.

---

## Current Status: Phase 1 (Foundation & Manifold Optimization)
- **Active Champion Checkpoint**: `cerebro_spheroid_v4_step_48300.pt` (43/43 pure tensors, 0 dropped).
- **Core Architecture**: Concentric spherical shells ($K=8$, $D=148$) with 5th-order Newton-Schulz polar isometry.
- **Context Management**: Elastic Sliding Context Window ($W_{\text{total}}=1024$ tokens) across 5 complementary static buckets (`short_cot`, `standard_cot`, `balanced`, `deep_reasoning [256p, 512t]`, `long_context [768p, 256t]`).
- **Batching & Hardware**: JAX high-throughput engine on Cloud TPU v2/v3/v5e with `batch_size=64` (grad accum = 1).
- **Protocol Framing**: Biological enclosed delimiters `<stim> ... </stim>` $\to$ deliberate $\to$ `<act> ... </act></s>`.
- **Observed Emergence**: Proto-reasoning verified on Step 48,300:
  - Markdown code generation (`def is_prime(n): if return False for...`).
  - Algebraic derivation decomposition (`To solve for x: ... 2x - 6 is 1 ...`).
  - 6,597x causal thought re-afference error explosion upon thought channel ablation.

---

## Phase 2: Explicit `<think>` Deliberation Integration (Roadmap)

### 1. Mathematical Rationale
Mathematical validation demonstrated that adding an untrained continuous latent tensor ($\mathbf{W}_{\text{think}}$) would discard the 48,300 steps of pre-existing optimization and lack direct gradient supervision. 

In contrast, **discrete `<think>` deliberation tokens**:
1. Provide **dense step-by-step cross-entropy loss** on every intermediate operation:
   $$\mathcal{L}_{\text{CoT}} = -\sum_{m=1}^M \log P(y_m^{\text{think}} \mid \mathbf{S}_m)$$
2. Harness the existing 43/43 tensors without changing weight shapes or resetting pretrained parameters.
3. Align directly with the 512-token target budget of the `deep_reasoning` elastic bucket.

---

### 2. Thinking Dataset Sourcing & Curation Tasks
- [ ] **Task 2.1: Ingest Verified CoT Datasets**
  - **OpenR1-Math / Bespoke-Stratos-17k**: High-quality distilled reasoning traces formatted with explicit `<think>` blocks.
  - **NuminaMath-CoT**: 860k diverse mathematical competition problems with multi-step derivations.
  - **GSM8K Thinking Tag Transformation**: Convert standard GSM8K steps into `<think> ... </think> <act> #### {answer} </act>`.
- [ ] **Task 2.2: Synthetic Thinking Data Generation (STaR)**
  - Use CerebroSpheroid V4 to sample multiple reasoning paths for GSM8K questions.
  - Verify arithmetic consistency against final numerical answers.
  - Keep correct paths as training samples (Self-Taught Reasoner loop).

---

### 3. Deliberation Protocol Specification
The unified protocol encloses the internal monologue within `<think>` and the outward motor execution within `<act>`:

$$\underbrace{\text{<stim> Prompt </stim>}}_{\text{Sensory Input}} \longrightarrow \underbrace{\text{<think> Step-by-Step Derivation </think>}}_{\text{Internal Deliberation}} \longrightarrow \underbrace{\text{<act> Final Answer </act></s>}}_{\text{Motor Execution}}$$

#### Implementation Deliverables:
- [ ] **`src/corpus.py`**:
  - Implement `format_thinking_response(reasoning, answer)`.
  - Implement regex extraction in `clean_motor_response(text, return_thought=False)`:
    - If `return_thought=True`: Returns `(thought_str, answer_str)`.
    - If `return_thought=False`: Returns clean `answer_str` stripped of `<think>` blocks.
- [ ] **`training/train_v4_jax.py` & `training/train_v4_full.py`**:
  - Add `--thinking_data` flag to load datasets containing verified reasoning traces.
  - Route samples into the `deep_reasoning (256, 512)` elastic bucket.

---

### 4. Evaluation Milestones & Exit Criteria
- [ ] **Milestone 2A**: Zero loss degradation on standard conversational prompts when `<think>` datasets are mixed in.
- [ ] **Milestone 2B**: GSM8K pass@1 accuracy improves from proto-reasoning (qualitative) to measurable formal accuracy (>30% at Step 75,000, >50% at Step 100,000).
- [ ] **Milestone 2C**: Organic emission of `</think>` before `<act>` verified with >95% grammatical reliability.
