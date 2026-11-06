"""
Advanced Multi-Curriculum Reasoning & Biological Mechanism Dataset Engine
==========================================================================
Engineered specifically for CerebroSpheroid Living Organoid Architectures.
Disciplines:
1. ARC-AGI Spatial & Topological Grid Transformations (Reflections, Symmetry, Pathfinding)
2. Big-Bench Hard (BBH) Algorithmic Logic (Boolean Chains, Object Tracking, Temporal Ordering)
3. Advanced Quantitative & Proof-Style Reasoning (Modular Arithmetic, Combinatorics, Kinematics)
4. Biochemical & Cellular Causal Mechanisms (Enzyme Kinetics, Glycolysis, Chemiosmosis, Le Chatelier)
5. Direct Natural Environmental Stimulus -> Cognitive Articulation (Zero chatbot boilerplate)
"""

import os
import re
import math
import random
import torch
from typing import Tuple, List, Dict, Generator


# =========================================================================
# 1. ARC-AGI Spatial & Topological Transformation Reasoning
# =========================================================================

def gen_arc_spatial_reflection():
    """Generates 2D matrix reflection reasoning tasks (Horizontal/Vertical)."""
    grid_size = random.choice([3, 4])
    symbols = [".", "#", "*", "+", "O", "@"]
    grid = [[random.choice(symbols) for _ in range(grid_size)] for _ in range(grid_size)]
    axis = random.choice(["horizontal", "vertical"])

    grid_str = "\n".join(" ".join(row) for row in grid)

    if axis == "horizontal":
        # Flip across horizontal axis (reverse rows)
        flipped = list(reversed(grid))
        op_desc = "reflecting across the horizontal axis reverses row order"
    else:
        # Flip across vertical axis (reverse columns within rows)
        flipped = [list(reversed(row)) for row in grid]
        op_desc = "reflecting across the vertical axis reverses column order within each row"

    flipped_str = "\n".join(" ".join(row) for row in flipped)

    q = (f"Given the {grid_size}x{grid_size} spatial matrix:\n{grid_str}\n"
         f"Transform this matrix by applying a {axis} reflection.")
    a = (f"To apply a {axis} reflection, {op_desc}. "
         f"The transformed {grid_size}x{grid_size} matrix is:\n{flipped_str}")
    return q, a


def gen_arc_bounding_box_and_invariance():
    """Generates ARC object bounding box extraction and invariant counting."""
    w, h = random.randint(4, 6), random.randint(4, 6)
    bg = "."
    fg = random.choice(["#", "*", "X"])
    # Create empty grid
    grid = [[bg for _ in range(w)] for _ in range(h)]
    # Place a random rectangle
    rx = random.randint(0, w - 3)
    ry = random.randint(0, h - 3)
    rw = random.randint(2, min(3, w - rx))
    rh = random.randint(2, min(3, h - ry))

    count = rw * rh
    for r in range(ry, ry + rh):
        for c in range(rx, rx + rw):
            grid[r][c] = fg

    grid_str = "\n".join(" ".join(row) for row in grid)
    q = (f"Analyze the spatial grid below:\n{grid_str}\n"
         f"Identify the bounding box dimensions (width x height) and total count of active '{fg}' symbols.")
    a = (f"The connected cluster of '{fg}' spans from column {rx + 1} to {rx + rw} (width {rw}) "
         f"and row {ry + 1} to {ry + rh} (height {rh}). "
         f"The bounding box is {rw} x {rh}. "
         f"Multiplying width by height gives {rw} * {rh} = {count} active symbols. "
         f"The final count is {count}.")
    return q, a


def gen_arc_topological_connectivity():
    """Generates path connectivity / maze traversal between two coordinates."""
    path_len = random.randint(3, 5)
    start_col = random.randint(1, 3)
    steps = []
    curr_c = start_col
    for r in range(1, path_len + 1):
        move = random.choice([-1, 0, 1])
        curr_c = max(1, min(5, curr_c + move))
        steps.append((r, curr_c))

    coords_str = " -> ".join([f"({r},{c})" for r, c in steps])
    target = steps[-1]

    q = (f"A continuous topological wave initiates at coordinate ({steps[0][0]},{steps[0][1]}) "
         f"and propagates along trajectory {coords_str}. "
         f"What is the final coordinate reached, and what is the total Manhattan displacement from origin?")
    dr = abs(target[0] - steps[0][0])
    dc = abs(target[1] - steps[0][1])
    disp = dr + dc
    a = (f"The terminal node of the trajectory is ({target[0]},{target[1]}). "
         f"The displacement across rows is |{target[0]} - {steps[0][0]}| = {dr}, "
         f"and across columns is |{target[1]} - {steps[0][1]}| = {dc}. "
         f"The total Manhattan displacement is {dr} + {dc} = {disp}. "
         f"The final answer is coordinate ({target[0]},{target[1]}) with displacement {disp}.")
    return q, a


# =========================================================================
# 2. Big-Bench Hard (BBH) Algorithmic Logic
# =========================================================================

def gen_bbh_boolean_chain():
    """Generates multi-operator Boolean evaluation tasks."""
    vals = [True, False]
    v1, v2, v3 = random.choice(vals), random.choice(vals), random.choice(vals)
    op1 = random.choice(["and", "or"])
    op2 = random.choice(["and", "or"])
    negate2 = random.choice([True, False])

    expr_str = f"({v1} {op1} {'not ' if negate2 else ''}{v2}) {op2} {v3}"

    sub2 = (not v2) if negate2 else v2
    if op1 == "and":
        sub1 = v1 and sub2
    else:
        sub1 = v1 or sub2

    if op2 == "and":
        final = sub1 and v3
    else:
        final = sub1 or v3

    q = f"Evaluate the truth value of the Boolean expression: {expr_str}."
    a = (f"Evaluating the inner term: '{v1} {op1} {'not ' if negate2 else ''}{v2}'. "
         f"{'not ' + str(v2) if negate2 else str(v2)} evaluates to {sub2}. "
         f"Then {v1} {op1} {sub2} evaluates to {sub1}. "
         f"Finally, evaluating {sub1} {op2} {v3} yields {final}. "
         f"The answer is {final}.")
    return q, a


def gen_bbh_shuffled_objects():
    """Generates tracking shuffled objects across state transitions."""
    names = ["Alice", "Bob", "Charlie", "David"]
    balls = ["Red", "Blue", "Green", "Yellow"]
    random.shuffle(names)
    random.shuffle(balls)
    state = dict(zip(names, balls))

    # Pick 2 swap operations
    p1, p2 = random.sample(names, 2)
    p3, p4 = random.sample(names, 2)

    init_str = ", ".join([f"{n} has the {b} ball" for n, b in state.items()])

    # Apply swaps
    state[p1], state[p2] = state[p2], state[p1]
    state[p3], state[p4] = state[p4], state[p3]

    target_person = random.choice(names)
    correct_ball = state[target_person]

    q = (f"Initially: {init_str}. "
         f"First swap: {p1} and {p2} swap balls. "
         f"Second swap: {p3} and {p4} swap balls. "
         f"Who has the {correct_ball} ball now, and which ball does {target_person} have?")
    a = (f"Tracking ball positions step by step: "
         f"After {p1} and {p2} swap, {p1} holds {state[p1] if target_person not in [p1,p2] else 'swapped ball'}. "
         f"After {p3} and {p4} swap, all positions settle. "
         f"{target_person} holds the {correct_ball} ball. "
         f"The answer is {target_person} has the {correct_ball} ball.")
    return q, a


# =========================================================================
# 3. High-Level Quantitative & Proof-Style Reasoning
# =========================================================================

def gen_modular_arithmetic():
    """Generates modular congruence and remainder deduction."""
    mod = random.choice([7, 11, 13, 17])
    a = random.randint(15, 60)
    b = random.randint(15, 60)
    op = random.choice(["+", "*"])

    if op == "+":
        result = (a + b) % mod
        calc_str = f"({a} + {b}) = {a + b}"
    else:
        result = (a * b) % mod
        calc_str = f"({a} * {b}) = {a * b}"

    q = f"Compute the remainder when {calc_str.split(' = ')[0]} is divided by {mod} (i.e. find {calc_str.split(' = ')[0]} mod {mod})."
    a = (f"First, calculate the value: {calc_str}. "
         f"Next, dividing {calc_str.split(' = ')[1]} by {mod}: {calc_str.split(' = ')[1]} = {int(calc_str.split(' = ')[1]) // mod} * {mod} + {result}. "
         f"The remainder is {result}. "
         f"The answer is {result}.")
    return q, a


def gen_combinatorics_probability():
    """Generates urn model probability calculations."""
    red = random.randint(3, 8)
    blue = random.randint(3, 8)
    total = red + blue
    prob_pct = (red / total) * 100.0

    q = f"An urn contains {red} red marbles and {blue} blue marbles. One marble is drawn at random. What is the probability of drawing a red marble?"
    a = (f"The total number of marbles is {red} + {blue} = {total}. "
         f"The number of favorable outcomes (drawing red) is {red}. "
         f"The probability is {red}/{total}, which is approximately {prob_pct:.1f}%. "
         f"The answer is {red}/{total}.")
    return q, a


# =========================================================================
# 4. Biochemical, Cellular & Physical Mechanism Reasoning
# =========================================================================

BIO_MECHANISMS = [
    {
        "q": "What happens to the rate of an enzyme-catalyzed reaction when the substrate concentration [S] is much greater than the Michaelis constant Km ([S] >> Km)?",
        "a": "When [S] >> Km, all enzyme active sites become saturated with substrate. The reaction rate approaches maximum velocity Vmax, obeying zero-order kinetics where further increases in substrate concentration do not increase velocity.",
    },
    {
        "q": "How does a competitive enzyme inhibitor affect the apparent Vmax and Km of an enzyme?",
        "a": "A competitive inhibitor binds reversibly to the active site, competing directly with the substrate. It increases the apparent Km (lowering enzyme affinity) while leaving Vmax unchanged, because high substrate concentrations can outcompete the inhibitor.",
    },
    {
        "q": "What is the primary driving force for ATP synthesis during oxidative phosphorylation in mitochondria?",
        "a": "The primary driving force is the proton-motive force (electrochemical proton gradient) across the inner mitochondrial membrane, established by proton pumping from complexes I, III, and IV, which drives protons through the F0F1 ATP synthase complex.",
    },
    {
        "q": "According to Le Chatelier's principle, how will an exothermic reversible chemical reaction (A + B <=> C + Heat) respond to an increase in temperature?",
        "a": "An increase in temperature adds thermal energy to the system. By Le Chatelier's principle, the equilibrium shifts in the endothermic reverse direction (toward reactants A and B) to absorb excess heat, decreasing product C yield.",
    },
    {
        "q": "Why does RNA possess lower chemical stability and higher spontaneous hydrolysis rates compared to DNA?",
        "a": "RNA contains a 2'-hydroxyl group on its ribose sugar ring, which acts as an internal nucleophile that attacks the adjacent phosphodiester bond, causing spontaneous strand cleavage. DNA lacks this group (having 2'-deoxyribose), conferring greater chemical stability.",
    },
    {
        "q": "How does myelin sheath insulation increase the conduction velocity of action potentials along an axon?",
        "a": "Myelin increases membrane electrical resistance and decreases membrane capacitance, preventing ion leakage. Action potentials jump rapidly between unmyelinated Nodes of Ranvier where voltage-gated sodium channels concentrate, a process termed saltatory conduction.",
    },
]


def gen_bio_mechanism():
    item = random.choice(BIO_MECHANISMS)
    return item["q"], item["a"]


# =========================================================================
# 5. Master Multi-Curriculum Generator & Streamer
# =========================================================================

GENERATORS = [
    gen_arc_spatial_reflection,
    gen_arc_bounding_box_and_invariance,
    gen_arc_topological_connectivity,
    gen_bbh_boolean_chain,
    gen_bbh_shuffled_objects,
    gen_modular_arithmetic,
    gen_combinatorics_probability,
    gen_bio_mechanism,
]


def generate_advanced_sample() -> Tuple[str, str]:
    """Generates a random high-level reasoning or causal mechanism sample."""
    fn = random.choice(GENERATORS)
    return fn()


def get_advanced_organoid_stream(
    tokenizer,
    batch_size: int = 4,
    max_prompt_len: int = 512,
    max_resp_len: int = 128,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Infinite generator streaming advanced multi-curriculum samples:
    ARC-AGI spatial + BBH algorithmic + Math proofs + Biochemical mechanisms.
    Strictly formatted as natural stimulus -> articulation pairs.
    """
    batch_p, batch_t = [], []

    while True:
        q, a = generate_advanced_sample()
        prompt_text = q.strip()
        resp_text = f" {a.strip()}"

        p_toks = tokenizer.encode(prompt_text, max_length=max_prompt_len, truncation=True)
        t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

        if len(p_toks) >= 4 and len(t_toks) >= 4:
            batch_p.append(p_toks)
            batch_t.append(t_toks)

            if len(batch_p) == batch_size:
                max_p = max(len(p) for p in batch_p)
                max_t = max(len(t) for t in batch_t)
                p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
                t_pad = [t + [-100] * (max_t - len(t)) for t in batch_t]
                yield (
                    torch.tensor(p_pad, dtype=torch.long),
                    torch.tensor(t_pad, dtype=torch.long),
                )
                batch_p, batch_t = [], []


# =========================================================================
# 6. Continuous Multimodal Visual & Audio Sensory Generators
# =========================================================================

def generate_synthetic_multimodal_sample() -> Tuple[str, str, torch.Tensor, torch.Tensor]:
    """
    Synthesizes a continuous cross-modal environmental stimulus:
    - 2D Visual Image [3, 64, 64]: geometric primitives (circle, box, cross) + color channels.
    - 1D Audio Waveform [1, 16000]: acoustic tones, frequency sweeps, or harmonic intervals.
    - Natural Sensory Question & Deductive Articulation.
    """
    # 1. Generate 2D Visual Image
    img = torch.zeros(3, 64, 64, dtype=torch.float32)
    shapes = ["circle", "rectangle", "cross"]
    colors = ["red", "green", "blue"]
    chosen_shape = random.choice(shapes)
    chosen_color = random.choice(colors)

    y, x = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij")
    cx, cy = random.choice([-0.25, 0.0, 0.25]), random.choice([-0.25, 0.0, 0.25])

    c_idx = {"red": 0, "green": 1, "blue": 2}[chosen_color]

    if chosen_shape == "circle":
        r = 0.35
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    elif chosen_shape == "rectangle":
        mask = ((x - cx).abs() <= 0.30) & ((y - cy).abs() <= 0.20)
    else:  # cross
        mask = (((x - cx).abs() <= 0.08) & ((y - cy).abs() <= 0.40)) | (((x - cx).abs() <= 0.40) & ((y - cy).abs() <= 0.08))

    img[c_idx, mask] = 0.85
    # Add gentle ambient spatial gradient
    img += 0.05 * torch.randn_like(img)

    # 2. Generate 1D Audio Waveform (1 second at 16kHz)
    sr = 16000
    t = torch.linspace(0, 1.0, sr, dtype=torch.float32)
    audio_modes = ["pure_tone", "ascending_chirp", "harmonic_chord"]
    chosen_audio = random.choice(audio_modes)

    if chosen_audio == "pure_tone":
        freq = random.choice([220, 440, 880])
        audio = torch.sin(2.0 * math.pi * freq * t).unsqueeze(0)
        audio_desc = f"a pure {freq} Hz acoustic tone"
    elif chosen_audio == "ascending_chirp":
        audio = torch.sin(2.0 * math.pi * (200.0 + 600.0 * t * 0.5) * t).unsqueeze(0)
        audio_desc = "an ascending frequency sweep from 200 Hz to 800 Hz"
    else:  # harmonic chord
        f1, f2 = 440.0, 660.0  # Perfect fifth
        audio = (0.6 * torch.sin(2.0 * math.pi * f1 * t) + 0.4 * torch.sin(2.0 * math.pi * f2 * t)).unsqueeze(0)
        audio_desc = "a harmonic fifth acoustic chord (440 Hz and 660 Hz)"

    # Add soft amplitude envelope and noise
    env = torch.sin(math.pi * t)
    audio = (audio * env + 0.02 * torch.randn_like(audio)).clamp(-1.0, 1.0)

    # 3. Formulate Natural Stimulus & Direct Articulation
    q = f"Describe the visual geometry and acoustic frequency characteristics present in the sensory field."
    a = (f"The visual retinotopic sheet detects a {chosen_color} {chosen_shape} on Hemisphere A. "
         f"Simultaneously, the auditory tonotopic sheet resonates with {audio_desc}. "
         f"The unified sensory field binds the {chosen_color} {chosen_shape} with {audio_desc}.")

    return q, a, img, audio


def get_multimodal_organoid_stream(
    tokenizer,
    batch_size: int = 4,
    max_prompt_len: int = 256,
    max_resp_len: int = 64,
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """
    Yields batched multimodal inputs:
    (prompt_tokens, target_tokens, images, audio)
    """
    b_p, b_t, b_img, b_aud = [], [], [], []

    while True:
        q, a, img, aud = generate_synthetic_multimodal_sample()
        p_toks = tokenizer.encode(q.strip(), max_length=max_prompt_len, truncation=True)
        t_toks = tokenizer.encode(f" {a.strip()}", max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

        b_p.append(p_toks)
        b_t.append(t_toks)
        b_img.append(img)
        b_aud.append(aud)

        if len(b_p) == batch_size:
            max_p = max(len(p) for p in b_p)
            max_t = max(len(t) for t in b_t)
            p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in b_p]
            t_pad = [t + [-100] * (max_t - len(t)) for t in b_t]

            yield (
                torch.tensor(p_pad, dtype=torch.long),
                torch.tensor(t_pad, dtype=torch.long),
                torch.stack(b_img, dim=0),
                torch.stack(b_aud, dim=0),
            )
            b_p, b_t, b_img, b_aud = [], [], [], []

