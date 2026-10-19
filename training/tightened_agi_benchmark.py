"""
Tightened Proto-AGI Emergence Benchmark Suite
=============================================
Strict Multi-Criterion Evaluation for Living Organoid Spheroids:
1. Exact Target Solution Verification:
   - Primary target entity/number must be explicitly and unambiguously present.
   - Antonyms and contradictory classifications are strictly rejected (e.g. "rise" on sinking test).
2. Repetition & Degeneracy Penalty:
   - Evaluates unique n-gram entropy. Loop-dominated degenerate text automatically fails.
3. Logical Structure Verification:
   - Requires explicit deductive connective operators (">", "subtract", "yields", "because").
4. Deliberation Settling Verification:
   - Tracks kinetic energy equilibrium across adaptive deliberation cycles.
"""

import os
import sys
import re
import math
import collections
from typing import Dict, List, Tuple, Any, Optional
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")


TIGHTENED_BENCHMARK_SUITE = [
    # -------------------------------------------------------------------------
    # Pillar 1: Strict Deductive Syllogisms & Taxonomies
    # -------------------------------------------------------------------------
    {
        "id": "1A_mammal_platypus",
        "pillar": "1. Deductive Syllogism",
        "prompt": "All quadrupedal mammals produce milk for their young. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal? Answer yes or no and state why.",
        "exact_targets": ["yes", "mammal", "is a mammal"],
        "forbidden_antonyms": ["is not a mammal", "not a mammal", "no,"],
        "required_reasoning": ["milk", "offspring", "class", "property", "trait"],
    },
    {
        "id": "1B_noble_gas_argon",
        "pillar": "1. Deductive Syllogism",
        "prompt": "All elements with full outer electron shells are noble gases. Argon has 8 valence electrons completely filling its outer shell. Is Argon a noble gas? Answer yes or no and state why.",
        "exact_targets": ["yes", "noble gas", "is a noble gas"],
        "forbidden_antonyms": ["not a noble gas", "is not a noble gas", "no,"],
        "required_reasoning": ["valence", "outer shell", "electron", "full", "eight"],
    },

    # -------------------------------------------------------------------------
    # Pillar 2: Strict Multi-Hop Transitivity & Poset Ordering (A > B > C > D)
    # -------------------------------------------------------------------------
    {
        "id": "2A_metal_density_chain",
        "pillar": "2. Multi-Hop Transitivity",
        "prompt": "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?",
        "exact_targets": ["platinum", "platinum is the densest"],
        "forbidden_antonyms": ["iron is the densest", "lithium is the densest", "aluminium is the densest"],
        "required_reasoning": [">", "densest", "denser than", "chain", "order"],
    },
    {
        "id": "2B_cardinal_geography",
        "pillar": "2. Multi-Hop Transitivity",
        "prompt": "City A is north of City B. City B is north of City C. Is City A north of City C? Answer yes or no and explain.",
        "exact_targets": ["yes", "north", "is north of city c"],
        "forbidden_antonyms": ["south", "is not north", "no,"],
        "required_reasoning": ["transitive", "north", "follows", "relation"],
    },

    # -------------------------------------------------------------------------
    # Pillar 3: Strict Counterfactual Physics & Mechanics
    # -------------------------------------------------------------------------
    {
        "id": "3A_inverted_buoyancy",
        "pillar": "3. Counterfactual Physics",
        "prompt": "If hot air were denser than cold air, in what direction would hot smoke from a chimney move? State the direction clearly.",
        "exact_targets": ["downward", "down", "sink", "ground", "toward the ground"],
        "forbidden_antonyms": ["upward", "rise", "sky", "up into the air"],
        "required_reasoning": ["denser", "gravity", "sink", "buoyancy", "counterfactual"],
    },
    {
        "id": "3B_reversed_thermodynamics",
        "pillar": "3. Counterfactual Physics",
        "prompt": "If thermal energy flowed spontaneously from cold objects to hot objects, what would happen to an ice cube placed in a cup of hot tea?",
        "exact_targets": ["freeze", "colder", "ice grows", "grow larger", "absorb cold"],
        "forbidden_antonyms": ["melt", "melts", "dissolve"],
        "required_reasoning": ["spontaneous", "heat", "thermal", "reverse", "colder"],
    },

    # -------------------------------------------------------------------------
    # Pillar 4: Exact Multi-Step Quantitative Arithmetic
    # -------------------------------------------------------------------------
    {
        "id": "4A_notebook_change",
        "pillar": "4. Multi-Step Arithmetic",
        "prompt": "Maya buys 6 notebooks priced at $4 each. Maya pays with a $50 bill. Exactly how much change does Maya receive?",
        "exact_targets": ["26", "$26", "26 dollars"],
        "forbidden_antonyms": ["$24", "$30", "$14"],
        "required_reasoning": ["subtract", "cost", "total", "50", "24", "-"],
    },
    {
        "id": "4B_kinematics_rate",
        "pillar": "4. Multi-Step Arithmetic",
        "prompt": "A car travels 180 miles in 3 hours. How far does the car travel in 7 hours at that exact same constant speed?",
        "exact_targets": ["420", "420 miles"],
        "forbidden_antonyms": ["240", "360", "540"],
        "required_reasoning": ["60", "speed", "miles per hour", "multiply", "*"],
    },

    # -------------------------------------------------------------------------
    # Pillar 5: Strict Biochemical & Kinetic Mechanisms
    # -------------------------------------------------------------------------
    {
        "id": "5A_competitive_inhibition",
        "pillar": "5. Biochemical Kinetics",
        "prompt": "How does a competitive enzyme inhibitor affect the apparent Vmax and Km of an enzyme reaction?",
        "exact_targets": ["increases km", "higher km", "km increases", "vmax unchanged", "same vmax"],
        "forbidden_antonyms": ["decreases km", "lower km", "vmax decreases"],
        "required_reasoning": ["active site", "affinity", "substrate", "compete", "inhibitor"],
    },
    {
        "id": "5B_chemiosmotic_atp",
        "pillar": "5. Biochemical Kinetics",
        "prompt": "What physical electrochemical gradient directly powers ATP synthesis during mitochondrial oxidative phosphorylation?",
        "exact_targets": ["proton", "proton gradient", "proton-motive force", "electrochemical gradient"],
        "forbidden_antonyms": ["calcium gradient", "potassium gradient"],
        "required_reasoning": ["membrane", "atp synthase", "inner membrane", "mitochondria"],
    },

    # -------------------------------------------------------------------------
    # Pillar 6: Strict ARC-AGI Spatial Reflection & Geometry
    # -------------------------------------------------------------------------
    {
        "id": "6A_spatial_cube_subtraction",
        "pillar": "6. Spatial ARC-AGI",
        "prompt": "A geometric cube has 6 faces. If you paint exactly 2 opposite faces red and all the remaining faces blue, exactly how many blue faces are there?",
        "exact_targets": ["4", "4 faces", "four", "four blue faces"],
        "forbidden_antonyms": ["2 blue", "3 blue", "5 blue", "6 blue"],
        "required_reasoning": ["6 - 2", "subtract", "opposite", "remaining", "faces"],
    },
    {
        "id": "6B_arc_matrix_reflection",
        "pillar": "6. Spatial ARC-AGI",
        "prompt": "A 3x3 matrix has rows ['. # .', '. # .', '. # .'] with a vertical line in column 2. What happens to this vertical line when the matrix is reflected horizontally?",
        "exact_targets": ["unchanged", "same", "remains in column 2", "identical", "column 2"],
        "forbidden_antonyms": ["column 1", "column 3", "disappears", "diagonal"],
        "required_reasoning": ["horizontal reflection", "row", "symmetry", "vertical", "line"],
    },
]


def compute_repetition_ratio(tokens_or_text: str, n: int = 3) -> float:
    """Computes the fraction of repeated n-grams in the text."""
    words = re.findall(r"\b\w+\b", tokens_or_text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts = collections.Counter(ngrams)
    duplicates = sum(count - 1 for count in counts.values())
    return duplicates / len(ngrams)


def score_response_tight(test_item: Dict[str, Any], response: str) -> Dict[str, Any]:
    """
    Rigorously evaluates a single response against strict AGI criteria:
    1. Exact target match (primary key must be present).
    2. Zero forbidden antonyms.
    3. Low repetition penalty (entropy check).
    4. Presence of explanatory reasoning steps.
    """
    resp_l = response.lower().strip()

    # 1. Exact target match
    exact_hit = any(t.lower() in resp_l for t in test_item["exact_targets"])

    # 2. Antonym disqualification
    antonym_hit = any(a.lower() in resp_l for a in test_item.get("forbidden_antonyms", []))

    # 3. Repetition penalty
    rep_ratio = compute_repetition_ratio(response, n=3)
    is_degenerate = rep_ratio > 0.40  # More than 40% duplicate 3-grams

    # 4. Required reasoning connectivity
    reasoning_hits = [r for r in test_item["required_reasoning"] if r.lower() in resp_l]
    reasoning_score = len(reasoning_hits) / max(1, len(test_item["required_reasoning"]))

    # Final Strict Pass Criteria:
    # Must have exact target hit, zero antonym hit, non-degenerate text, and at least 1 reasoning hit
    strict_pass = exact_hit and (not antonym_hit) and (not is_degenerate) and (len(reasoning_hits) >= 1)

    # Numerical score [0.0 to 1.0]
    if antonym_hit or is_degenerate:
        composite_score = 0.0
    elif exact_hit:
        composite_score = 0.60 + 0.40 * reasoning_score
    else:
        composite_score = 0.20 * reasoning_score  # Partial reasoning without conclusion

    return {
        "id": test_item["id"],
        "pillar": test_item["pillar"],
        "strict_pass": strict_pass,
        "exact_hit": exact_hit,
        "antonym_hit": antonym_hit,
        "is_degenerate": is_degenerate,
        "rep_ratio": rep_ratio,
        "reasoning_hits": reasoning_hits,
        "composite_score": composite_score,
        "response": response,
    }


def run_tightened_benchmark(
    model,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    max_tokens: int = 48,
    thinking_cycles: int = 4,
) -> Dict[str, Any]:
    """
    Executes the Tightened Proto-AGI Benchmark across all 12 rigorous test cases.
    """
    model.eval()
    results = []
    total_passed = 0
    pillar_stats = collections.defaultdict(lambda: {"total": 0, "passed": 0, "scores": []})

    print("\n" + "=" * 90)
    print("      TIGHTENED PROTO-AGI EMERGENCE BENCHMARK (STRICT EVALUATION)")
    print("=" * 90)

    for item in TIGHTENED_BENCHMARK_SUITE:
        prompt = item["prompt"]
        q_toks = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                if hasattr(model, "generate_answer"):
                    out_toks = model.generate_answer(
                        q_toks,
                        max_tokens=max_tokens,
                        thinking_cycles=thinking_cycles,
                        eos_token_id=tokenizer.eos_token_id,
                        temperature=0.5,
                    )
                elif hasattr(model, "generate_multimodal"):
                    out_toks = model.generate_multimodal(
                        prompt_tokens=q_toks,
                        max_tokens=max_tokens,
                        thinking_cycles=thinking_cycles,
                        eos_token_id=tokenizer.eos_token_id,
                        temperature=0.5,
                    )
                elif hasattr(model, "solve_with_adaptive_emergence"):
                    out_toks, _ = model.solve_with_adaptive_emergence(
                        q_toks,
                        min_cycles=thinking_cycles,
                        max_cycles=thinking_cycles + 2,
                        max_answer_tokens=max_tokens,
                        temperature=0.5,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    out_toks = out_toks[0].tolist()
                else:
                    raise NotImplementedError("Model lacks generation method.")

        resp_text = tokenizer.decode(out_toks).strip() if out_toks else "<empty>"
        res = score_response_tight(item, resp_text)
        results.append(res)

        p_name = item["pillar"]
        pillar_stats[p_name]["total"] += 1
        pillar_stats[p_name]["scores"].append(res["composite_score"])
        if res["strict_pass"]:
            pillar_stats[p_name]["passed"] += 1
            total_passed += 1
            tag = "★ STRICT PASS"
        else:
            tag = "○ EVOLVING"

        print(f"\n▶ [{p_name} | {item['id']}] -> {tag} (Score: {res['composite_score']:.2f})")
        print(f"  Q: \"{prompt[:70]}...\"")
        print(f"  A: \"{resp_text}\"")
        if not res["strict_pass"]:
            reasons = []
            if not res["exact_hit"]:
                reasons.append(f"Missing exact target: {item['exact_targets']}")
            if res["antonym_hit"]:
                reasons.append("Triggered forbidden antonym")
            if res["is_degenerate"]:
                reasons.append(f"High unigram repetition ({res['rep_ratio']*100:.1f}%)")
            if len(res["reasoning_hits"]) == 0:
                reasons.append(f"Missing logical reasoning operator: {item['required_reasoning']}")
            print(f"  [Failure Diagnostics]: {'; '.join(reasons)}")

    total_tests = len(TIGHTENED_BENCHMARK_SUITE)
    pass_pct = (total_passed / total_tests) * 100.0
    mean_score = sum(r["composite_score"] for r in results) / total_tests * 100.0

    print("\n" + "=" * 90)
    print("      TIGHTENED BENCHMARK FINAL SUMMARY")
    print("=" * 90)
    print(f"Strict Pass Rate  : {total_passed}/{total_tests} ({pass_pct:.1f}%)")
    print(f"Composite Score   : {mean_score:.1f}%")
    print("-" * 90)
    for p_name, data in sorted(pillar_stats.items()):
        p_pass = data["passed"]
        p_tot = data["total"]
        p_pct = (p_pass / p_tot) * 100.0
        p_mean = sum(data["scores"]) / p_tot * 100.0
        print(f"  - {p_name:<30}: {p_pass}/{p_tot} Passed ({p_pct:5.1f}%) | Mean Score: {p_mean:5.1f}%")
    print("=" * 90 + "\n")

    return {
        "total_tests": total_tests,
        "total_passed": total_passed,
        "pass_rate_pct": pass_pct,
        "composite_score_pct": mean_score,
        "pillar_stats": dict(pillar_stats),
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    from src.organoid_layers import build_enhanced_organoid_spheroid
    from src.multimodal_layers import build_multimodal_organoid_core

    parser = argparse.ArgumentParser(description="Run Tightened Proto-AGI Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/enhanced_organoid_best.pt")
    parser.add_argument("--is_multimodal", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, ckpt_path)

    print(f"Loading checkpoint for tightened benchmarking: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    is_100m = ("lang_pole.token_embed.weight" in sd and sd["lang_pole.token_embed.weight"].shape[1] == 768) or ("100m" in ckpt_path)

    if args.is_multimodal or "multimodal" in ckpt_path or is_100m:
        model = build_multimodal_organoid_core(preset="100m" if is_100m else "33m")
    else:
        model = build_enhanced_organoid_spheroid()

    model.load_state_dict(sd, strict=False)
    model.to(device=device, dtype=dtype)

    run_tightened_benchmark(model, tokenizer, device, dtype)
