"""
Tightened Curriculum Generator with Answer Anchors
===================================================
Produces high-density deductive trajectories with explicit target anchors.
Eliminates verbose fluff and grounds the 3D organoid manifold on exact answers:
1. Exact Categorical Syllogisms (Yes/Mammal, Yes/Noble Gas, Yes/Carnivore)
2. Exact Multi-Hop Transitivity (Platinum densest, City A north of C)
3. Exact Counterfactual Mechanics (Hot air denser -> sinks downward, cold-to-hot -> freezes)
4. Exact Multi-Step Arithmetic (Change = $26, Distance = 420 miles, 4 faces)
5. Exact Biochemical Laws (Km increases, Vmax unchanged, Proton-motive force)
6. Exact ARC-AGI Spatial Reflections (Horizontal flip preserves vertical column)
"""

import random
import torch
import math
from typing import Tuple, List, Dict, Generator


TIGHT_TEMPLATES = [
    # 1. Deductive Syllogisms
    {
        "q": "All quadrupedal mammals produce milk for their young. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal? Answer yes or no and state why.",
        "a": "Yes, because it nurses its offspring with milk, a platypus is classified as a mammal.",
    },
    {
        "q": "All elements with full outer electron shells are noble gases. Argon has 8 valence electrons completely filling its outer shell. Is Argon a noble gas? Answer yes or no and state why.",
        "a": "Yes, because its outer shell is completely filled with 8 valence electrons, Argon is a noble gas.",
    },
    {
        "q": "All reptiles are cold-blooded ectotherms. A chameleon cannot internally regulate its body temperature and relies on ambient heat. Is a chameleon a reptile? Answer yes or no.",
        "a": "Yes, because it relies on ambient heat as a cold-blooded ectotherm, a chameleon is a reptile.",
    },

    # 2. Transitive Ordering
    {
        "q": "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?",
        "a": "Tracing the density chain: Platinum > Iron > Aluminium > Lithium. Therefore, Platinum is the densest metal.",
    },
    {
        "q": "City A is north of City B. City B is north of City C. Is City A north of City C? Answer yes or no and explain.",
        "a": "Yes, by transitive spatial relation, City A is north of City C.",
    },
    {
        "q": "Diamond is harder than Quartz. Corundum is harder than Quartz but softer than Diamond. Talc is softer than Quartz. Which mineral is the hardest?",
        "a": "Tracing the hardness chain: Diamond > Corundum > Quartz > Talc. Therefore, Diamond is the hardest mineral.",
    },

    # 3. Counterfactual Physics
    {
        "q": "If hot air were denser than cold air, in what direction would hot smoke from a chimney move? State the direction clearly.",
        "a": "Because denser fluids sink under gravity, hot smoke would move downward toward the ground.",
    },
    {
        "q": "If thermal energy flowed spontaneously from cold objects to hot objects, what would happen to an ice cube placed in a cup of hot tea?",
        "a": "Thermal energy would leave the ice cube and enter the tea, causing the ice cube to become colder and freeze further.",
    },
    {
        "q": "If gravity repelled mass instead of attracting it, what would happen to an apple released from a tree branch?",
        "a": "Because repulsive gravity pushes mass away from Earth, the apple would accelerate upward into the sky.",
    },

    # 4. Multi-Step Arithmetic
    {
        "q": "Maya buys 6 notebooks priced at $4 each. Maya pays with a $50 bill. Exactly how much change does Maya receive?",
        "a": "6 notebooks at $4 each cost 6 * $4 = $24. Subtracting $24 from $50 gives $50 - $24 = $26. Maya receives $26 in change.",
    },
    {
        "q": "A car travels 180 miles in 3 hours. How far does the car travel in 7 hours at that exact same constant speed?",
        "a": "Speed equals distance divided by time: 180 / 3 = 60 miles per hour. In 7 hours, the car travels 60 * 7 = 420 miles. The final distance is 420 miles.",
    },
    {
        "q": "A baker prepares 80 cookies. The baker sells 35 in the morning and 25 in the afternoon. Exactly how many cookies remain?",
        "a": "Total sales are 35 + 25 = 60 cookies. Subtracting from the batch gives 80 - 60 = 20 cookies. Exactly 20 cookies remain.",
    },

    # 5. Biochemical Kinetics
    {
        "q": "How does a competitive enzyme inhibitor affect the apparent Vmax and Km of an enzyme reaction?",
        "a": "A competitive inhibitor competes with substrate for the active site. It increases Km while leaving Vmax unchanged.",
    },
    {
        "q": "What physical electrochemical gradient directly powers ATP synthesis during mitochondrial oxidative phosphorylation?",
        "a": "ATP synthesis is driven by the proton-motive force, which is an electrochemical proton gradient across the inner mitochondrial membrane.",
    },
    {
        "q": "How does increasing temperature affect an exothermic reversible reaction at chemical equilibrium?",
        "a": "By Le Chatelier's principle, adding heat shifts the equilibrium in the endothermic reverse direction toward reactants.",
    },

    # 6. Spatial ARC-AGI
    {
        "q": "A geometric cube has 6 faces. If you paint exactly 2 opposite faces red and all the remaining faces blue, exactly how many blue faces are there?",
        "a": "A cube has 6 faces in total. Subtracting the 2 red faces gives 6 - 2 = 4. There are exactly 4 blue faces.",
    },
    {
        "q": "A 3x3 matrix has rows ['. # .', '. # .', '. # .'] with a vertical line in column 2. What happens to this vertical line when the matrix is reflected horizontally?",
        "a": "Horizontal reflection flips rows vertically. Because column 2 has '#' in every row, the vertical line remains in column 2 and is unchanged.",
    },
    {
        "q": "A 3x3 matrix has a single active pixel at row 1, column 1: ['# . .', '. . .', '. . .']. Where is the pixel after a horizontal reflection?",
        "a": "Horizontal reflection inverts rows, moving row 1 to row 3 while keeping columns identical. The pixel moves to row 3, column 1.",
    },
]


def gen_dynamic_tight_math() -> Tuple[str, str]:
    """Generates parameterized exact arithmetic problems with explicit targets."""
    names = ["Liam", "Sophia", "Noah", "Olivia", "Ethan", "Emma", "Lucas", "Maya"]
    items = ["markers", "pencils", "stickers", "cards", "cupcakes", "tickets", "notebooks"]
    name = random.choice(names)
    item = random.choice(items)

    count = random.randint(3, 9)
    price = random.randint(3, 8)
    subtotal = count * price
    bill = random.choice([50, 100])
    while bill <= subtotal:
        bill += 50
    change = bill - subtotal

    q = f"{name} buys {count} {item} priced at ${price} each. {name} pays with a ${bill} bill. Exactly how much change does {name} receive?"
    a = f"{count} {item} at ${price} each cost {count} * ${price} = ${subtotal}. Subtracting ${subtotal} from ${bill} leaves ${bill} - ${subtotal} = ${change}. {name} receives ${change} in change."
    return q, a


def gen_dynamic_tight_rate() -> Tuple[str, str]:
    """Generates parameterized speed/distance problems with explicit targets."""
    vehicles = ["train", "car", "bus", "cyclist"]
    vehicle = random.choice(vehicles)
    speeds = [40, 50, 60, 70, 80]
    speed = random.choice(speeds)
    h1 = random.choice([2, 3, 4])
    d1 = speed * h1
    h2 = random.choice([5, 6, 7, 8])
    d2 = speed * h2

    q = f"A {vehicle} travels {d1} miles in {h1} hours. How far does the {vehicle} travel in {h2} hours at that exact same constant speed?"
    a = f"Speed is distance divided by time: {d1} / {h1} = {speed} miles per hour. In {h2} hours, the {vehicle} travels {speed} * {h2} = {d2} miles. The final distance is {d2} miles."
    return q, a


def gen_dynamic_tight_cube() -> Tuple[str, str]:
    """Generates parameterized geometric face subtraction."""
    total_faces = 6
    painted = random.choice([1, 2, 3, 4])
    rem = total_faces - painted
    c1 = random.choice(["red", "green", "black"])
    c2 = random.choice(["blue", "white", "yellow"])

    q = f"A geometric cube has 6 faces. If you paint exactly {painted} faces {c1} and the remaining faces {c2}, exactly how many {c2} faces are there?"
    a = f"A cube has 6 faces in total. Subtracting the {painted} {c1} faces leaves 6 - {painted} = {rem}. There are exactly {rem} {c2} faces."
    return q, a


def gen_dynamic_tight_syllogism() -> Tuple[str, str]:
    """Generates deductive syllogisms with clear categorical affirmation."""
    cases = [
        ("quadrupedal mammals", "produce milk for their young", "platypus", "lays eggs but nurses its young with milk", "mammal"),
        ("quadrupedal mammals", "produce milk for their young", "bat", "flies with membranous wings and nurses its pups with milk", "mammal"),
        ("quadrupedal mammals", "produce milk for their young", "echidna", "lays leathery eggs and secretes milk from mammary glands", "mammal"),
        ("elements with full outer electron shells", "are noble gases", "Argon", "has 8 valence electrons completely filling its outer shell", "noble gas"),
        ("elements with full outer electron shells", "are noble gases", "Neon", "has a complete stable octet of 8 valence electrons", "noble gas"),
        ("elements with full outer electron shells", "are noble gases", "Helium", "has 2 valence electrons filling its 1s shell", "noble gas"),
        ("carnivorous plants", "capture and digest insect prey", "Venus flytrap", "snaps closed on flies and absorbs nitrogenous nutrients", "carnivorous plant"),
        ("superconductors", "expel internal magnetic fields via the Meissner effect", "YBaCuO ceramic", "levitates magnets by expelling magnetic flux below 93 Kelvin", "superconductor"),
    ]
    rule_subj, rule_prop, entity, entity_fact, category = random.choice(cases)
    q = f"All {rule_subj} {rule_prop}. An {entity} {entity_fact}. Is {entity} a {category}? Answer yes or no and state why."
    a = f"Yes, because {entity} {entity_fact}, {entity} is classified as a {category}."
    return q, a


def gen_dynamic_tight_transitivity() -> Tuple[str, str]:
    """Generates multi-hop transitive relation chains with strict connective logic."""
    types = [
        ("density", [("Platinum", 21.45), ("Gold", 19.3), ("Iron", 7.87), ("Aluminium", 2.70), ("Lithium", 0.53)], "densest"),
        ("hardness", [("Diamond", 10), ("Corundum", 9), ("Topaz", 8), ("Quartz", 7), ("Talc", 1)], "hardest"),
        ("speed", [("Peregrine falcon", 389), ("Cheetah", 120), ("Horse", 70), ("Elephant", 40), ("Snail", 0.05)], "fastest"),
        ("elevation", [("Everest", 8848), ("K2", 8611), ("Kangchenjunga", 8586), ("Lhotse", 8516), ("Makalu", 8485)], "highest"),
    ]
    prop_name, items, superlative = random.choice(types)
    a_name, b_name, c_name, d_name = [x[0] for x in items[:4]]
    q = f"{b_name} has more {prop_name} than {c_name}. {a_name} has more {prop_name} than {b_name}. {d_name} has less {prop_name} than {c_name}. Which of these four is the {superlative}?"
    a = f"Tracing the {prop_name} chain: {a_name} > {b_name} > {c_name} > {d_name}. Therefore, {a_name} is the {superlative}."
    return q, a


def gen_dynamic_tight_counterfactual() -> Tuple[str, str]:
    """Generates counterfactual physical scenarios requiring inverted causality."""
    cases = [
        ("hot air were denser than cold air", "what direction would hot smoke from a chimney move", "Because denser fluids sink under gravity, hot smoke would move downward toward the ground."),
        ("thermal energy flowed spontaneously from cold objects to hot objects", "what would happen to an ice cube placed in a cup of hot tea", "Thermal energy would leave the ice cube and enter the tea, causing the ice cube to become colder and freeze further."),
        ("gravity repelled mass instead of attracting it", "what would happen to an apple released from a tree branch", "Because repulsive gravity pushes mass away from Earth, the apple would accelerate upward into the sky."),
        ("friction vanished completely between surfaces", "what would happen to a rolling ball on a level floor", "Without friction or drag to dissipate kinetic energy, the ball would roll forever at constant velocity."),
    ]
    premise, query, answer = random.choice(cases)
    q = f"If {premise}, {query}? State the outcome clearly."
    return q, answer


def gen_dynamic_tight_kinetics() -> Tuple[str, str]:
    """Generates biochemical and physical rate law questions."""
    cases = [
        ("How does a competitive enzyme inhibitor affect the apparent Vmax and Km of an enzyme reaction?",
         "A competitive inhibitor competes with substrate for the active site. It increases Km while leaving Vmax unchanged."),
        ("How does a non-competitive allosteric inhibitor affect the apparent Vmax and Km of an enzyme reaction?",
         "A non-competitive inhibitor binds an allosteric site to reduce turnover. It decreases Vmax while leaving Km unchanged."),
        ("What physical electrochemical gradient directly powers ATP synthesis during mitochondrial oxidative phosphorylation?",
         "ATP synthesis is driven by the proton-motive force, which is an electrochemical proton gradient across the inner mitochondrial membrane."),
        ("How does increasing temperature affect an exothermic reversible reaction at chemical equilibrium?",
         "By Le Chatelier's principle, adding heat shifts the equilibrium in the endothermic reverse direction toward reactants."),
    ]
    return random.choice(cases)


def gen_dynamic_tight_arc() -> Tuple[str, str]:
    """Generates geometric and matrix ARC-AGI transformations."""
    cases = [
        ("A 3x3 matrix has rows ['. # .', '. # .', '. # .'] with a vertical line in column 2. What happens to this vertical line when the matrix is reflected horizontally?",
         "Horizontal reflection flips rows vertically. Because column 2 has '#' in every row, the vertical line remains in column 2 and is unchanged."),
        ("A 3x3 matrix has a single active pixel at row 1, column 1: ['# . .', '. . .', '. . .']. Where is the pixel after a horizontal reflection?",
         "Horizontal reflection inverts rows, moving row 1 to row 3 while keeping columns identical. The pixel moves to row 3, column 1."),
    ]
    return random.choice(cases)


def generate_tightened_sample() -> Tuple[str, str]:
    """Picks across all 6 rigorous proto-AGI pillars."""
    generators = [
        gen_dynamic_tight_syllogism,
        gen_dynamic_tight_transitivity,
        gen_dynamic_tight_counterfactual,
        gen_dynamic_tight_math,
        gen_dynamic_tight_rate,
        gen_dynamic_tight_cube,
        gen_dynamic_tight_kinetics,
        gen_dynamic_tight_arc,
    ]
    if random.random() < 0.25:
        item = random.choice(TIGHT_TEMPLATES)
        return item["q"], item["a"]
    else:
        gen = random.choice(generators)
        return gen()


def get_tightened_stream(
    tokenizer,
    batch_size: int = 4,
    max_prompt_len: int = 256,
    max_resp_len: int = 64,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Streams tightened, answer-anchored reasoning samples.
    """
    batch_p, batch_t = [], []

    while True:
        q, a = generate_tightened_sample()
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
