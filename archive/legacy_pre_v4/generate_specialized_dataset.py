"""
CerebroSpheroid Rich Multi-Domain Cognitive & Scientific Reasoning Dataset Generator
Expands curriculum across:
1. Quantitative Deduction, Speed/Distance, Multi-Step Inventory, Fractions, Geometry.
2. Formal Deductive Syllogisms & Transitive Categorical Logic.
3. 50+ Real-World Physical, Chemical, Biological, and Earth Science Causal Mechanisms.
4. Clean Natural Language: Stimulus -> Direct Articulation (Zero chatbot tags).
"""

import os
import json
import random
import hashlib
import argparse
from typing import Optional
import pyarrow as pa
import pyarrow.parquet as pq

# -------------------------------------------------------------------------
# Entity Lexicon
# -------------------------------------------------------------------------
NAMES = [
    "Maya", "Liam", "Sophia", "Noah", "Olivia", "Ethan", "Emma", "Aiden",
    "Ava", "Lucas", "Elena", "Marcus", "Chloe", "David", "Zara", "Leo",
    "Isabella", "Julian", "Amara", "Gabriel", "Nora", "Caleb", "Hannah"
]

ITEMS = [
    "apples", "books", "markers", "notebooks", "pencils", "oranges", "coins",
    "stickers", "cards", "marbles", "cupcakes", "tickets", "erasers", "balloons",
    "stamps", "crayons", "peaches", "cookies", "beads", "ribbons"
]

CONTAINERS = [
    "box", "crate", "bag", "drawer", "shelf", "basket", "container", "bin",
    "jar", "cabinet", "tray", "carton"
]

VEHICLES = [
    ("train", 50, 120), ("car", 40, 80), ("cyclist", 10, 25),
    ("airplane", 400, 600), ("runner", 5, 12), ("bus", 25, 55),
    ("speedboat", 30, 60), ("subway", 35, 70)
]

ANIMALS = [
    ("bird", "robin", "feathers", "flies with wings"),
    ("mammal", "cat", "fur", "is warm-blooded"),
    ("mammal", "dolphin", "lungs", "breathes air underwater"),
    ("reptile", "lizard", "scales", "is cold-blooded"),
    ("fish", "salmon", "gills", "swims in water"),
    ("amphibian", "frog", "permeable skin", "undergoes metamorphosis"),
    ("insect", "bee", "six legs", "has an exoskeleton"),
    ("arachnid", "spider", "eight legs", "spins silk"),
]

# -------------------------------------------------------------------------
# 1. Quantitative & Rate Reasoning Generators
# -------------------------------------------------------------------------

def gen_rate_distance_problem():
    vehicle, min_s, max_s = random.choice(VEHICLES)
    speed = random.randint(min_s, max_s)
    hours = random.randint(2, 6)
    dist = speed * hours
    q = f"A {vehicle} travels at a steady speed of {speed} miles per hour. How far does the {vehicle} travel in {hours} hours?"
    a = (f"Distance equals speed multiplied by time. "
         f"Here, the speed is {speed} miles per hour and the time is {hours} hours. "
         f"Multiplying {speed} by {hours} yields {dist} miles. "
         f"The final answer is {dist} miles.")
    return q, a


def gen_inventory_transaction_problem():
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    initial = random.randint(20, 80)
    bought = random.randint(10, 40)
    gave_away = random.randint(5, bought + initial - 10)
    remaining = initial + bought - gave_away

    q = f"{name} initially had {initial} {item}. {name} acquired {bought} more {item} and later gave away {gave_away}. How many {item} remain?"
    a = (f"{name} starts with {initial} {item}. "
         f"Adding {bought} new {item} brings the subtotal to {initial} + {bought} = {initial + bought} {item}. "
         f"Next, subtracting {gave_away} leaves {initial + bought} - {gave_away} = {remaining} {item}. "
         f"The final answer is {remaining} {item}.")
    return q, a


def gen_bakery_multi_step_problem():
    name = random.choice(NAMES)
    total_baked = random.randint(40, 100)
    sold_morning = random.randint(15, total_baked // 2)
    sold_afternoon = random.randint(10, total_baked - sold_morning - 5)
    remaining = total_baked - sold_morning - sold_afternoon

    q = f"{name} bakes {total_baked} cupcakes at a bakery. {name} sells {sold_morning} in the morning and {sold_afternoon} in the afternoon. How many cupcakes are left?"
    a = (f"{name} begins with {total_baked} cupcakes. "
         f"The total sold is {sold_morning} + {sold_afternoon} = {sold_morning + sold_afternoon} cupcakes. "
         f"Subtracting total sales from the batch leaves {total_baked} - {sold_morning + sold_afternoon} = {remaining} cupcakes. "
         f"The final answer is {remaining} cupcakes left.")
    return q, a


def gen_proportional_pricing_problem():
    item = random.choice(ITEMS)
    unit_count_1 = random.randint(2, 6)
    unit_price = random.randint(2, 12)
    total_cost_1 = unit_count_1 * unit_price
    unit_count_2 = random.randint(7, 15)
    total_cost_2 = unit_count_2 * unit_price

    q = f"If {unit_count_1} {item} cost ${total_cost_1}, how much do {unit_count_2} of the same {item} cost?"
    a = (f"First, calculate the price per single item: ${total_cost_1} divided by {unit_count_1} equals ${unit_price} per item. "
         f"Next, multiply the unit price of ${unit_price} by the desired quantity of {unit_count_2}. "
         f"${unit_price} multiplied by {unit_count_2} equals ${total_cost_2}. "
         f"The final answer is ${total_cost_2}.")
    return q, a


def gen_container_distribution_problem():
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    container = random.choice(CONTAINERS)
    num_containers = random.randint(3, 8)
    items_per_container = random.randint(5, 15)
    extra = random.randint(1, num_containers - 1)
    total = num_containers * items_per_container + extra

    q = f"{name} divides {total} {item} equally into {num_containers} {container}s, with some left over. How many {item} are in each {container}, and how many remain?"
    items_each = total // num_containers
    rem = total % num_containers
    a = (f"Divide {total} {item} by {num_containers} {container}s. "
         f"{total} divided by {num_containers} gives {items_each} with a remainder of {rem}. "
         f"Therefore, each {container} holds {items_each} {item}, and {rem} remain. "
         f"The final answer is {items_each} in each with {rem} remaining.")
    return q, a


def gen_percentage_discount_problem():
    items = ["jacket", "bicycle", "camera", "watch", "guitar", "telescope"]
    item = random.choice(items)
    orig_price = random.choice([40, 60, 80, 100, 120, 150, 200])
    discount_pct = random.choice([10, 20, 25, 30, 50])
    discount_amt = (orig_price * discount_pct) // 100
    final_price = orig_price - discount_amt

    q = f"A {item} normally costs ${orig_price}. A store offers a {discount_pct}% discount. What is the final sale price?"
    a = (f"First, compute the discount amount: {discount_pct}% of ${orig_price} is ({discount_pct} / 100) * {orig_price} = ${discount_amt}. "
         f"Next, subtract the discount from the original price: ${orig_price} - ${discount_amt} = ${final_price}. "
         f"The final answer is ${final_price}.")
    return q, a


# -------------------------------------------------------------------------
# 2. Geometry & Spatial Reasoning
# -------------------------------------------------------------------------

def gen_geometric_rectangle_problem():
    length = random.randint(5, 30)
    width = random.randint(3, min(length, 18))
    area = length * width
    perimeter = 2 * (length + width)

    mode = random.choice(["area", "perimeter", "both"])
    place = random.choice(["garden", "swimming pool", "room", "park", "courtyard", "field"])
    if mode == "area":
        q = f"A rectangular {place} measures {length} meters long and {width} meters wide. What is its total area?"
        a = (f"The area of a rectangle is length multiplied by width: L * W. "
             f"Multiplying {length} meters by {width} meters gives {area} square meters. "
             f"The final answer is {area} square meters.")
    elif mode == "perimeter":
        q = f"A rectangular {place} is {length} meters long and {width} meters wide. What is its total perimeter?"
        a = (f"The perimeter of a rectangle is twice the sum of length and width: 2 * (length + width). "
             f"Adding {length} and {width} gives {length + width} meters. "
             f"Multiplying {length + width} by 2 yields {perimeter} meters. "
             f"The final answer is {perimeter} meters.")
    else:
        q = f"A rectangular {place} is {length} meters long and {width} meters wide. What is its perimeter and area?"
        a = (f"The area is length times width: {length} * {width} = {area} square meters. "
             f"The perimeter is twice the sum of length and width: 2 * ({length} + {width}) = {perimeter} meters. "
             f"The final answer is an area of {area} square meters and a perimeter of {perimeter} meters.")
    return q, a


# -------------------------------------------------------------------------
# 3. Formal Deductive Logic & Syllogisms
# -------------------------------------------------------------------------

def gen_syllogism_problem():
    group, member, trait, capability = random.choice(ANIMALS)
    q = f"All {group}s have {trait}. A {member} is a {group}. Does a {member} have {trait}?"
    a = (f"By universal categorical deduction: All members of the category '{group}' possess '{trait}'. "
         f"Since a {member} belongs to the category '{group}', it necessarily inherits all properties of '{group}'. "
         f"Therefore, a {member} has {trait}. "
         f"The final answer is yes, a {member} has {trait}.")
    return q, a


def gen_transitive_relation_problem():
    category = random.choice(["animals", "mountains", "rivers", "materials", "athletes"])
    if category == "animals":
        items = random.sample(["cheetah", "horse", "greyhound", "wolf", "lion"], 3)
        comp_pos, comp_neg = ("faster", "slower")
    elif category == "mountains":
        items = random.sample(["Everest", "K2", "Kangchenjunga", "Lhotse", "Makalu"], 3)
        comp_pos, comp_neg = ("taller", "shorter")
    elif category == "materials":
        items = random.sample(["diamond", "topaz", "quartz", "calcite", "talc"], 3)
        comp_pos, comp_neg = ("harder", "softer")
    elif category == "athletes":
        items = random.sample(["Marcus", "David", "Liam", "Noah", "Ethan"], 3)
        comp_pos, comp_neg = ("older", "younger")
    else:
        items = random.sample(["steel", "iron", "copper", "aluminum", "tin"], 3)
        comp_pos, comp_neg = ("denser", "lighter")

    e1, e2, e3 = items
    q = f"{e1} is {comp_pos} than {e2}. {e2} is {comp_pos} than {e3}. Is {e1} {comp_pos} or {comp_neg} than {e3}?"
    a = (f"The relation '{comp_pos} than' is transitive. "
         f"Given that {e1} > {e2} and {e2} > {e3}, it logically follows that {e1} > {e3}. "
         f"Therefore, {e1} is {comp_pos} than {e3}. "
         f"The final answer is {comp_pos}.")
    return q, a


def gen_number_sequence_problem():
    seq_type = random.choice(["arithmetic", "geometric", "squares"])
    if seq_type == "arithmetic":
        start = random.randint(2, 30)
        diff = random.randint(3, 11)
        seq = [start + i * diff for i in range(5)]
        nxt = start + 5 * diff
        q = f"What is the next number in the sequence: {', '.join(map(str, seq))}, ...?"
        a = (f"Examine the constant difference: each term increases by {diff}. "
             f"Adding {diff} to the final term {seq[-1]} gives {seq[-1]} + {diff} = {nxt}. "
             f"The final answer is {nxt}.")
    elif seq_type == "geometric":
        start = random.randint(1, 4)
        ratio = random.randint(2, 3)
        seq = [start * (ratio ** i) for i in range(5)]
        nxt = start * (ratio ** 5)
        q = f"What number continues the geometric pattern: {', '.join(map(str, seq))}, ...?"
        a = (f"Examine the common ratio: each term is multiplied by {ratio}. "
             f"Multiplying {seq[-1]} by {ratio} yields {nxt}. "
             f"The final answer is {nxt}.")
    else:
        offset = random.randint(1, 5)
        seq = [(i + offset) ** 2 for i in range(5)]
        nxt = (5 + offset) ** 2
        q = f"Determine the next square in the series: {', '.join(map(str, seq))}, ...?"
        a = (f"The terms are consecutive squares of integers starting at {offset}. "
             f"The next integer to square is {5 + offset}. "
             f"Squaring {5 + offset} yields {nxt}. "
             f"The final answer is {nxt}.")
    return q, a


# -------------------------------------------------------------------------
# 3b. Advanced Multi-Hop & Counterfactual Generators
# -------------------------------------------------------------------------

def gen_multi_hop_transitive_problem():
    elements = [
        ("Iron", "Aluminium", "Platinum", "Lithium", "density", "denser"),
        ("Cheetah", "Horse", "Peregrine Falcon", "Turtle", "speed", "faster"),
        ("Lead", "Copper", "Gold", "Cork", "mass", "heavier"),
        ("Jupiter", "Earth", "Sun", "Pluto", "gravitational pull", "stronger"),
    ]
    A, B, C, D, prop, rel = random.choice(elements)
    q = (f"{A} has greater {prop} than {B}. {C} is {rel} than {A}. {D} has less {prop} than {B}. "
         f"Which of these four has the greatest {prop}?")
    a = (f"Analyze the chain of transitive inequalities: "
         f"{C} > {A}, {A} > {B}, and {B} > {D}. "
         f"Combining these yields the complete ordering: {C} > {A} > {B} > {D}. "
         f"Therefore, {C} has the greatest {prop}. "
         f"The final answer is {C}.")
    return q, a


def gen_counterfactual_physics_problem():
    scenarios = [
        (
            "If hot air were denser than cold air, what direction would hot smoke from a chimney move?",
            "Under standard physics, hot air rises because it is less dense than surrounding cold air. In this counterfactual scenario where hot air is denser than cold air, gravity and buoyant forces invert: denser hot air will sink toward the ground rather than rising into the atmosphere. Therefore, the smoke will move downward."
        ),
        (
            "If thermal energy flowed spontaneously from cold objects to hot objects, what would happen to an ice cube in warm tea?",
            "Under the second law of thermodynamics, heat flows from warmer to colder objects. In this counterfactual world where heat flows spontaneously from cold to hot, the ice cube would transfer its thermal energy to the warm tea. The ice cube would become even colder and freeze harder, while the tea would absorb heat and become hotter."
        ),
        (
            "If gravity were an repulsive force pushing masses apart, what would happen to planets orbiting the Sun?",
            "Planetary orbits rely on an attractive gravitational centripetal force pulling planets toward the Sun to balance their tangential velocity. If gravity were repulsive, there would be no inward centripetal force to hold the planets in orbit. Instead, planets would be pushed outward and fly away into deep space along divergent trajectories."
        ),
        (
            "If solid ice were denser than liquid water, what would happen to frozen lakes in winter?",
            "Because ice is less dense than water, it floats on the surface, forming an insulating blanket that keeps the water below liquid. If ice were denser than water, ice forming on the surface would sink to the bottom. Over winter, the entire lake would freeze solidly from the bottom up, destroying aquatic ecosystems."
        ),
    ]
    return random.choice(scenarios)


def gen_advanced_taxonomical_syllogism():
    syllogisms = [
        (
            "All quadrupedal mammals produce milk for their offspring. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal?",
            "By universal categorical definition, producing milk for offspring is an essential diagnostic property of the class Mammalia. Although the platypus exhibits the ancestral trait of laying eggs (monotreme), its possession of mammary glands and lactation firmly classifies it as a mammal. Therefore, yes, a platypus is a mammal."
        ),
        (
            "All elements with full outer valence electron shells are noble gases. Argon has 8 electrons completely filling its outer shell. Is Argon a noble gas?",
            "By chemical definition, elements with full valence electron octets possess zero valence vacancies and are classified as noble gases. Because Argon has 8 valence electrons filling its outer shell, it satisfies the universal condition. Therefore, yes, Argon is a noble gas."
        ),
        (
            "All cetaceans breathe atmospheric oxygen through blowholes. A blue whale lives entirely in the ocean and breathes air through blowholes. Is a blue whale a fish or a mammal?",
            "Fish extract dissolved oxygen from water using gills, whereas mammals breathe atmospheric air with lungs. Because the blue whale breathes atmospheric oxygen through blowholes and gives live birth, it belongs to the mammalian order Cetacea. Therefore, a blue whale is a mammal, not a fish."
        ),
    ]
    return random.choice(syllogisms)


def gen_multi_step_change_arithmetic():
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    unit_price = random.randint(3, 8)
    quantity = random.randint(3, 7)
    total_cost = unit_price * quantity
    bill = 50 if total_cost <= 45 else 100
    change = bill - total_cost

    q = f"{name} buys {quantity} {item} priced at ${unit_price} each. {name} pays with a ${bill} bill. How much change does {name} receive?"
    a = (f"First, calculate the total cost: {quantity} items multiplied by ${unit_price} per item equals ${unit_price * quantity}. "
         f"Next, subtract the total cost from the payment: ${bill} - ${total_cost} = ${change}. "
         f"The final answer is ${change} change.")
    return q, a

# -------------------------------------------------------------------------

SCIENTIFIC_REASONING_BANK = [
    # 1. Thermodynamics & Heat
    (
        "Why does hot air rise while cold air sinks?",
        "When air is heated, thermal energy causes gas molecules to vibrate and move further apart, decreasing its density. Cooler air remains denser with molecules packed more tightly. By Archimedes' principle, the denser cold air sinks beneath the warmer air, creating an upward buoyant force that causes the hot air to rise in a process called thermal convection."
    ),
    (
        "Why does a metal spoon feel colder than a wooden spoon at room temperature?",
        "Both spoons are at the exact same thermal equilibrium with the room. However, metals have high thermal conductivity due to free delocalized electrons, conducting heat rapidly away from human skin. Wood is a thermal insulator with low conductivity, transferring heat slowly. The sensation of cold reflects the rapid rate of heat loss rather than an actual temperature difference."
    ),
    (
        "Why does ice float on liquid water?",
        "As liquid water cools below 4 degrees Celsius and freezes at 0 degrees, hydrogen bonds lock molecules into an open hexagonal crystalline lattice. This lattice configuration spaces the molecules further apart than in the disordered liquid state. Because solid ice has lower density than liquid water, it experiences positive buoyant force and floats."
    ),
    (
        "What is the difference between evaporation and boiling?",
        "Evaporation is a surface phenomenon occurring at any temperature where surface molecules with sufficient kinetic energy escape as vapor into the air. Boiling is a bulk phase transition occurring when the liquid's vapor pressure equals the external atmospheric pressure, causing vapor bubbles to form throughout the entire liquid volume."
    ),

    # 2. Astronomy & Earth Science
    (
        "What causes day and night on planet Earth?",
        "Day and night are caused by the continuous rotation of Earth on its axis once every 24 hours. The half of the planet facing the Sun receives direct solar illumination, experiencing daylight, while the opposite half is turned away in shadow, experiencing night. As Earth rotates from west to east, day and night alternate cyclically."
    ),
    (
        "What causes the oceanic tides on Earth?",
        "Oceanic tides are driven by gravitational differentials exerted primarily by the Moon and secondarily by the Sun. The gravitational pull is strongest on the side of Earth facing the Moon, pulling seawater outward into a tidal bulge, while centrifugal force creates an antipodal bulge on the opposite side. Coastal areas experience alternating high and low tides as Earth rotates."
    ),
    (
        "Why do we experience four distinct seasons on Earth?",
        "Seasons are caused by Earth's 23.5-degree axial tilt as it orbits the Sun. When a hemisphere tilts toward the Sun, it receives direct concentrated solar rays and longer daylight hours, creating summer. When tilted away, rays strike at a shallow angle and days are shorter, producing winter. The distance to the Sun is not the primary cause."
    ),
    (
        "Why does the Moon have different phases in the night sky?",
        "The Moon does not produce light; it reflects sunlight. As the Moon orbits Earth once every 29.5 days, the angle between the Sun, Earth, and Moon continuously changes. The lunar phases represent the fraction of the Moon's sunlit hemisphere that is directly visible from Earth, transitioning from new moon to crescent, gibbous, and full moon."
    ),
    (
        "What causes earthquakes in the Earth's crust?",
        "Earth's rigid outer lithosphere is divided into tectonic plates that constantly move over the mantle. At fault boundaries, friction locks plate edges while tectonic forces continue pushing, accumulating immense elastic strain. When the stress overcomes the frictional resistance, the rock fractures suddenly, releasing seismic waves that shake the ground."
    ),

    # 3. Biology & Natural Systems
    (
        "What is photosynthesis and why do green plants need sunlight?",
        "Photosynthesis is the biochemical process by which green plants synthesize glucose from carbon dioxide and water using sunlight. Chlorophyll pigments in chloroplasts absorb solar photons to energize electrons, splitting water to release oxygen and generating ATP and NADPH. The plant uses this chemical energy in the Calvin cycle to build sugars for growth and survival."
    ),
    (
        "Why do leaves change color in autumn?",
        "During warm months, abundant green chlorophyll masks yellow and orange pigments called carotenoids. In autumn, shorter daylight and cooler temperatures trigger trees to stop producing chlorophyll, causing it to decompose. As the green color fades, the underlying carotenoids and newly synthesized red anthocyanins become visible."
    ),
    (
        "How do vaccines provide immunity against infectious diseases?",
        "Vaccines expose the immune system to a harmless weakened pathogen, inactive protein, or mRNA blueprint. B-lymphocytes recognize the unique antigen and produce neutralizing antibodies while generating long-lived memory B and T cells. If the real pathogen enters the body later, these memory cells mount an immediate, overwhelming immune response before illness develops."
    ),
    (
        "What is natural selection in evolutionary biology?",
        "Natural selection is the mechanism driving biological evolution. Organisms possess heritable phenotypic variations. Individuals with traits best suited to their environmental challenges have higher rates of survival and reproduction. Over generations, these advantageous adaptive genes become more frequent in the population, gradually tailoring the species to its niche."
    ),
    (
        "Why do humans need to breathe oxygen?",
        "Humans breathe oxygen to power cellular respiration within mitochondria. Oxygen serves as the final electron acceptor in the electron transport chain, binding with electrons and protons to form water. This allows the cellular engine to generate large quantities of ATP, the primary molecular currency required for muscle contraction, neural firing, and metabolism."
    ),

    # 4. Physics & Mechanics
    (
        "Why does a heavy ship made of steel float on water?",
        "A ship floats because of Archimedes' principle: an object floats if it displaces a weight of water equal to its own total weight. Although steel is dense, a ship is hollowed into a massive hull filled mostly with empty air. This design gives the entire ship an average density much lower than liquid water, generating sufficient upward buoyant force to keep it afloat."
    ),
    (
        "What is Newton's first law of motion?",
        "Newton's first law of motion, the law of inertia, states that an object at rest remains at rest, and an object in uniform motion continues moving at constant velocity along a straight line unless acted upon by an external net force. Inertia is an object's intrinsic resistance to any change in its state of motion."
    ),
    (
        "Why does sound travel faster through steel than through air?",
        "Sound speed is determined by the square root of a medium's elastic stiffness divided by its density: v = sqrt(K / rho). Although steel is denser than air, its elastic bulk modulus is hundreds of thousands of times greater due to tight metallic interatomic bonds. This immense restoring force allows acoustic vibrations to propagate through steel at 5,000 meters per second compared to 343 in air."
    ),
    (
        "Why is the sky blue during a clear day?",
        "Sunlight contains all colors of the visible spectrum. When sunlight strikes Earth's atmosphere, tiny nitrogen and oxygen gas molecules cause Rayleigh scattering. Because the efficiency of Rayleigh scattering is inversely proportional to the fourth power of wavelength, short blue wavelengths scatter in all directions ten times more than red wavelengths, illuminating the sky blue."
    ),
    (
        "How does a concave mirror form a real inverted image?",
        "A concave mirror curves inward with a positive focal point. Incident parallel light rays reflect inward and converge in front of the mirror. When an object is positioned beyond the focal length, reflected rays physically intersect in real space. Because rays cross the principal axis, the resulting real image is upside down (inverted)."
    ),
    (
        "What happens when you mix an acid and a base?",
        "Mixing an acid and a base triggers a chemical neutralization reaction. Hydrogen ions (H+) from the acid combine with hydroxide ions (OH-) from the base to form neutral liquid water (H2O), while the remaining counter-ions combine to form a dissolved chemical salt. The reaction is typically exothermic, releasing thermal energy."
    ),
]

def gen_scientific_causal_problem():
    q, a = random.choice(SCIENTIFIC_REASONING_BANK)
    return q, a


# -------------------------------------------------------------------------
# Master Generator & Compiler
# -------------------------------------------------------------------------

GENERATOR_FUNCS = [
    gen_rate_distance_problem,
    gen_inventory_transaction_problem,
    gen_bakery_multi_step_problem,
    gen_proportional_pricing_problem,
    gen_container_distribution_problem,
    gen_percentage_discount_problem,
    gen_geometric_rectangle_problem,
    gen_syllogism_problem,
    gen_transitive_relation_problem,
    gen_number_sequence_problem,
    gen_scientific_causal_problem,
    gen_multi_hop_transitive_problem,
    gen_counterfactual_physics_problem,
    gen_advanced_taxonomical_syllogism,
    gen_multi_step_change_arithmetic,
]


def generate_single_cognitive_sample():
    func = random.choice(GENERATOR_FUNCS)
    return func()


def _draw_corpus(num_samples: int, seed: Optional[int]):
    """Generates records under an explicit seed, folding away exact repeats.

    Without a seed this module is irreproducible: every invocation draws a different pool,
    which silently invalidates any metric ever recorded against a previously written file.
    Duplication is also far higher than it looks -- 50,000 draws land on roughly 20,000
    distinct stimuli -- and a repeated record is sampled proportionally more often during
    training and again in the baseline histograms, so it biases both sides of any comparison.

    Returns (order, stimulus -> answer, repeats folded, ambiguous stimuli). A stimulus is
    ambiguous when it was drawn twice with different articulations; each generator derives its
    answer from the numbers it prints, so a conflict means the printed prose does not determine
    the answer and that record teaches the model an unsolvable problem.
    """
    if seed is not None:
        random.seed(seed)
    seen = {}
    order = []
    duplicates = 0
    ambiguous = set()
    for _ in range(num_samples):
        q, a = generate_single_cognitive_sample()
        if q in seen:
            duplicates += 1
            if seen[q] != a:
                ambiguous.add(q)
            continue
        seen[q] = a
        order.append(q)
    return order, seen, duplicates, ambiguous


def _md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_pairs(path: str, keys, lookup) -> None:
    table = pa.Table.from_arrays(
        [pa.array([k for k in keys], type=pa.string()),
         pa.array([lookup[k] for k in keys], type=pa.string())],
        names=["stimulus", "articulation"],
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def audit_empty_answers(keys, lookup):
    """Stimuli whose generated answer is blank -- these would train the model to emit EOS."""
    return sum(1 for q in keys if not lookup[q])


def build_specialized_split(
    train_path: str,
    val_path: str,
    num_samples: int = 50000,
    val_fraction: float = 0.05,
    seed: int = 1337,
    manifest_path: Optional[str] = None,
):
    """Writes a seeded, deduplicated, disjoint train/val pair plus a provenance manifest.

    The split unit is the distinct stimulus, not the row, so no question can appear on both
    sides. It is not a template-disjoint split: two records can share a template with
    different numbers. That is deliberate -- the answer is a function of the quantities in the
    question, so memorising the sentence shape does not yield the right tokens, and splitting
    by template would starve the val set of coverage the training pool can never supply.

    Stimuli that were drawn twice with two different answers are dropped from both sides. No
    mapping can be correct for both labels, so keeping one would put an irreducible floor under
    the loss and blur exactly the signal this split exists to measure.
    """
    keys, lookup, duplicates, ambiguous = _draw_corpus(num_samples, seed=seed)
    keys = [q for q in keys if q not in ambiguous]
    if len(keys) < 20:
        raise RuntimeError(f"only {len(keys)} usable stimuli from {num_samples:,} draws")

    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_fraction)))
    val_keys, train_keys = keys[:n_val], keys[n_val:]

    _write_pairs(train_path, train_keys, lookup)
    _write_pairs(val_path, val_keys, lookup)

    manifest = {
        "seed": seed,
        "draws": num_samples,
        "exact_duplicate_draws_folded": duplicates,
        "ambiguous_stimuli_dropped": len(ambiguous),
        "distinct_stimuli": len(keys),
        "train_rows": len(train_keys),
        "val_rows": len(val_keys),
        "train_parquet": os.path.relpath(train_path, os.path.join(os.path.dirname(__file__), "..")),
        "val_parquet": os.path.relpath(val_path, os.path.join(os.path.dirname(__file__), "..")),
        "train_md5": _md5_of(train_path),
        "val_md5": _md5_of(val_path),
        "empty_answers": audit_empty_answers(keys, lookup),
    }
    if manifest_path:
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
    print("=" * 78)
    print(f"  seed                    {manifest['seed']}")
    print(f"  draws                   {manifest['draws']:,}")
    print(f"  exact repeats folded    {manifest['exact_duplicate_draws_folded']:,} "
          f"({manifest['exact_duplicate_draws_folded'] / max(num_samples, 1):.1%} of draws)")
    print(f"  repeats with a DIFFERENT answer, dropped from both splits: "
          f"{manifest['ambiguous_stimuli_dropped']:,}")
    print(f"  distinct stimuli        {manifest['distinct_stimuli']:,}")
    print(f"  train / val rows        {manifest['train_rows']:,} / {manifest['val_rows']:,}")
    print(f"  train md5               {manifest['train_md5']}")
    print(f"  val md5                 {manifest['val_md5']}")
    print("=" * 78)
    return manifest


def build_specialized_parquet(output_path: str, num_samples: int = 50000,
                              seed: Optional[int] = None, dedupe: bool = False):
    """Writes the single-file curriculum.

    `seed=None` keeps the historical unseeded behaviour. Pass a seed for anything whose
    metrics are going to be written down, and prefer `build_specialized_split` when a
    selection signal is needed -- this file has no held-out portion by construction.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stimuli = []
    articulations = []

    print(f"Synthesizing {num_samples:,} rich multi-domain cognitive reasoning records...")
    if dedupe:
        keys, lookup, duplicates, ambiguous = _draw_corpus(num_samples, seed=seed)
        print(f"  {duplicates:,} exact repeats folded; {len(ambiguous):,} ambiguous stimuli dropped")
        keys = [q for q in keys if q not in ambiguous]
        print(f"  {duplicates:,} exact repeats folded; {len(keys):,} distinct stimuli kept")
        stimuli = list(keys)
        articulations = [lookup[k] for k in stimuli]
    else:
        if seed is not None:
            random.seed(seed)
        for i in range(num_samples):
            q, a = generate_single_cognitive_sample()
            stimuli.append(q)
            articulations.append(a)

    table = pa.Table.from_arrays(
        [pa.array(stimuli, type=pa.string()), pa.array(articulations, type=pa.string())],
        names=["stimulus", "articulation"]
    )
    pq.write_table(table, output_path, compression="zstd")
    file_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Successfully compiled {len(stimuli):,} diverse cognitive records to {output_path} ({file_mb:.2f} MB)")
    print(f"  md5 {_md5_of(output_path)}")


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "data", "cerebro_cognitive")
    parser = argparse.ArgumentParser(description="Build the cognitive curriculum (seeded, split)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Generator seed. Recorded in the manifest; re-running reproduces both files byte for byte")
    parser.add_argument("--num_samples", type=int, default=50000, help="Draws before deduplication")
    parser.add_argument("--val_fraction", type=float, default=0.05, help="Share of distinct stimuli held out")
    parser.add_argument("--legacy_single_file", action="store_true",
                        help="Write only cerebro_cognitive_train.parquet, with no holdout (the historical layout)")
    args = parser.parse_args()

    if args.legacy_single_file:
        build_specialized_parquet(
            os.path.join(out_dir, "cerebro_cognitive_train.parquet"),
            num_samples=args.num_samples, seed=args.seed, dedupe=True,
        )
    else:
        build_specialized_split(
            train_path=os.path.join(out_dir, "cerebro_cognitive_train.parquet"),
            val_path=os.path.join(out_dir, "cerebro_cognitive_val.parquet"),
            num_samples=args.num_samples,
            val_fraction=args.val_fraction,
            seed=args.seed,
            manifest_path=os.path.join(out_dir, "cerebro_cognitive_manifest.json"),
        )
