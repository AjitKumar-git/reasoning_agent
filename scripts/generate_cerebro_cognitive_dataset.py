#!/usr/bin/env python3
"""
CerebroCognitive-10k Dataset Generator.
Builds a comprehensive GSM8K-league dataset (9,500 - 10,000 examples)
addressing core language model reasoning pitfalls:
  Track 1: Orthographic & Character Scratchpad Reasoning (2,000)
  Track 2: Temporal & Clock State-Machine Reasoning (2,500)
  Track 3: Counterfactual & Premise-Rejection / Anti-Sycophancy (2,500)
  Track 4: Multi-Step Elementary Arithmetic (GSM8k) (2,500)
"""

import os
import sys
import json
import random
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

random.seed(42)

# =====================================================================
# TRACK 1: ORTHOGRAPHIC & CHARACTER SCRATCHPAD REASONING (2,000)
# =====================================================================

VOCAB_WORDS = [
    "strawberry", "mississippi", "banana", "encyclopedia", "rhythm",
    "astronomy", "photosynthesis", "attractor", "spherical", "algorithm",
    "consciousness", "subterranean", "parallel", "bookkeeper", "committee",
    "hippopotamus", "kaleidoscope", "chrysanthemum", "microscopic", "phenomenon",
    "equilibrium", "constellation", "juxtaposition", "hierarchical", "thermodynamics",
    "electromagnetic", "metamorphosis", "biodiversity", "cryptography", "neuroscience",
    "spheroid", "cerebro", "contractive", "manifold", "trajectory",
    "refractory", "resonance", "oscillation", "stochastic", "curvature",
    "apple", "elephant", "computer", "mountain", "telescope",
    "galaxy", "universe", "quantum", "gravity", "dimension",
    "chocolate", "avalanche", "butterfly", "dinosaur", "architecture",
    "frequency", "velocity", "acceleration", "momentum", "friction",
    "chemistry", "biology", "geology", "meteorology", "psychology",
    "philosophy", "literature", "symphony", "orchestra", "canvas",
    "sculpture", "compass", "navigation", "expedition", "adventure",
    "whisper", "thunder", "lightning", "hurricane", "tornado",
    "volcano", "tsunami", "earthquake", "glacier", "canyon",
    "diamond", "sapphire", "emerald", "platinum", "titanium",
    "hydrogen", "oxygen", "nitrogen", "carbon", "silicon",
    "calcium", "potassium", "phosphorus", "aluminum", "magnesium",
    "water", "sunlight", "atmosphere", "horizon", "twilight",
    "midnight", "equator", "solstice", "equinox", "meridian"
]


def generate_orthographic_examples(target_count=2000):
    examples = []
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    vowels = set("aeiou")

    for i in range(target_count):
        task_type = i % 5
        word = random.choice(VOCAB_WORDS)

        if task_type == 0:
            # Letter counting (e.g. How many 'r's in strawberry)
            char = random.choice(list(word))
            actual_count = word.count(char)
            question = f"How many times does the letter '{char}' appear in the word '{word}'?"
            
            # Step by step scratchpad breakdown
            steps = [f"To count the occurrences of '{char}' in '{word}', let's inspect each letter in sequence:"]
            c_found = 0
            for idx, letter in enumerate(word, 1):
                match = (letter.lower() == char.lower())
                if match:
                    c_found += 1
                    steps.append(f"Position {idx}: '{letter}' is a match (running count = {c_found}).")
                else:
                    steps.append(f"Position {idx}: '{letter}' is not '{char}'.")
            steps.append(f"Across the entire word, '{char}' appears {actual_count} time(s).")
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {actual_count}"

        elif task_type == 1:
            # Character indexing (e.g. What is the 3rd letter of ...)
            max_idx = min(len(word), len(ordinals))
            idx = random.randint(0, max_idx - 1)
            ord_name = ordinals[idx]
            target_char = word[idx]
            question = f"What is the {ord_name} letter of the word '{word}'?"
            
            steps = [f"Let's spell out the word '{word}' and identify the letter at position {idx + 1}:"]
            for pos in range(idx + 1):
                steps.append(f"Position {pos + 1}: '{word[pos]}'")
            steps.append(f"The {ord_name} letter is '{target_char}'.")
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {target_char}"

        elif task_type == 2:
            # String reversal
            rev_word = word[::-1]
            question = f"Reverse the word '{word}' letter by letter."
            steps = [f"To reverse '{word}', we write its letters in reverse order from last to first:"]
            rev_chars = []
            for pos in range(len(word) - 1, -1, -1):
                rev_chars.append(word[pos])
                steps.append(f"From index {pos + 1}: '{word[pos]}'")
            steps.append(f"Combining the letters in reverse order yields '{rev_word}'.")
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {rev_word}"

        elif task_type == 3:
            # Vowel count
            v_count = sum(1 for c in word if c.lower() in vowels)
            question = f"How many vowels (a, e, i, o, u) are in the word '{word}'?"
            steps = [f"Let's identify and count each vowel in '{word}':"]
            current_v = 0
            for idx, c in enumerate(word, 1):
                if c.lower() in vowels:
                    current_v += 1
                    steps.append(f"Position {idx}: '{c}' is a vowel (count = {current_v}).")
                else:
                    steps.append(f"Position {idx}: '{c}' is a consonant.")
            steps.append(f"Total number of vowels in '{word}' is {v_count}.")
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {v_count}"

        else:
            # Length and comparison
            c1, c2 = random.sample(list(set(word)), 2) if len(set(word)) >= 2 else (word[0], 'z')
            cnt1 = word.count(c1)
            cnt2 = word.count(c2)
            question = f"In the word '{word}', which letter appears more frequently: '{c1}' or '{c2}'?"
            steps = [
                f"Let's count the occurrences of each letter in '{word}':",
                f"1. Counting '{c1}': appears {cnt1} time(s).",
                f"2. Counting '{c2}': appears {cnt2} time(s)."
            ]
            if cnt1 > cnt2:
                res = f"'{c1}' appears more frequently ({cnt1} vs {cnt2})"
                final_sym = c1
            elif cnt2 > cnt1:
                res = f"'{c2}' appears more frequently ({cnt2} vs {cnt1})"
                final_sym = c2
            else:
                res = f"Both '{c1}' and '{c2}' appear with equal frequency ({cnt1} times each)"
                final_sym = "equal"
            steps.append(f"Conclusion: {res}.")
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {final_sym}"

        examples.append({
            "question": question,
            "answer": answer,
            "category": "orthographic_scratchpad"
        })

    return examples


# =====================================================================
# TRACK 2: TEMPORAL & CLOCK STATE-MACHINE REASONING (2,500)
# =====================================================================

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MONTHS = [
    ("January", 31), ("February", 28), ("March", 31), ("April", 30),
    ("May", 31), ("June", 30), ("July", 31), ("August", 31),
    ("September", 30), ("October", 31), ("November", 30), ("December", 31)
]


def generate_temporal_examples(target_count=2500):
    examples = []

    for i in range(target_count):
        sub_task = i % 5

        if sub_task == 0:
            # 12-hour AM/PM Addition
            start_hour = random.randint(1, 12)
            start_min = random.choice([0, 15, 30, 45, 10, 20, 50])
            start_ampm = random.choice(["AM", "PM"])
            dur_hours = random.randint(1, 8)
            dur_mins = random.choice([0, 15, 30, 45, 20, 40])

            # Compute in 24hr minutes from midnight
            base_24_h = start_hour % 12 + (12 if start_ampm == "PM" else 0)
            total_start_mins = base_24_h * 60 + start_min
            total_dur_mins = dur_hours * 60 + dur_mins
            total_end_mins = (total_start_mins + total_dur_mins) % (24 * 60)

            end_24_h = total_end_mins // 60
            end_min = total_end_mins % 60
            end_ampm = "AM" if end_24_h < 12 else "PM"
            end_12_h = end_24_h % 12
            if end_12_h == 0:
                end_12_h = 12

            question = (f"A train departs at {start_hour}:{start_min:02d} {start_ampm} and travels for "
                        f"{dur_hours} hour(s) and {dur_mins} minute(s). What time does it arrive?")
            
            steps = [
                f"1. Start time is {start_hour}:{start_min:02d} {start_ampm}.",
                f"2. Add the elapsed hours ({dur_hours}h) and minutes ({dur_mins}m):",
                f"   - Total elapsed minutes = ({dur_hours} * 60) + {dur_mins} = {total_dur_mins} minutes.",
                f"   - Converted into 24-hour time, arrival is at {end_24_h:02d}:{end_min:02d}.",
                f"3. In 12-hour format, {end_24_h:02d}:{end_min:02d} corresponds to {end_12_h}:{end_min:02d} {end_ampm}."
            ]
            ans_str = f"{end_12_h}:{end_min:02d} {end_ampm}"
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {ans_str}"

        elif sub_task == 1:
            # Weekday modular arithmetic
            day_idx = random.randint(0, 6)
            day_name = DAYS_OF_WEEK[day_idx]
            offset = random.randint(1, 14)
            target_idx = (day_idx + offset) % 7
            target_name = DAYS_OF_WEEK[target_idx]

            question = f"If today is {day_name}, what day of the week will it be in {offset} days?"
            steps = [
                f"1. There are 7 days in a week (cyclic mod 7): {', '.join(DAYS_OF_WEEK)}.",
                f"2. Today is {day_name} (index {day_idx}).",
                f"3. Adding {offset} days: ({day_idx} + {offset}) mod 7 = {day_idx + offset} mod 7 = {target_idx}.",
                f"4. Index {target_idx} corresponds to {target_name}."
            ]
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {target_name}"

        elif sub_task == 2:
            # Yesterday / Tomorrow chain logic
            base_idx = random.randint(0, 6)
            yesterday_name = DAYS_OF_WEEK[base_idx]
            today_name = DAYS_OF_WEEK[(base_idx + 1) % 7]
            tomorrow_name = DAYS_OF_WEEK[(base_idx + 2) % 7]

            question = f"If yesterday was {yesterday_name}, what day is tomorrow?"
            steps = [
                f"1. If yesterday was {yesterday_name}, then today is {today_name}.",
                f"2. Tomorrow is the day following today.",
                f"3. The day after {today_name} is {tomorrow_name}."
            ]
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {tomorrow_name}"

        elif sub_task == 3:
            # Duration between two times on the same day
            h1 = random.randint(6, 11)
            m1 = random.choice([0, 15, 30])
            h2 = random.randint(1, 9)
            m2 = random.choice([0, 15, 30, 45])
            
            t1_mins = h1 * 60 + m1
            t2_mins = (h2 + 12) * 60 + m2
            diff_mins = t2_mins - t1_mins
            diff_h = diff_mins // 60
            diff_m = diff_mins % 60

            question = f"How many hours and minutes are there between {h1}:{m1:02d} AM and {h2}:{m2:02d} PM on the same day?"
            steps = [
                f"1. Convert both times to minutes from midnight:",
                f"   - {h1}:{m1:02d} AM = {t1_mins} minutes.",
                f"   - {h2}:{m2:02d} PM = ({h2} + 12)*60 + {m2} = {t2_mins} minutes.",
                f"2. Subtract start from end: {t2_mins} - {t1_mins} = {diff_mins} minutes.",
                f"3. Convert back to hours and minutes: {diff_mins} // 60 = {diff_h} hour(s), remainder = {diff_m} minute(s)."
            ]
            dur_str = f"{diff_h} hours and {diff_m} minutes" if diff_m > 0 else f"{diff_h} hours"
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {dur_str}"

        else:
            # Calendar day forward arithmetic
            m_idx = random.randint(0, 10)
            m_name, m_days = MONTHS[m_idx]
            d = random.randint(1, m_days - 10)
            add_days = random.randint(2, 8)
            new_day = d + add_days

            question = f"If today is {m_name} {d}, what will the date be in {add_days} days?"
            steps = [
                f"1. Today is {m_name} {d}.",
                f"2. Adding {add_days} days within {m_name} (which has {m_days} days):",
                f"   - {d} + {add_days} = {new_day}.",
                f"3. The date will be {m_name} {new_day}."
            ]
            ans_str = f"{m_name} {new_day}"
            reasoning = "\n".join(steps)
            answer = f"{reasoning}\n#### {ans_str}"

        examples.append({
            "question": question,
            "answer": answer,
            "category": "temporal_state_machine"
        })

    return examples


# =====================================================================
# TRACK 3: COUNTERFACTUAL & PREMISE-REJECTION / ANTI-SYCOPHANCY (2,500)
# =====================================================================

COUNTERFACTUAL_TEMPLATES = [
    {
        "pattern": "Who was the first person to walk on the {celestial}?",
        "celestial_list": ["Sun", "core of Jupiter", "rings of Saturn", "center of a black hole", "surface of Venus"],
        "reason": "Nobody has ever walked on the {c}. It has extreme physical conditions (such as immense plasma temperatures exceeding thousands of degrees or crushing atmospheric pressures) that would instantly destroy any human or spacecraft.",
        "final": "Nobody"
    },
    {
        "pattern": "What is the capital city of {fake_place}?",
        "places": [
            ("Mars", "Mars is an uninhabited planet with no countries, cities, or human government."),
            ("Antarctica", "Antarctica is a continent governed by international scientific treaty and has no sovereign countries or capital city."),
            ("the Pacific Ocean", "The Pacific Ocean is a body of water and does not contain nations or capital cities."),
            ("the Moon", "The Moon is an astronomical body with no sovereign nation or capital city."),
            ("the Earth's mantle", "The Earth's mantle is an interior geological layer composed of solid and semi-molten rock with no human habitation.")
        ]
    },
    {
        "pattern": "Can a human {impossible_act} without {equipment}?",
        "acts": [
            ("breathe underwater", "equipment or gills", "No. Humans have lungs designed to extract oxygen from air, not dissolved oxygen in water. Inhaling water causes drowning."),
            ("fly by flapping their arms", "wings, gliders, or aircraft", "No. Human arms cannot generate aerodynamic lift, and human muscle-to-weight ratios are insufficient for self-powered flight."),
            ("survive drinking pure molten lava", "immediate death", "No. Molten lava reaches temperatures between 700°C and 1,200°C, which would incinerate organic tissue immediately."),
            ("travel faster than the speed of light in vacuum", "infinite energy", "No. According to special relativity, accelerating any massive object to or beyond the speed of light would require infinite energy, which is physically impossible."),
            ("live without any water intake for 3 months", "hydration", "No. Human biology requires water for cellular function; death from dehydration typically occurs within 3 to 7 days without water.")
        ]
    },
    {
        "pattern": "How many {tech} did {historical_figure} use during {event}?",
        "scenarios": [
            ("iPhones", "Abraham Lincoln", "the American Civil War", "Abraham Lincoln lived from 1809 to 1865, while the iPhone was invented in 2007. Mobile phones did not exist in the 19th century.", "0"),
            ("airplanes", "Julius Caesar", "the conquest of Gaul", "Julius Caesar lived in the 1st century BC. Powered aircraft were not invented until 1903 by the Wright brothers.", "0"),
            ("laptops", "William Shakespeare", "writing Hamlet", "Shakespeare lived during the Elizabethan and Jacobean eras (1564–1616). Computers and laptops were invented in the 20th century.", "0"),
            ("automobiles", "Alexander the Great", "his Persian campaign", "Alexander the Great lived in the 4th century BC. Motor vehicles were invented over two millennia later in the late 19th century.", "0"),
            ("satellites", "George Washington", "the Battle of Yorktown", "George Washington fought in the 18th century. Artificial satellites were not launched until Sputnik 1 in 1957.", "0")
        ]
    },
    {
        "pattern": "Which is heavier: {n} kilograms of {item1} or {n} kilograms of {item2}?",
        "pairs": [
            ("feathers", "lead"),
            ("cotton", "gold"),
            ("bricks", "water"),
            ("steel", "straw"),
            ("sand", "iron")
        ]
    }
]


def generate_counterfactual_examples(target_count=2500):
    examples = []

    for i in range(target_count):
        tmpl_idx = i % len(COUNTERFACTUAL_TEMPLATES)

        if tmpl_idx == 0:
            c = random.choice(COUNTERFACTUAL_TEMPLATES[0]["celestial_list"])
            question = f"Who was the first person to walk on the {c}?"
            explanation = (f"Nobody has ever walked on the {c}. The {c} possesses extreme environmental conditions "
                           f"(such as surface temperatures of thousands of degrees or crushing atmospheric forces) "
                           f"that make human survival or presence physically impossible.")
            answer = f"{explanation}\n#### Nobody"

        elif tmpl_idx == 1:
            place, reason = random.choice(COUNTERFACTUAL_TEMPLATES[1]["places"])
            question = f"What is the capital city of {place}?"
            explanation = f"{place} does not have a capital city. {reason}"
            answer = f"{explanation}\n#### None"

        elif tmpl_idx == 2:
            act, equip, reason = random.choice(COUNTERFACTUAL_TEMPLATES[2]["acts"])
            question = f"Can a human {act} without {equip}?"
            explanation = f"{reason}"
            answer = f"{explanation}\n#### No"

        elif tmpl_idx == 3:
            tech, figure, event, reason, zero = random.choice(COUNTERFACTUAL_TEMPLATES[3]["scenarios"])
            question = f"How many {tech} did {figure} use during {event}?"
            explanation = f"{figure} used zero {tech}. {reason}"
            answer = f"{explanation}\n#### 0"

        else:
            n = random.randint(2, 50)
            item1, item2 = random.choice(COUNTERFACTUAL_TEMPLATES[4]["pairs"])
            question = f"Which is heavier: {n} kilograms of {item1} or {n} kilograms of {item2}?"
            explanation = (f"Both weigh the exact same. A mass of {n} kilograms of {item1} is identical in weight to "
                           f"{n} kilograms of {item2}, as both have a mass of {n} kg.")
            answer = f"{explanation}\n#### They weigh the same"

        examples.append({
            "question": question,
            "answer": answer,
            "category": "counterfactual_premise_rejection"
        })

    return examples


# =====================================================================
# TRACK 4: GSM8K MULTI-STEP ELEMENTARY ARITHMETIC (2,500)
# =====================================================================

def load_gsm8k_curated_examples(parquet_path: str, target_count=2500):
    print(f"📖 Curating {target_count} multi-step math problems from {parquet_path}...")
    table = pq.read_table(parquet_path)
    questions = table["question"]
    answers = table["answer"]

    total = len(questions)
    indices = list(range(total))
    random.shuffle(indices)
    curated_indices = indices[:min(target_count, total)]

    examples = []
    for idx in curated_indices:
        q = questions[idx].as_py().strip()
        a = answers[idx].as_py().strip()
        examples.append({
            "question": q,
            "answer": a,
            "category": "elementary_math_gsm8k"
        })

    return examples


def main():
    print("=" * 75)
    print("🧠 CEREBRO-COGNITIVE-10K DATASET ENGINE")
    print("===========================================================================")

    output_dir = os.path.join(ROOT_DIR, "data", "cerebro_cognitive_8k")
    os.makedirs(output_dir, exist_ok=True)

    gsm8k_parquet = os.path.join(ROOT_DIR, "data", "gsm8k", "gsm8k_train.parquet")
    if not os.path.exists(gsm8k_parquet):
        print(f"❌ Error: {gsm8k_parquet} not found!")
        sys.exit(1)

    print("\n[1/4] Generating Orthographic Scratchpad Reasoning examples (2,000)...")
    ortho_examples = generate_orthographic_examples(2000)
    print(f"      ✅ Generated {len(ortho_examples)} orthographic problems.")

    print("\n[2/4] Generating Temporal & Clock State-Machine examples (2,500)...")
    temporal_examples = generate_temporal_examples(2500)
    print(f"      ✅ Generated {len(temporal_examples)} temporal state-machine problems.")

    print("\n[3/4] Generating Counterfactual & Premise-Rejection examples (2,500)...")
    counterfactual_examples = generate_counterfactual_examples(2500)
    print(f"      ✅ Generated {len(counterfactual_examples)} premise-rejection problems.")

    print("\n[4/4] Ingesting GSM8K Multi-Step Math examples (2,500)...")
    math_examples = load_gsm8k_curated_examples(gsm8k_parquet, 2500)
    print(f"      ✅ Ingested {len(math_examples)} GSM8k problems.")

    all_records = ortho_examples + temporal_examples + counterfactual_examples + math_examples
    random.shuffle(all_records)
    total_count = len(all_records)

    print(f"\n📊 Total Compiled Dataset Size: {total_count:,} records.")

    # Save to Parquet
    parquet_path = os.path.join(output_dir, "cerebro_cognitive_10k.parquet")
    schema = pa.schema([
        ("question", pa.string()),
        ("answer", pa.string()),
        ("category", pa.string()),
    ])
    q_col = pa.array([r["question"] for r in all_records], type=pa.string())
    a_col = pa.array([r["answer"] for r in all_records], type=pa.string())
    c_col = pa.array([r["category"] for r in all_records], type=pa.string())
    table = pa.Table.from_arrays([q_col, a_col, c_col], schema=schema)
    pq.write_table(table, parquet_path, compression="snappy")
    print(f"💾 Saved Parquet dataset to: {parquet_path} ({os.path.getsize(parquet_path) / 1024:.1f} KB)")

    # Save to JSONL for quick inspection
    jsonl_path = os.path.join(output_dir, "cerebro_cognitive_10k.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"💾 Saved JSONL dataset to  : {jsonl_path} ({os.path.getsize(jsonl_path) / 1024:.1f} KB)")

    # Also make available in eval/ directory for testing
    eval_benchmark_path = os.path.join(ROOT_DIR, "eval", "cerebro_cognitive_sample_200.json")
    with open(eval_benchmark_path, "w", encoding="utf-8") as f:
        json.dump(all_records[:200], f, indent=2)
    print(f"💾 Saved 200-sample test benchmark to: {eval_benchmark_path}")

    print("\n✅ Dataset generation complete.")


if __name__ == "__main__":
    main()
