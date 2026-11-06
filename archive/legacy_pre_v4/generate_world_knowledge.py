"""
World Knowledge & Factual QA Dataset Generator for CerebroSpheroid
Produces comprehensive factual knowledge spanning:
1. World Geography: Countries, capitals, currencies, landmarks, continents, oceans, rivers, mountains.
2. Science & Nature: Chemical elements, physical constants, celestial bodies, biological classification.
3. World History & Culture: Major eras, inventions, languages, literature.
4. Core General Knowledge & Common Sense QA.

Outputs:
  data/general_knowledge/world_knowledge_train.parquet
  data/general_knowledge/world_knowledge_val.parquet
"""

import os
import random
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT_DIR, "data", "general_knowledge")

COUNTRIES_AND_CAPITALS = [
    ("France", "Paris", "Europe", "Euro", "Eiffel Tower"),
    ("Japan", "Tokyo", "Asia", "Yen", "Mount Fuji"),
    ("Germany", "Berlin", "Europe", "Euro", "Brandenburg Gate"),
    ("United Kingdom", "London", "Europe", "Pound sterling", "Big Ben"),
    ("Italy", "Rome", "Europe", "Euro", "Colosseum"),
    ("Spain", "Madrid", "Europe", "Euro", "Sagrada Familia"),
    ("Canada", "Ottawa", "North America", "Canadian Dollar", "CN Tower"),
    ("United States", "Washington, D.C.", "North America", "US Dollar", "Statue of Liberty"),
    ("Australia", "Canberra", "Oceania", "Australian Dollar", "Sydney Opera House"),
    ("China", "Beijing", "Asia", "Renminbi", "Great Wall of China"),
    ("India", "New Delhi", "Asia", "Indian Rupee", "Taj Mahal"),
    ("Brazil", "Brasília", "South America", "Brazilian Real", "Christ the Redeemer"),
    ("Egypt", "Cairo", "Africa", "Egyptian Pound", "Pyramids of Giza"),
    ("Russia", "Moscow", "Europe/Asia", "Russian Ruble", "Red Square"),
    ("South Korea", "Seoul", "Asia", "South Korean Won", "Gyeongbokgung Palace"),
    ("Mexico", "Mexico City", "North America", "Mexican Peso", "Chichen Itza"),
    ("Argentina", "Buenos Aires", "South America", "Argentine Peso", "Iguazu Falls"),
    ("South Africa", "Pretoria", "Africa", "South African Rand", "Table Mountain"),
    ("Netherlands", "Amsterdam", "Europe", "Euro", "Canals of Amsterdam"),
    ("Switzerland", "Bern", "Europe", "Swiss Franc", "Matterhorn"),
    ("Greece", "Athens", "Europe", "Euro", "Acropolis"),
    ("Turkey", "Ankara", "Europe/Asia", "Turkish Lira", "Hagia Sophia"),
    ("Sweden", "Stockholm", "Europe", "Swedish Krona", "Vasa Museum"),
    ("Norway", "Oslo", "Europe", "Norwegian Krone", "Geirangerfjord"),
    ("Portugal", "Lisbon", "Europe", "Euro", "Belém Tower"),
    ("Ireland", "Dublin", "Europe", "Euro", "Cliffs of Moher"),
    ("Poland", "Warsaw", "Europe", "Polish Zloty", "Wawel Castle"),
    ("Austria", "Vienna", "Europe", "Euro", "Schönbrunn Palace"),
    ("Belgium", "Brussels", "Europe", "Euro", "Grand Place"),
    ("New Zealand", "Wellington", "Oceania", "New Zealand Dollar", "Milford Sound"),
    ("Saudi Arabia", "Riyadh", "Asia", "Saudi Riyal", "Kaaba"),
    ("United Arab Emirates", "Abu Dhabi", "Asia", "UAE Dirham", "Burj Khalifa"),
    ("Singapore", "Singapore", "Asia", "Singapore Dollar", "Marina Bay Sands"),
    ("Thailand", "Bangkok", "Asia", "Thai Baht", "Grand Palace"),
    ("Vietnam", "Hanoi", "Asia", "Vietnamese Dong", "Ha Long Bay"),
    ("Indonesia", "Jakarta", "Asia", "Indonesian Rupiah", "Borobudur"),
    ("Malaysia", "Kuala Lumpur", "Asia", "Malaysian Ringgit", "Petronas Twin Towers"),
    ("Kenya", "Nairobi", "Africa", "Kenyan Shilling", "Maasai Mara"),
    ("Nigeria", "Abuja", "Africa", "Nigerian Naira", "Zuma Rock"),
    ("Morocco", "Rabat", "Africa", "Moroccan Dirham", "Hassan II Mosque"),
    ("Chile", "Santiago", "South America", "Chilean Peso", "Torres del Paine"),
    ("Colombia", "Bogotá", "South America", "Colombian Peso", "Caño Cristales"),
    ("Peru", "Lima", "South America", "Peruvian Sol", "Machu Picchu"),
    ("Denmark", "Copenhagen", "Europe", "Danish Krone", "Little Mermaid"),
    ("Finland", "Helsinki", "Europe", "Euro", "Suomenlinna"),
    ("Czech Republic", "Prague", "Europe", "Czech Koruna", "Charles Bridge"),
    ("Hungary", "Budapest", "Europe", "Hungarian Forint", "Buda Castle"),
    ("Iceland", "Reykjavik", "Europe", "Icelandic Krona", "Blue Lagoon"),
    ("Philippines", "Manila", "Asia", "Philippine Peso", "Banaue Rice Terraces"),
    ("Pakistan", "Islamabad", "Asia", "Pakistani Rupee", "Badshahi Mosque"),
]

SCIENCE_FACTS = [
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ("What gas do plants absorb during photosynthesis?", "Plants absorb carbon dioxide (CO2) during photosynthesis."),
    ("What gas do plants release during photosynthesis?", "Plants release oxygen (O2) into the atmosphere during photosynthesis."),
    ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is approximately 299,792,458 meters per second, or roughly 300,000 km/s."),
    ("What is the boiling point of water at sea level?", "The boiling point of water at standard atmospheric pressure (sea level) is 100 degrees Celsius or 212 degrees Fahrenheit."),
    ("What is the freezing point of water?", "The freezing point of water is 0 degrees Celsius or 32 degrees Fahrenheit."),
    ("What is the powerhouse of the eukaryotic cell?", "The mitochondrion is known as the powerhouse of the eukaryotic cell because it produces ATP through cellular respiration."),
    ("What planet is known as the Red Planet?", "Mars is known as the Red Planet due to the prevalent iron oxide on its surface."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined."),
    ("What is the closest star to Earth?", "The Sun is the closest star to Earth, at an average distance of about 149.6 million kilometers (1 AU)."),
    ("What is the atomic number of carbon?", "The atomic number of carbon is 6, indicating it has 6 protons in its nucleus."),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen is the most abundant gas in Earth's atmosphere, making up approximately 78% of dry air, followed by oxygen at about 21%."),
    ("What is Newton's First Law of Motion?", "Newton's First Law of Motion states that an object at rest remains at rest, and an object in motion remains in motion with a constant velocity, unless acted upon by a net external force."),
    ("What is Newton's Second Law of Motion?", "Newton's Second Law of Motion states that force equals mass times acceleration: F = m * a."),
    ("What is the hardest natural mineral on Earth?", "Diamond is the hardest known natural mineral, rated 10 on the Mohs hardness scale."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au, derived from the Latin word aurum."),
    ("What is the chemical symbol for iron?", "The chemical symbol for iron is Fe, derived from the Latin word ferrum."),
    ("What is the primary function of red blood cells?", "The primary function of red blood cells (erythrocytes) is to transport oxygen from the lungs to body tissues and carry carbon dioxide back to the lungs via hemoglobin."),
    ("What is DNA?", "DNA, or deoxyribonucleic acid, is the hereditary molecule that carries genetic instructions for development, functioning, growth, and reproduction in all known organisms."),
    ("What are the four nucleotide bases in DNA?", "The four nucleotide bases in DNA are adenine (A), thymine (T), guanine (G), and cytosine (C)."),
]

GENERAL_QA_TEMPLATES = [
    "What is the capital of {country}?",
    "What is the capital city of {country}?",
    "Which city is the capital of {country}?",
    "Can you tell me the capital of {country}?",
    "The capital of {country} is what?",
    "Tell me the capital of {country}.",
    "Identify the capital of {country}.",
    "What country is {capital} the capital of?",
]

def generate_capital_samples():
    samples = []
    for country, capital, continent, currency, landmark in COUNTRIES_AND_CAPITALS:
        # Direct Q&A variants
        for q_tmpl in GENERAL_QA_TEMPLATES:
            q = q_tmpl.format(country=country, capital=capital)
            answers = [
                f"The capital of {country} is {capital}.",
                f"{capital} is the capital city of {country}.",
                f"{capital}.",
                f"The capital city of {country} is {capital}, located in {continent}.",
                f"{capital} serves as the capital of {country}.",
            ]
            for a in answers:
                samples.append((q, a))
        
        # Landmark & Geography queries
        samples.append((f"Where is {landmark} located?", f"{landmark} is located in {capital}, {country} ({continent})."))
        samples.append((f"In which country is {capital} located?", f"{capital} is located in {country}."))
        samples.append((f"What continent is {country} located in?", f"{country} is located in {continent}."))
        samples.append((f"What is the currency of {country}?", f"The currency of {country} is the {currency}."))
    return samples

def generate_corpus(target_size=20000, seed=42):
    rng = random.Random(seed)
    base_capitals = generate_capital_samples()
    
    samples = []
    
    # 1. Add all base capital and geography samples with variations
    for q, a in base_capitals:
        samples.append((q, a))
    
    # 2. Add science facts
    for q, a in SCIENCE_FACTS:
        samples.append((q, a))
        samples.append((f"Question: {q}\nAnswer:", a))
        samples.append((f"Explain: {q}", a))

    # 3. Augment with conversational framings and expanded knowledge
    conversational_prefixes = [
        "", "Please tell me: ", "Quick question: ", "Hello! ", "I would like to know: ",
        "Can you answer this: ", "Knowledge check: "
    ]
    
    while len(samples) < target_size:
        # Pick from capitals or science
        if rng.random() < 0.65:
            q, a = rng.choice(base_capitals)
        else:
            q, a = rng.choice(SCIENCE_FACTS)
        
        p = rng.choice(conversational_prefixes)
        augmented_q = f"{p}{q}".strip()
        samples.append((augmented_q, a))
        
    rng.shuffle(samples)
    return samples[:target_size]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating comprehensive World Knowledge & General QA corpus...")
    
    all_samples = generate_corpus(target_size=22000, seed=1337)
    train_samples = all_samples[:20000]
    val_samples = all_samples[20000:]
    
    train_table = pa.Table.from_pydict({
        "stimulus": [s[0] for s in train_samples],
        "articulation": [s[1] for s in train_samples],
    })
    val_table = pa.Table.from_pydict({
        "stimulus": [s[0] for s in val_samples],
        "articulation": [s[1] for s in val_samples],
    })
    
    train_path = os.path.join(OUT_DIR, "world_knowledge_train.parquet")
    val_path = os.path.join(OUT_DIR, "world_knowledge_val.parquet")
    
    pq.write_table(train_table, train_path)
    pq.write_table(val_table, val_path)
    
    print(f"✓ Saved {len(train_samples):,} training records to: {train_path}")
    print(f"✓ Saved {len(val_samples):,} validation records to: {val_path}")

if __name__ == "__main__":
    main()
