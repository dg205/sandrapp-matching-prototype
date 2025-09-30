import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

# === Step 1: Define ALL social features (old + new) ===
FEATURES = [
    # Existing
    "age_diff", "same_city", "same_religion", "same_mobility",
    "interest_overlap", "tech_compatibility", "comm_style_compatibility",
    "comfort_compatibility",

    # New from PDF
    "food_cuisine_overlap", "dietary_restriction_conflict",
    "cultural_background_match", "holiday_overlap", "multilingual_fluency_match",
    "hobby_overlap", "spirituality_match",
    "life_stage_needs_alignment",
    "pref_comm_style_match",
    "tech_affinity_gap",
    "companionship_gap_overlap",
    "volunteering_help_match",
    "shared_memory_trigger_overlap"
]

# === Step 2: Load training data (new expanded dataset) ===
df = pd.read_csv("user_match_pairs_expanded_synthetic.csv")
X = df[FEATURES]
y = df["match_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_scaled, y)

# === Step 3: Get inputs for two users (now expanded) ===
def get_user_input(user_id):
    print(f"\nEnter info for User {user_id}:")
    age = int(input("Age: "))
    city = input("City (e.g., ATL, CHI): ")
    religion = input("Religion: ")
    mobility = input("Mobility level (low, med, high): ")
    interests = input("List interests (comma-separated): ").lower().split(',')
    tech = int(input("Tech comfort (1-5): "))
    comm = input("Comm style (direct, indirect): ")
    comfort = int(input("Comfort with new people (1-5): "))

    # NEW features (quick prompts, placeholders for now)
    food = int(input("Share similar cuisine? (1=yes,0=no): "))
    diet_conflict = int(input("Dietary conflict? (1=yes,0=no): "))
    culture = int(input("Same cultural background? (1=yes,0=no): "))
    holiday = int(input("Share holiday traditions? (1=yes,0=no): "))
    lang = int(input("Multilingual match? (1=yes,0=no): "))
    hobby = int(input("Hobby overlap? (1=yes,0=no): "))
    spirit = int(input("Spirituality match? (1=yes,0=no): "))
    life_stage = int(input("Life stage alignment? (1=yes,0=no): "))
    comm_pref = int(input("Same preferred comm style? (1=yes,0=no): "))
    tech_gap = int(input("Tech gap present? (1=yes,0=no): "))
    comp_gap = int(input("Companionship gap overlap? (1=yes,0=no): "))
    volunteer = int(input("Volunteering/help match? (1=yes,0=no): "))
    memory = int(input("Shared memory trigger overlap? (1=yes,0=no): "))

    return {
        "age": age,
        "city": city.strip().lower(),
        "religion": religion.strip().lower(),
        "mobility": mobility.strip().lower(),
        "interests": set([i.strip() for i in interests]),
        "tech": tech,
        "comm": comm.strip().lower(),
        "comfort": comfort,
        "food": food, "diet_conflict": diet_conflict,
        "culture": culture, "holiday": holiday, "lang": lang,
        "hobby": hobby, "spirit": spirit, "life_stage": life_stage,
        "comm_pref": comm_pref, "tech_gap": tech_gap, "comp_gap": comp_gap,
        "volunteer": volunteer, "memory": memory
    }

def encode_pair(u1, u2):
    return {
        # Old features
        "age_diff": abs(u1["age"] - u2["age"]),
        "same_city": int(u1["city"] == u2["city"]),
        "same_religion": int(u1["religion"] == u2["religion"]),
        "same_mobility": int(u1["mobility"] == u2["mobility"]),
        "interest_overlap": len(u1["interests"] & u2["interests"]),
        "tech_compatibility": int(abs(u1["tech"] - u2["tech"]) <= 1),
        "comm_style_compatibility": int(u1["comm"] == u2["comm"]),
        "comfort_compatibility": int(abs(u1["comfort"] - u2["comfort"]) <= 1),

        # New features (direct use from user input)
        "food_cuisine_overlap": u1["food"] & u2["food"],
        "dietary_restriction_conflict": max(u1["diet_conflict"], u2["diet_conflict"]),
        "cultural_background_match": u1["culture"] & u2["culture"],
        "holiday_overlap": u1["holiday"] & u2["holiday"],
        "multilingual_fluency_match": u1["lang"] & u2["lang"],
        "hobby_overlap": u1["hobby"] & u2["hobby"],
        "spirituality_match": u1["spirit"] & u2["spirit"],
        "life_stage_needs_alignment": u1["life_stage"] & u2["life_stage"],
        "pref_comm_style_match": u1["comm_pref"] & u2["comm_pref"],
        "tech_affinity_gap": max(u1["tech_gap"], u2["tech_gap"]),
        "companionship_gap_overlap": u1["comp_gap"] & u2["comp_gap"],
        "volunteering_help_match": u1["volunteer"] & u2["volunteer"],
        "shared_memory_trigger_overlap": u1["memory"] & u2["memory"]
    }

# === Step 4: Predict match score ===
def predict_match_score(pair_features):
    pair_df = pd.DataFrame([pair_features])
    scaled = scaler.transform(pair_df)
    prob = mlp.predict_proba(scaled)[0][1]
    return round(prob, 3)

# === Step 5: Run matching simulation ===
def run_matching_simulation():
    print("Enter 3 users for matching simulation (A, B, C):")
    users = [get_user_input(i+1) for i in range(3)]

    # Generate all pairwise match scores
    scores = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                scores[i][j] = -np.inf
            else:
                pair = encode_pair(users[i], users[j])
                scores[i][j] = predict_match_score(pair)

    print("\nMatch Score Matrix (Row i prefers higher column scores):")
    df_scores = pd.DataFrame(scores, columns=["User 1", "User 2", "User 3"], index=["User 1", "User 2", "User 3"])
    print(df_scores)

    # Apply Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(-scores)
    print("\nOptimal Pairings:")
    for r, c in zip(row_ind, col_ind):
        if r != c:
            print(f"User {r+1} <> User {c+1} (Score: {scores[r][c]})")

# === Start ===
run_matching_simulation()
