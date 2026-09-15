import pandas as pd
from sklearn.metrics import cohen_kappa_score

INPUT = "outputs/human_reply_ratings.csv"
OUTPUT = "outputs/human_agreement_results.csv"

df = pd.read_csv(INPUT)

dimensions = [
    "Correctness",
    "Relevance",
    "Grounding",
    "Helpfulness",
    "No_Hallucination"
]

results = []

for col in dimensions:
    if col not in df.columns:
        print(f"Skipping {col}: column not found")
        continue

    human = pd.to_numeric(df[col], errors="coerce").dropna()

    if len(human) == 0:
        continue

    # This file contains manual ratings only.
    # We therefore calculate descriptive statistics,
    # not human-vs-LLM agreement.
    results.append({
        "dimension": col,
        "samples": len(human),
        "mean_rating": round(human.mean(), 2),
        "min_rating": int(human.min()),
        "max_rating": int(human.max())
    })

result_df = pd.DataFrame(results)

print("\nManual Reply Rating Summary")
print("=" * 50)
print(result_df.to_string(index=False))

result_df.to_csv(OUTPUT, index=False)

print(f"\nSaved to: {OUTPUT}")