import pandas as pd
import os

# Load cleaned Amazon conversations
input_file = "data/processed/amazon_clean.csv"

df = pd.read_csv(input_file)

# Select 200 random examples
sample = df.sample(
    n=200,
    random_state=42
).copy()

# Add empty label columns
sample["intent"] = ""
sample["label_notes"] = ""

# Create golden directory
os.makedirs("data/golden", exist_ok=True)

# Save
output_file = "data/golden/labeling_sample.csv"

sample.to_csv(
    output_file,
    index=False
)

print("Created:", output_file)
print("Number of examples:", len(sample))

print("\nColumns:")
print(sample.columns.tolist())

print("\nFirst 10 examples:")
print(
    sample[
        ["customer_tweet_id", "customer_message", "intent"]
    ].head(10).to_string(index=False)
)