import pandas as pd

# Load cleaned Amazon conversations
df = pd.read_csv("data/processed/amazon_clean.csv")

# Remove the 200 golden examples
golden = pd.read_csv("data/golden/labeling_sample.csv")

golden_ids = set(golden["customer_tweet_id"])

df = df[~df["customer_tweet_id"].isin(golden_ids)]

# Take 500 examples for training
training = df.sample(n=500, random_state=123)

# Add empty label columns
training["intent"] = ""
training["label_notes"] = ""

# Save
training.to_csv("data/processed/training_sample.csv", index=False)

print("Training examples created:", len(training))
print("\nColumns:")
print(training.columns.tolist())

print("\nSample:")
print(training[["customer_tweet_id", "customer_message"]].head(20).to_string(index=False))