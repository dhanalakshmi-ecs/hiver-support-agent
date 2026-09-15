import pandas as pd
from sklearn.model_selection import train_test_split

# Load the 200-example golden set
df = pd.read_csv("data/golden/labeling_sample.csv")

# Take 50 examples for escalation development
dev, test = train_test_split(
    df,
    test_size=150,
    random_state=42,
    stratify=df["intent"]
)

# Keep only the columns needed for escalation labeling
dev = dev[
    [
        "customer_tweet_id",
        "customer_message",
        "intent",
        "amazon_reply"
    ]
].copy()

# Add empty columns for human labeling
dev["escalation_label"] = ""
dev["escalation_notes"] = ""

# Save the 50 examples
dev.to_csv(
    "data/golden/escalation_sample.csv",
    index=False
)

print("Escalation sample created successfully!")
print("Number of examples:", len(dev))

print("\nIntent distribution:")
print(dev["intent"].value_counts())

print("\nFile saved to:")
print("data/golden/escalation_sample.csv")