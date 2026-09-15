import pandas as pd

input_file = "data/processed/training_sample.csv"
output_file = "data/processed/training_sample.csv"

df = pd.read_csv(input_file)

# Make sure intent can accept text labels
df["intent"] = df["intent"].astype("object")

# Show first 50 unlabeled examples
batch = df[df["intent"].isna() | (df["intent"] == "")].head(50)

print("Number of examples in this batch:", len(batch))
print("\n========================================")
print("TRAINING LABELING BATCH 1")
print("========================================\n")

for i, (_, row) in enumerate(batch.iterrows(), start=1):
    print(f"{i}. ID: {row['customer_tweet_id']}")
    print(f"   Message: {row['customer_message']}")
    print()

    labels = [
    "other",
    "other",
    "other",
    "refund_return",
    "order_issue",
    "delivery_issue",
    "prime_issue",
    "other",
    "other",
    "prime_issue",
    "payment_issue",
    "order_issue",
    "delivery_issue",
    "other",
    "payment_issue",
    "product_device_issue",
    "delivery_issue",
    "delivery_issue",
    "other",
    "account_issue",
    "refund_return",
    "delivery_issue",
    "product_device_issue",
    "delivery_issue",
    "delivery_issue",
    "refund_return",
    "other",
    "delivery_issue",
    "other",
    "product_device_issue",
    "product_device_issue",
    "payment_issue",
    "delivery_issue",
    "product_device_issue",
    "delivery_issue",
    "other",
    "delivery_issue",
    "account_issue",
    "other",
    "other",
    "other",
    "other",
    "refund_return",
    "order_issue",
    "product_device_issue",
    "delivery_issue",
    "delivery_issue",
    "prime_issue",
    "delivery_issue",
    "delivery_issue"
]

    if len(labels) != len(batch):
        raise ValueError("Number of labels does not match number of examples.")

for (_, row), label in zip(batch.iterrows(), labels):
    df.loc[df["customer_tweet_id"] == row["customer_tweet_id"], "intent"] = label

df.to_csv(output_file, index=False)

print("\nBatch 1 labels saved successfully!")
print("\nLabels assigned:")
print(batch[["customer_tweet_id", "customer_message"]].copy())