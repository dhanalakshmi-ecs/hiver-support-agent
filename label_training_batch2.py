import pandas as pd

input_file = "data/processed/training_sample.csv"
output_file = "data/processed/training_sample.csv"

df = pd.read_csv(input_file)

df["intent"] = df["intent"].astype("object")

batch = df[df["intent"].isna() | (df["intent"] == "")].head(50)

print("Number of examples in this batch:", len(batch))

print("\n========================================")
print("TRAINING LABELING BATCH 2")
print("========================================\n")

for i, (_, row) in enumerate(batch.iterrows(), start=1):
    print(f"{i}. ID: {row['customer_tweet_id']}")
    print(f"   Message: {row['customer_message']}")
    print()

labels = [
    "payment_issue",          # 1
    "other",                  # 2
    "other",                  # 3
    "order_issue",            # 4
    "account_issue",          # 5
    "product_device_issue",   # 6
    "other",                  # 7
    "refund_return",          # 8
    "prime_issue",            # 9
    "other",                  # 10
    "order_issue",            # 11
    "other",                  # 12
    "delivery_issue",         # 13
    "payment_issue",          # 14
    "other",                  # 15
    "delivery_issue",         # 16
    "other",                  # 17
    "other",                  # 18
    "delivery_issue",         # 19
    "refund_return",          # 20
    "order_issue",            # 21
    "payment_issue",          # 22
    "prime_issue",            # 23
    "product_device_issue",   # 24
    "delivery_issue",         # 25
    "delivery_issue",         # 26
    "delivery_issue",         # 27
    "other",                  # 28
    "order_issue",            # 29
    "product_device_issue",   # 30
    "delivery_issue",         # 31
    "other",                  # 32
    "delivery_issue",         # 33
    "order_issue",            # 34
    "other",                  # 35
    "other",                  # 36
    "delivery_issue",         # 37
    "delivery_issue",         # 38
    "delivery_issue",         # 39
    "delivery_issue",         # 40
    "delivery_issue",         # 41
    "delivery_issue",         # 42
    "delivery_issue",         # 43
    "delivery_issue",         # 44
    "order_issue",            # 45
    "other",                  # 46
    "delivery_issue",         # 47
    "other",                  # 48
    "other",                  # 49
    "other"                   # 50
]

if len(labels) != len(batch):
    raise ValueError("Number of labels does not match number of examples.")

for (_, row), label in zip(batch.iterrows(), labels):
    df.loc[
        df["customer_tweet_id"] == row["customer_tweet_id"],
        "intent"
    ] = label

df.to_csv(output_file, index=False)

print("\nBatch 2 labels saved successfully!")