import pandas as pd

input_file = "data/processed/training_sample.csv"
output_file = "data/processed/training_sample.csv"

df = pd.read_csv(input_file)

df["intent"] = df["intent"].astype("object")

batch = df[df["intent"].isna() | (df["intent"] == "")].head(50)

print("Number of examples in this batch:", len(batch))

print("\n========================================")
print("TRAINING LABELING BATCH 4")
print("========================================\n")

for i, (_, row) in enumerate(batch.iterrows(), start=1):
    print(f"{i}. ID: {row['customer_tweet_id']}")
    print(f"   Message: {row['customer_message']}")
    print()

labels = [
    "refund_return",          # 1
    "other",                  # 2
    "delivery_issue",         # 3
    "refund_return",          # 4
    "delivery_issue",         # 5
    "product_device_issue",   # 6
    "delivery_issue",         # 7
    "prime_issue",            # 8
    "delivery_issue",         # 9
    "account_issue",          # 10
    "refund_return",          # 11
    "other",                  # 12
    "other",                  # 13
    "delivery_issue",         # 14
    "other",                  # 15
    "delivery_issue",         # 16
    "product_device_issue",   # 17
    "delivery_issue",         # 18
    "delivery_issue",         # 19
    "prime_issue",            # 20
    "other",                  # 21
    "other",                  # 22
    "delivery_issue",         # 23
    "product_device_issue",   # 24
    "other",                  # 25
    "delivery_issue",         # 26
    "delivery_issue",         # 27
    "prime_issue",            # 28
    "delivery_issue",         # 29
    "product_device_issue",   # 30
    "other",                  # 31
    "account_issue",          # 32
    "delivery_issue",         # 33
    "other",                  # 34
    "delivery_issue",         # 35
    "other",                  # 36
    "other",                  # 37
    "other",                  # 38
    "other",                  # 39
    "payment_issue",          # 40
    "account_issue",          # 41
    "other",                  # 42
    "cancellation_issue",     # 43
    "other",                  # 44
    "delivery_issue",         # 45
    "product_device_issue",   # 46
    "delivery_issue",         # 47
    "payment_issue",          # 48
    "order_issue",            # 49
    "refund_return"           # 50
]

if len(labels) != len(batch):
    raise ValueError("Number of labels does not match number of examples.")

for (_, row), label in zip(batch.iterrows(), labels):
    df.loc[
        df["customer_tweet_id"] == row["customer_tweet_id"],
        "intent"
    ] = label

df.to_csv(output_file, index=False)

print("\nBatch 4 labels saved successfully!")