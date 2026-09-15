import pandas as pd

df = pd.read_csv("data/processed/training_sample.csv")

unlabeled = df[df["intent"].isna()].head(50)

print("Number of examples:", len(unlabeled))
print("\nExamples:\n")

for i, (_, row) in enumerate(unlabeled.iterrows(), start=1):
    print(f"{i}. ID: {row['customer_tweet_id']}")
    print(f"   {row['customer_message']}")
    print()

# Add labels here after I review the printed examples
labels = [
    "other",                  # 1
    "other",                  # 2
    "other",                  # 3
    "other",                  # 4
    "other",                  # 5
    "other",                  # 6
    "other",                  # 7
    "delivery_issue",         # 8
    "refund_return",          # 9
    "delivery_issue",         # 10
    "delivery_issue",         # 11
    "delivery_issue",         # 12
    "order_issue",            # 13
    "delivery_issue",         # 14
    "delivery_issue",         # 15
    "other",                  # 16
    "other",                  # 17
    "delivery_issue",         # 18
    "other",                  # 19
    "other",                  # 20
    "delivery_issue",         # 21
    "refund_return",          # 22
    "payment_issue",          # 23
    "other",                  # 24
    "product_device_issue",   # 25
    "refund_return",          # 26
    "delivery_issue",         # 27
    "other",                  # 28
    "product_device_issue",   # 29
    "other",                  # 30
    "delivery_issue",         # 31
    "delivery_issue",         # 32
    "delivery_issue",         # 33
    "refund_return",          # 34
    "delivery_issue",         # 35
    "delivery_issue",         # 36
    "delivery_issue",         # 37
    "prime_issue",            # 38
    "other",                  # 39
    "other",                  # 40
    "delivery_issue",         # 41
    "delivery_issue",         # 42
    "order_issue",            # 43
    "delivery_issue",         # 44
    "delivery_issue",         # 45
    "prime_issue",            # 46
    "product_device_issue",   # 47
    "delivery_issue",         # 48
    "other",                  # 49
    "product_device_issue"    # 50
]

if len(labels) != len(unlabeled):
    raise ValueError("Number of labels does not match number of examples.")

df.loc[unlabeled.index, "intent"] = labels

df.to_csv("data/processed/training_sample.csv", index=False)

print("Batch 5 labels saved successfully!")