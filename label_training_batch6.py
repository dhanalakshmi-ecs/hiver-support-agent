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
    "delivery_issue",         # 3
    "other",                  # 4
    "delivery_issue",         # 5
    "other",                  # 6
    "delivery_issue",         # 7
    "other",                  # 8
    "other",                  # 9
    "order_issue",            # 10
    "delivery_issue",         # 11
    "other",                  # 12
    "delivery_issue",         # 13
    "delivery_issue",         # 14
    "other",                  # 15
    "delivery_issue",         # 16
    "other",                  # 17
    "delivery_issue",         # 18
    "cancellation_issue",     # 19
    "other",                  # 20
    "other",                  # 21
    "other",                  # 22
    "product_device_issue",   # 23
    "prime_issue",            # 24
    "delivery_issue",         # 25
    "delivery_issue",         # 26
    "payment_issue",          # 27
    "delivery_issue",         # 28
    "other",                  # 29
    "account_issue",          # 30
    "other",                  # 31
    "refund_return",          # 32
    "refund_return",          # 33
    "order_issue",            # 34
    "cancellation_issue",     # 35
    "other",                  # 36
    "order_issue",            # 37
    "delivery_issue",         # 38
    "refund_return",          # 39
    "other",                  # 40
    "product_device_issue",   # 41
    "other",                  # 42
    "delivery_issue",         # 43
    "delivery_issue",         # 44
    "delivery_issue",         # 45
    "other",                  # 46
    "other",                  # 47
    "delivery_issue",         # 48
    "delivery_issue",         # 49
    "delivery_issue"          # 50
]

if len(labels) != len(unlabeled):
    raise ValueError("Number of labels does not match number of examples.")

df.loc[unlabeled.index, "intent"] = labels

df.to_csv("data/processed/training_sample.csv", index=False)

print("Batch 6 labels saved successfully!")