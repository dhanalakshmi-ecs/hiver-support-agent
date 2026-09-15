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
    "delivery_issue",        # 2
    "product_device_issue",   # 3
    "product_device_issue",   # 4
    "delivery_issue",        # 5
    "delivery_issue",        # 6
    "other",                  # 7
    "other",                  # 8
    "delivery_issue",        # 9
    "delivery_issue",        # 10
    "delivery_issue",        # 11
    "other",                  # 12
    "other",                  # 13
    "delivery_issue",        # 14
    "product_device_issue",  # 15
    "other",                  # 16
    "delivery_issue",        # 17
    "delivery_issue",        # 18
    "other",                  # 19
    "other",                  # 20
    "delivery_issue",        # 21
    "cancellation_issue",    # 22
    "cancellation_issue",    # 23
    "delivery_issue",        # 24
    "delivery_issue",        # 25
    "product_device_issue",  # 26
    "other",                  # 27
    "delivery_issue",        # 28
    "delivery_issue",        # 29
    "other",                  # 30
    "delivery_issue",        # 31
    "other",                  # 32
    "delivery_issue",        # 33
    "order_issue",            # 34
    "refund_return",         # 35
    "refund_return",         # 36
    "account_issue",         # 37
    "product_device_issue",  # 38
    "product_device_issue",  # 39
    "delivery_issue",        # 40
    "delivery_issue",        # 41
    "delivery_issue",        # 42
    "other",                  # 43
    "other",                  # 44
    "delivery_issue"         # 45
]
if len(labels) != len(unlabeled):
    raise ValueError("Number of labels does not match number of examples.")

df.loc[unlabeled.index, "intent"] = labels

df.to_csv("data/processed/training_sample.csv", index=False)

print("Batch 7 labels saved successfully!")