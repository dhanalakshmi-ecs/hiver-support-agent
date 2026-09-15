import pandas as pd

file = "data/golden/labeling_sample.csv"

df = pd.read_csv(file)

# Make sure intent column can store text
df["intent"] = df["intent"].astype("object")

labels = {
    99806: "other",
    91251: "other",
    11064: "product_device_issue",
    77410: "other",
    76061: "other",
    124057: "delivery_issue",
    59550: "delivery_issue",
    99878: "product_device_issue",
    24558: "delivery_issue",
    69969: "other",
    31834: "delivery_issue",
    77427: "other",
    693: "prime_issue",
    11976: "order_issue",
    43646: "delivery_issue",
    3751: "other",
    35909: "delivery_issue",
    36539: "other",
    66177: "other",
    74619: "delivery_issue",
    48897: "other",
    28939: "delivery_issue",
    91370: "delivery_issue",
    46561: "order_issue",
    113742: "prime_issue",
    99933: "account_issue",
    93505: "delivery_issue",
    89257: "other",
    69727: "prime_issue",
    78833: "prime_issue",
}

for tweet_id, intent in labels.items():
    df.loc[df["customer_tweet_id"] == tweet_id, "intent"] = intent

df.to_csv(file, index=False)

print("Batch 2 labeled successfully.")
print("\nCurrent label counts:")
print(df["intent"].value_counts(dropna=False))