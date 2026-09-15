import pandas as pd

file = "data/golden/labeling_sample.csv"

df = pd.read_csv(file)

# Make sure intent column can store text
df["intent"] = df["intent"].astype("object")

labels = {
    108201: "order_issue",
    66193: "refund_return",
    99774: "delivery_issue",
    21551: "other",
    16107: "delivery_issue",
    59540: "delivery_issue",
    23208: "delivery_issue",
    65787: "other",
    108072: "delivery_issue",
    29577: "delivery_issue",
    77373: "order_issue",
    105059: "delivery_issue",
    98810: "other",
    14298: "delivery_issue",
    64042: "delivery_issue",
    89091: "other",
    18897: "refund_return",
    13558: "delivery_issue",
    91341: "other",
    52167: "account_issue",
    80742: "other",
    126119: "payment_issue",
    38838: "prime_issue",
    29488: "other",
    28439: "other",
    24947: "other",
    69750: "other",
    104999: "delivery_issue",
    107288: "prime_issue",
    36348: "delivery_issue",
}

for tweet_id, intent in labels.items():
    df.loc[df["customer_tweet_id"] == tweet_id, "intent"] = intent

df.to_csv(file, index=False)

print("Batch 4 labeled successfully.")
print("\nCurrent label counts:")
print(df["intent"].value_counts(dropna=False))