import pandas as pd

file = "data/golden/labeling_sample.csv"

df = pd.read_csv(file)

# Make sure intent column can store text
df["intent"] = df["intent"].astype("object")

labels = {
    25012: "other",
    124740: "other",
    69657: "other",
    69806: "order_issue",
    120187: "other",
    121070: "delivery_issue",
    34494: "order_issue",
    115873: "other",
    122002: "other",
    105063: "delivery_issue",
    17193: "prime_issue",
    91439: "delivery_issue",
    21393: "payment_issue",
    49155: "other",
    51592: "other",
    75916: "delivery_issue",
    65869: "refund_return",
    43265: "product_device_issue",
    26928: "delivery_issue",
    22172: "refund_return",
    96786: "other",
    74845: "delivery_issue",
    49997: "other",
    83528: "delivery_issue",
    22472: "other",
    25881: "payment_issue",
    18122: "delivery_issue",
    109868: "other",
    73589: "other",
    21025: "other"
}

for tweet_id, intent in labels.items():
    df.loc[df["customer_tweet_id"] == tweet_id, "intent"] = intent

df.to_csv(file, index=False)

print("Batch 6 labeled successfully.")
print("\nCurrent label counts:")
print(df["intent"].value_counts(dropna=False))