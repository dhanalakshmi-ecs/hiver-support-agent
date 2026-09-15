import pandas as pd

file = "data/golden/labeling_sample.csv"

df = pd.read_csv(file)

# Make sure intent column can store text
df["intent"] = df["intent"].astype("object")

labels = {
    33248: "other",
    9790: "order_issue",
    113798: "prime_issue",
    24963: "other",
    69702: "delivery_issue",
    22193: "prime_issue",
    52232: "order_issue",
    16877: "other",
    121964: "other",
    22203: "cancellation_issue",
    80706: "other",
    18985: "cancellation_issue",
    25296: "prime_issue",
    27908: "order_issue",
    54370: "refund_return",
    22154: "other",
    99908: "delivery_issue",
    69746: "other",
    114373: "other",
    96850: "delivery_issue"
}

for tweet_id, intent in labels.items():
    df.loc[df["customer_tweet_id"] == tweet_id, "intent"] = intent

df.to_csv(file, index=False)

print("Batch 7 labeled successfully.")
print("\nFinal label counts:")
print(df["intent"].value_counts(dropna=False))