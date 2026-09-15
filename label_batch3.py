import pandas as pd

file = "data/golden/labeling_sample.csv"

df = pd.read_csv(file)

# Make sure intent column can store text
df["intent"] = df["intent"].astype("object")

labels = {
    9810: "refund_return",
    36547: "order_issue",
    93704: "product_device_issue",
    44503: "other",
    109810: "other",
    91722: "payment_issue",
    36527: "order_issue",
    76101: "other",
    664: "delivery_issue",
    91339: "other",
    69715: "delivery_issue",
    19130: "refund_return",
    29540: "refund_return",
    58099: "delivery_issue",
    74838: "order_issue",
    108076: "product_device_issue",
    27427: "other",
    86017: "refund_return",
    21880: "refund_return",
    74846: "cancellation_issue",
    18913: "delivery_issue",
    118838: "delivery_issue",
    47889: "cancellation_issue",
    71975: "delivery_issue",
    86208: "delivery_issue",
    43637: "delivery_issue",
    64062: "refund_return",
    64169: "refund_return",
    69804: "other",
    64034: "account_issue",
}

for tweet_id, intent in labels.items():
    df.loc[df["customer_tweet_id"] == tweet_id, "intent"] = intent

df.to_csv(file, index=False)

print("Batch 3 labeled successfully.")
print("\nCurrent label counts:")
print(df["intent"].value_counts(dropna=False))