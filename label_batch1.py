import pandas as pd

file = "data/golden/labeling_sample.csv"

df = pd.read_csv(file)

df["intent"] = df["intent"].astype("object")

labels = {
    13130: "other",
    26975: "other",
    120150: "prime_issue",
    41530: "product_device_issue",
    13919: "delivery_issue",
    39583: "delivery_issue",
    45539: "delivery_issue",
    77405: "other",
    93541: "delivery_issue",
    76039: "delivery_issue",
    26880: "payment_issue",
    91494: "other",
    61047: "refund_return",
    24955: "delivery_issue",
    38557: "other",
    36497: "product_device_issue",
    91238: "other",
    10629: "product_device_issue",
    25889: "other",
    105244: "delivery_issue",
    108170: "other",
    64151: "delivery_issue",
    93620: "delivery_issue",
    5137: "delivery_issue",
    28431: "order_issue",
    115854: "other",
    86233: "other",
    31361: "other",
    17190: "other",
    64083: "delivery_issue"
}

for tweet_id, intent in labels.items():
    df.loc[df["customer_tweet_id"] == tweet_id, "intent"] = intent

df.to_csv(file, index=False)

print("Batch 1 labeled successfully.")
print()
print(df["intent"].value_counts(dropna=False))