import pandas as pd

file = "data/golden/labeling_sample.csv"

df = pd.read_csv(file)

# Make sure intent column can store text
df["intent"] = df["intent"].astype("object")

labels = {
    52939: "refund_return",
    43275: "product_device_issue",
    58131: "delivery_issue",
    2594: "other",
    43673: "delivery_issue",
    71996: "other",
    24533: "prime_issue",
    69672: "order_issue",
    27846: "other",
    9770: "other",
    16805: "cancellation_issue",
    33261: "other",
    108153: "other",
    89153: "delivery_issue",
    98697: "delivery_issue",
    86126: "delivery_issue",
    108248: "other",
    21904: "product_device_issue",
    26906: "other",
    103121: "other",
    5789: "delivery_issue",
    73417: "delivery_issue",
    126085: "other",
    64100: "order_issue",
    83340: "other",
    115931: "cancellation_issue",
    32314: "refund_return",
    64166: "other",
    77533: "other",
    10636: "delivery_issue",
}

for tweet_id, intent in labels.items():
    df.loc[df["customer_tweet_id"] == tweet_id, "intent"] = intent

df.to_csv(file, index=False)

print("Batch 5 labeled successfully.")
print("\nCurrent label counts:")
print(df["intent"].value_counts(dropna=False))