import pandas as pd

file_path = "data/golden/labeling_sample.csv"

df = pd.read_csv(file_path)

# Make text columns able to accept string values
df["intent"] = df["intent"].astype("object")
df["label_notes"] = df["label_notes"].astype("object")

labels = {
    31: "other",
    32: "other",
    33: "product_device_issue",
    34: "other",
    35: "other",
    36: "delivery_issue",
    37: "delivery_issue",
    38: "other",
    39: "delivery_issue",
    40: "other",
    41: "delivery_issue",
    42: "other",
    43: "prime_issue",
    44: "order_issue",
    45: "delivery_issue",
    46: "other",
    47: "delivery_issue",
    48: "other",
    49: "other",
    50: "delivery_issue"
}

notes = {
    31: "Resolution requested but specific issue is unclear.",
    32: "Underlying order problem is unclear.",
    33: "Technical/device troubleshooting issue.",
    34: "General complaint about customer service.",
    35: "Resolution requested without specific issue.",
    36: "Delayed delivery and missed promised date.",
    37: "Shipping/delivery date changed.",
    38: "Return mentioned but exact underlying issue is unclear.",
    39: "Package location unknown/not received.",
    40: "Submitted details and awaiting help.",
    41: "Expected delivery date passed with no tracking.",
    42: "Repeatedly submitting information; issue unclear.",
    43: "Prime-member pre-order discount issue.",
    44: "Problem involving an order and received item.",
    45: "Delivery-person behavior complaint.",
    46: "Message is too vague to identify an intent.",
    47: "Explicit request for package delivery.",
    48: "General complaint without specific issue.",
    49: "No resolution after automated emails; issue unclear.",
    50: "Delivery-related problem was resolved."
}

for example_number, intent in labels.items():
    index = example_number - 1

    df.loc[index, "intent"] = intent
    df.loc[index, "label_notes"] = notes[example_number]

df.to_csv(file_path, index=False)

print("Examples 31-50 labeled successfully!")

print("\nCurrent labeling progress:")
print(df["intent"].notna().sum(), "of", len(df), "labeled")