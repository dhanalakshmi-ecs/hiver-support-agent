import pandas as pd

input_file = "data/processed/training_sample.csv"
output_file = "data/processed/training_sample.csv"

df = pd.read_csv(input_file)

df["intent"] = df["intent"].astype("object")

batch = df[df["intent"].isna() | (df["intent"] == "")].head(50)

print("Number of examples in this batch:", len(batch))

print("\n========================================")
print("TRAINING LABELING BATCH 3")
print("========================================\n")

for i, (_, row) in enumerate(batch.iterrows(), start=1):
    print(f"{i}. ID: {row['customer_tweet_id']}")
    print(f"   Message: {row['customer_message']}")
    print()

labels = [
    "product_device_issue",  # 1 refurbished products
    "other",                 # 2 unclear
    "delivery_issue",        # 3 delayed delivery
    "other",                 # 4 casual/comment
    "account_issue",         # 5 account creation/help
    "delivery_issue",        # 6 package not arrived
    "other",                 # 7 insult/complaint without specific issue
    "other",                 # 8 generic support complaint
    "delivery_issue",        # 9 delivered status but not received
    "product_device_issue",  # 10 Fire TV + Echo integration
    "other",                 # 11 issue already contacted support
    "order_issue",           # 12 order stuck
    "delivery_issue",        # 13 product not delivered
    "other",                 # 14 customer service complaint
    "cancellation_issue",    # 15 order cancellation
    "other",                 # 16 asks to contact by phone
    "other",                 # 17 vague
    "other",                 # 18 asks corporate contact
    "account_issue",         # 19 ID-check/account issue
    "other",                 # 20 fraud/service complaint
    "order_issue",           # 21 shipment/order email issue
    "other",                 # 22 third-party seller
    "order_issue",           # 23 replacement issue
    "payment_issue",         # 24 bank/payment issue
    "delivery_issue",        # 25 damaged package
    "other",                 # 26 image/link only
    "account_issue",         # 27 linking multiple accounts
    "delivery_issue",        # 28 package delivered to neighbor
    "delivery_issue",        # 29 delivery disappointment
    "prime_issue",           # 30 lost Prime membership time
    "delivery_issue",        # 31 lost orders
    "delivery_issue",        # 32 package not received
    "product_device_issue",  # 33 product complaint
    "delivery_issue",        # 34 delivery person handling
    "delivery_issue",        # 35 out for delivery but delayed
    "delivery_issue",        # 36 Prime orders delayed
    "payment_issue",         # 37 cash load/payment
    "order_issue",            # 38 new orders
    "other",                 # 39 vague link/details
    "delivery_issue",        # 40 late delivery
    "payment_issue",         # 41 deducted pay balance
    "product_device_issue",  # 42 Fire device stopped working
    "prime_issue",           # 43 offer in Amazon app
    "other",                 # 44 "Por Amazon" unclear
    "delivery_issue",        # 45 lost/delayed/opened packages
    "other",                 # 46 quiz/results
    "delivery_issue",        # 47 carrier problem
    "delivery_issue",        # 48 shipping not dispatched
    "other",                 # 49 email/details sent
    "delivery_issue"         # 50 repeated delivery failure
]

if len(labels) != len(batch):
    raise ValueError("Number of labels does not match number of examples.")

for (_, row), label in zip(batch.iterrows(), labels):
    df.loc[
        df["customer_tweet_id"] == row["customer_tweet_id"],
        "intent"
    ] = label

df.to_csv(output_file, index=False)

print("\nBatch 3 labels saved successfully!")