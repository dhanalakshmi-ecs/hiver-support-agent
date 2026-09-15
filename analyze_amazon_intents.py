import pandas as pd
import re

# Load cleaned data
df = pd.read_csv("data/processed/amazon_clean.csv")

print("Total rows:", len(df))
print("\n" + "=" * 60)
print("KEYWORD ANALYSIS")
print("=" * 60)

keywords = {
    "delivery": [
        "delivery", "delivered", "delivery date",
        "package", "parcel", "courier", "shipping"
    ],
    "order": [
        "order", "ordered", "order status",
        "pre-order", "preorder"
    ],
    "payment": [
        "payment", "paid", "charge", "charged",
        "credit card", "debit card", "billing"
    ],
    "refund": [
        "refund", "refunded", "money back",
        "reimburse"
    ],
    "return": [
        "return", "returning", "send back"
    ],
    "prime": [
        "prime", "membership", "subscription"
    ],
    "account": [
        "account", "login", "password",
        "sign in", "locked"
    ],
    "kindle": [
        "kindle", "fire tv", "echo", "alexa"
    ],
    "technical": [
        "error", "not working", "doesn't work",
        "doesnt work", "problem", "issue",
        "broken", "crash"
    ],
    "cancellation": [
        "cancel", "cancellation", "cancelled",
        "canceled"
    ]
}

# Combine customer messages
text = df["customer_message"].fillna("").str.lower()

results = []

for category, words in keywords.items():

    count = 0

    for message in text:
        if any(word in message for word in words):
            count += 1

    results.append((category, count))

results.sort(key=lambda x: x[1], reverse=True)

print("\nPossible intent frequencies:\n")

for category, count in results:
    percentage = count / len(df) * 100

    print(
        f"{category:15} : "
        f"{count:5} messages "
        f"({percentage:.1f}%)"
    )


print("\n" + "=" * 60)
print("EXAMPLE MESSAGES")
print("=" * 60)

# Show examples for each category
for category, words in results:

    print(f"\n\n--- {category.upper()} ---")

    matches = df[
        text.apply(
            lambda x: any(word in x for word in keywords[category])
        )
    ]

    if len(matches) > 0:

        samples = matches.sample(
            min(5, len(matches)),
            random_state=42
        )

        for message in samples["customer_message"]:
            print("-", message)