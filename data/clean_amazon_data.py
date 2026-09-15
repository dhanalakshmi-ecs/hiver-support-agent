import pandas as pd
import re

input_file = "data/processed_amazon_conversations.csv"
output_file = "data/processed/amazon_clean.csv"

df = pd.read_csv(input_file)

print("Original conversations:", len(df))


# Remove empty messages
df = df.dropna(subset=["customer_message", "amazon_reply"])


# Remove duplicate conversations
df = df.drop_duplicates(
    subset=["customer_message", "amazon_reply"]
)


# Remove very short customer messages
df = df[df["customer_message"].str.len() >= 20]


# Keep mostly English messages
def is_english(text):
    english_chars = len(re.findall(r"[a-zA-Z]", str(text)))
    total_chars = len(re.sub(r"\s", "", str(text)))

    if total_chars == 0:
        return False

    return (english_chars / total_chars) >= 0.5


df = df[df["customer_message"].apply(is_english)]


# Remove obvious thank-you / greeting-only messages
remove_words = [
    "thank you",
    "thanks",
    "you're welcome",
    "good morning",
    "good evening",
    "good night"
]


def is_useful_message(text):
    text = text.lower()

    for phrase in remove_words:
        if text.strip().replace("!", "").replace(".", "") == phrase:
            return False

    return True


df = df[df["customer_message"].apply(is_useful_message)]


# Save cleaned dataset
df.to_csv(output_file, index=False)

print("Clean conversations:", len(df))
print("Removed:", len(pd.read_csv(input_file)) - len(df))

print("\nSample cleaned conversations:")

print(
    df[
        ["customer_message", "amazon_reply"]
    ].head(20).to_string(index=False)
)

print("\nSaved to:", output_file)