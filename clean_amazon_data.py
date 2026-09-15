import pandas as pd
import re
import os

input_file = "data/processed_amazon_conversations.csv"
output_dir = "data/processed"
output_file = os.path.join(output_dir, "amazon_clean.csv")

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)
# Load data
df = pd.read_csv(input_file)

print("Original rows:", len(df))

# Remove missing values
df = df.dropna(subset=["customer_message", "amazon_reply"])

# Remove duplicate conversation pairs
df = df.drop_duplicates(
    subset=["customer_message", "amazon_reply"]
)

# Remove very short customer messages
df = df[df["customer_message"].str.len() >= 20]


# Keep mostly English messages
def is_english(text):
    text = str(text)

    letters = re.findall(r"[A-Za-z]", text)

    if len(text) == 0:
        return False

    return len(letters) / len(text) >= 0.5


df = df[df["customer_message"].apply(is_english)]


# Remove simple acknowledgements/greetings
noise_phrases = {
    "thank you",
    "thanks",
    "thank u",
    "ok",
    "okay",
    "hi",
    "hello",
    "great thanks",
    "thanks amazon"
}


def is_noise(text):
    cleaned = re.sub(
        r"[^a-zA-Z ]",
        "",
        str(text).lower()
    ).strip()

    return cleaned in noise_phrases


df = df[~df["customer_message"].apply(is_noise)]


# Save cleaned dataset
df.to_csv(output_file, index=False)


print("Clean rows:", len(df))
print("Removed rows:", 6893 - len(df))

print("\nSample cleaned conversations:\n")


for _, row in df.head(20).iterrows():

    print("CUSTOMER:", row["customer_message"])
    print("AMAZON:  ", row["amazon_reply"])

    print("-" * 80)


print("\nSaved to:", output_file)