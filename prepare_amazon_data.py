import pandas as pd

file_path = "data/raw/twcs.csv"

# Load dataset
df = pd.read_csv(file_path, nrows=100000)

# Get AmazonHelp replies
amazon_replies = df[df["author_id"] == "AmazonHelp"].copy()

# Remove rows where Amazon did not directly reply to another tweet
amazon_replies = amazon_replies.dropna(subset=["in_response_to_tweet_id"])

# Convert the ID so that it matches tweet_id
amazon_replies["in_response_to_tweet_id"] = (
    amazon_replies["in_response_to_tweet_id"].astype(int)
)

# Keep only the columns we need from Amazon's replies
amazon_replies = amazon_replies[
    ["tweet_id", "in_response_to_tweet_id", "text"]
]

amazon_replies = amazon_replies.rename(
    columns={
        "tweet_id": "amazon_reply_id",
        "in_response_to_tweet_id": "customer_tweet_id",
        "text": "amazon_reply"
    }
)

# Get customer tweets
customer_tweets = df[df["inbound"] == True].copy()

customer_tweets = customer_tweets[
    ["tweet_id", "author_id", "text"]
]

customer_tweets = customer_tweets.rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "author_id": "customer_id",
        "text": "customer_message"
    }
)

# Connect customer message with Amazon reply
conversations = customer_tweets.merge(
    amazon_replies,
    on="customer_tweet_id",
    how="inner"
)

# Keep useful columns
conversations = conversations[
    [
        "customer_tweet_id",
        "customer_id",
        "customer_message",
        "amazon_reply_id",
        "amazon_reply"
    ]
]

# Remove duplicate pairs
conversations = conversations.drop_duplicates()

# Save
output_path = "data/processed_amazon_conversations.csv"

conversations.to_csv(output_path, index=False)

print("Conversation pairs created:", len(conversations))
print("\nSaved to:", output_path)

print("\nSample conversations:")
print(
    conversations[
        ["customer_message", "amazon_reply"]
    ].head(10).to_string(index=False)
)