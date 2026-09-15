import pandas as pd

file_path = "data/raw/twcs.csv"

df = pd.read_csv(file_path, nrows=100000)

amazon = df[df["author_id"] == "AmazonHelp"].copy()

print("Amazon replies:", len(amazon))

# Get the tweets that Amazon replied to
customer_tweet_ids = amazon["in_response_to_tweet_id"].dropna().astype(int)

customers = df[df["tweet_id"].isin(customer_tweet_ids)]

print("Customer messages found:", len(customers))

print("\nCustomer messages:")
print(
    customers[
        ["tweet_id", "author_id", "inbound", "text"]
    ].head(30).to_string(index=False)
)