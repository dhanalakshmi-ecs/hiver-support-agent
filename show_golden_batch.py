import pandas as pd

file_path = "data/golden/labeling_sample.csv"

df = pd.read_csv(file_path)

# Show rows 31 to 50
batch = df.iloc[30:50]

for i, row in batch.iterrows():
    print("\n" + "=" * 80)
    print(f"Example {i + 1}")
    print("=" * 80)
    print("Customer message:")
    print(row["customer_message"])
    print("\nAmazon reply:")
    print(row["amazon_reply"])
    print("\nCurrent intent:", row["intent"])