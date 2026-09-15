import pandas as pd

file_path = "data/golden/labeling_sample.csv"

df = pd.read_csv(file_path)

print("Total examples:", len(df))
print("Labeled examples:", df["intent"].notna().sum())
print("Unlabeled examples:", df["intent"].isna().sum())

print("\nIntent distribution:")
print(df["intent"].value_counts())

print("\nFirst 10 examples:")
print(df[["customer_message", "intent"]].head(10).to_string(index=True))