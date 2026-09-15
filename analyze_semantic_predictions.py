import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer


# Load data
df = pd.read_csv("data/processed/training_sample.csv")

X = df["customer_message"].astype(str)
y = df["intent"].astype(str)


# Same split as before
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Load embedding model
print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Create embeddings
print("Creating embeddings...")

X_train_embeddings = model.encode(
    X_train.tolist(),
    show_progress_bar=True
)

X_test_embeddings = model.encode(
    X_test.tolist(),
    show_progress_bar=True
)


# Train classifier
classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(
    X_train_embeddings,
    y_train
)


# Predictions
y_pred = classifier.predict(X_test_embeddings)


# Create results table
results = pd.DataFrame({
    "message": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred
})


# Keep only mistakes
mistakes = results[
    results["actual"] != results["predicted"]
]


print("\n========================================")
print("SEMANTIC MODEL ERROR ANALYSIS")
print("========================================")

print("Total test examples:", len(results))
print("Incorrect predictions:", len(mistakes))


print("\nFirst 30 mistakes:")
print("----------------------------------------")

for i, row in mistakes.head(30).iterrows():

    print("\nMessage:")
    print(row["message"])

    print("Actual:", row["actual"])
    print("Predicted:", row["predicted"])

    print("----------------------------------------")


# Save mistakes
mistakes.to_csv(
    "outputs/semantic_mistakes.csv",
    index=False
)

print("\nMistakes saved to:")
print("outputs/semantic_mistakes.csv")