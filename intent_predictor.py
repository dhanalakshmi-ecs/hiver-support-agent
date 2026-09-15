import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# Load training data
# --------------------------------------------------

DATA_PATH = "data/processed/training_sample.csv"

print("Loading training data...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["customer_message", "intent"]
).reset_index(drop=True)

print("Training examples:", len(df))


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading sentence transformer...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Create embeddings
# --------------------------------------------------

print("Creating training embeddings...")

texts = (
    df["customer_message"]
    .astype(str)
    .tolist()
)

X = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

y = df["intent"]


# --------------------------------------------------
# Train classifier
# --------------------------------------------------

print("Training intent classifier...")

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(X, y)

print("Intent classifier ready!")


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_intent(customer_message):

    embedding = model.encode(
        [customer_message],
        normalize_embeddings=True
    )

    prediction = classifier.predict(embedding)[0]

    probabilities = classifier.predict_proba(embedding)[0]

    confidence = max(probabilities)

    return prediction, confidence


# --------------------------------------------------
# Test independently
# --------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("INTENT PREDICTOR")
    print("=" * 60)

    message = input(
        "\nEnter customer message:\n"
    )

    intent, confidence = predict_intent(message)

    print("\nPredicted intent:", intent)

    print(
        "Confidence:",
        round(confidence, 3)
    )