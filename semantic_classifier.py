import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load training data
# --------------------------------------------------

df = pd.read_csv("data/processed/training_sample.csv")

X = df["customer_message"].astype(str)
y = df["intent"].astype(str)


# --------------------------------------------------
# 2. Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Load sentence embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 4. Convert messages into embeddings
# --------------------------------------------------

print("Creating embeddings...")

X_train_embeddings = model.encode(
    X_train.tolist(),
    show_progress_bar=True
)

X_test_embeddings = model.encode(
    X_test.tolist(),
    show_progress_bar=True
)

print("Embeddings created.")


# --------------------------------------------------
# 5. Train classifier
# --------------------------------------------------

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(
    X_train_embeddings,
    y_train
)


# --------------------------------------------------
# 6. Predict
# --------------------------------------------------

y_pred = classifier.predict(X_test_embeddings)


# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("Semantic Embeddings + Logistic Regression")
print("========================================")

print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)