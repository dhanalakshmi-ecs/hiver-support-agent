import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load training data
# --------------------------------------------------

print("Loading training data...")

train_df = pd.read_csv(
    "data/processed/training_sample.csv"
)

X = train_df["customer_message"].astype(str)
y = train_df["intent"].astype(str)


# --------------------------------------------------
# 2. Create the same train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------

print("Creating training embeddings...")

X_train_embeddings = model.encode(
    X_train.tolist(),
    show_progress_bar=True
)

print("Creating test embeddings...")

X_test_embeddings = model.encode(
    X_test.tolist(),
    show_progress_bar=True
)

print("Embeddings created.")


# --------------------------------------------------
# 5. Train classifier
# --------------------------------------------------

print("Training classifier...")

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(
    X_train_embeddings,
    y_train
)


# --------------------------------------------------
# 6. Evaluate on original test set
# --------------------------------------------------

y_pred = classifier.predict(X_test_embeddings)

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n========================================")
print("TRAINING TEST SET RESULTS")
print("========================================")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Macro F1:",
    round(macro_f1, 4)
)

print(
    "Weighted F1:",
    round(weighted_f1, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 7. Load golden evaluation set
# --------------------------------------------------

print("\nLoading golden evaluation set...")

golden_df = pd.read_csv(
    "data/golden/labeling_sample.csv"
)

golden_messages = golden_df[
    "customer_message"
].astype(str)

golden_labels = golden_df[
    "intent"
].astype(str)


print(
    "Golden examples:",
    len(golden_df)
)


# --------------------------------------------------
# 8. Create golden-set embeddings
# --------------------------------------------------

print("Creating golden-set embeddings...")

golden_embeddings = model.encode(
    golden_messages.tolist(),
    show_progress_bar=True
)

print("Golden embeddings created.")


# --------------------------------------------------
# 9. Predict golden-set intents
# --------------------------------------------------

print("Predicting golden-set intents...")

golden_predictions = classifier.predict(
    golden_embeddings
)


# Get probability for every class
golden_probabilities = classifier.predict_proba(
    golden_embeddings
)


# Highest probability = model confidence
golden_confidences = golden_probabilities.max(
    axis=1
)


# --------------------------------------------------
# 10. Evaluate golden set
# --------------------------------------------------

golden_accuracy = accuracy_score(
    golden_labels,
    golden_predictions
)

golden_macro_f1 = f1_score(
    golden_labels,
    golden_predictions,
    average="macro"
)

golden_weighted_f1 = f1_score(
    golden_labels,
    golden_predictions,
    average="weighted"
)


print("\n========================================")
print("GOLDEN SET RESULTS")
print("========================================")

print(
    "Accuracy:",
    round(golden_accuracy, 4)
)

print(
    "Macro F1:",
    round(golden_macro_f1, 4)
)

print(
    "Weighted F1:",
    round(golden_weighted_f1, 4)
)


# --------------------------------------------------
# 11. Confidence statistics
# --------------------------------------------------

print("\n========================================")
print("GOLDEN SET CONFIDENCE")
print("========================================")

print(
    "Average confidence:",
    round(golden_confidences.mean(), 4)
)

print(
    "Minimum confidence:",
    round(golden_confidences.min(), 4)
)

print(
    "Maximum confidence:",
    round(golden_confidences.max(), 4)
)

print(
    "Median confidence:",
    round(pd.Series(golden_confidences).median(), 4)
)


# --------------------------------------------------
# 12. Classification report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        golden_labels,
        golden_predictions,
        zero_division=0
    )
)


# --------------------------------------------------
# 13. Confusion matrix
# --------------------------------------------------

labels = sorted(
    golden_labels.unique()
)

cm = confusion_matrix(
    golden_labels,
    golden_predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)


print("\n========================================")
print("GOLDEN SET CONFUSION MATRIX")
print("========================================")

print(cm_df)


# --------------------------------------------------
# 14. Save predictions + confidence
# --------------------------------------------------

golden_df["predicted_intent"] = (
    golden_predictions
)

golden_df["confidence"] = (
    golden_confidences
)

golden_df["correct"] = (
    golden_df["intent"]
    == golden_df["predicted_intent"]
)


golden_df.to_csv(
    "outputs/golden_intent_predictions.csv",
    index=False
)


print(
    "\nSaved predictions to:"
)

print(
    "outputs/golden_intent_predictions.csv"
)


# --------------------------------------------------
# 15. Display a few confidence examples
# --------------------------------------------------

print("\n========================================")
print("SAMPLE PREDICTIONS WITH CONFIDENCE")
print("========================================")

sample_output = golden_df[
    [
        "customer_message",
        "intent",
        "predicted_intent",
        "confidence",
        "correct"
    ]
].head(10)

print(sample_output.to_string(index=False))