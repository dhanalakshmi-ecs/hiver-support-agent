import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Load training data
# -----------------------------

train_df = pd.read_csv("data/processed/training_sample.csv")

X = train_df["customer_message"].astype(str)
y = train_df["intent"].astype(str)


# -----------------------------
# 2. Load golden evaluation set
# -----------------------------

golden_df = pd.read_csv("data/golden/labeling_sample.csv")

X_golden = golden_df["customer_message"].astype(str)
y_golden = golden_df["intent"].astype(str)


# -----------------------------
# 3. Same train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# MODEL 1: MAJORITY BASELINE
# =====================================================

majority_class = y_train.value_counts().idxmax()

majority_predictions = [majority_class] * len(y_golden)

majority_accuracy = accuracy_score(
    y_golden,
    majority_predictions
)

majority_macro_f1 = f1_score(
    y_golden,
    majority_predictions,
    average="macro",
    zero_division=0
)

print("\n==============================")
print("MAJORITY BASELINE")
print("==============================")

print("Majority class:", majority_class)
print("Golden Accuracy:", round(majority_accuracy, 4))
print("Golden Macro F1:", round(majority_macro_f1, 4))


# =====================================================
# MODEL 2: TF-IDF + LOGISTIC REGRESSION
# =====================================================

print("\n==============================")
print("TF-IDF MODEL")
print("==============================")

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_golden_tfidf = tfidf.transform(X_golden)

tfidf_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

tfidf_model.fit(X_train_tfidf, y_train)

tfidf_predictions = tfidf_model.predict(X_golden_tfidf)

tfidf_accuracy = accuracy_score(
    y_golden,
    tfidf_predictions
)

tfidf_macro_f1 = f1_score(
    y_golden,
    tfidf_predictions,
    average="macro",
    zero_division=0
)

print("Golden Accuracy:", round(tfidf_accuracy, 4))
print("Golden Macro F1:", round(tfidf_macro_f1, 4))


# =====================================================
# MODEL 3: SENTENCE TRANSFORMER + LOGISTIC REGRESSION
# =====================================================

print("\n==============================")
print("SEMANTIC MODEL")
print("==============================")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating training embeddings...")

X_train_embeddings = embedding_model.encode(
    X_train.tolist(),
    show_progress_bar=True
)

print("Creating golden embeddings...")

X_golden_embeddings = embedding_model.encode(
    X_golden.tolist(),
    show_progress_bar=True
)

semantic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

semantic_model.fit(
    X_train_embeddings,
    y_train
)

semantic_predictions = semantic_model.predict(
    X_golden_embeddings
)

semantic_accuracy = accuracy_score(
    y_golden,
    semantic_predictions
)

semantic_macro_f1 = f1_score(
    y_golden,
    semantic_predictions,
    average="macro",
    zero_division=0
)

print("Golden Accuracy:", round(semantic_accuracy, 4))
print("Golden Macro F1:", round(semantic_macro_f1, 4))


# =====================================================
# FINAL COMPARISON
# =====================================================

results = pd.DataFrame({
    "Model": [
        "Majority Baseline",
        "TF-IDF + Logistic Regression",
        "Sentence Transformer + Logistic Regression"
    ],
    "Accuracy": [
        majority_accuracy,
        tfidf_accuracy,
        semantic_accuracy
    ],
    "Macro F1": [
        majority_macro_f1,
        tfidf_macro_f1,
        semantic_macro_f1
    ]
})

print("\n==============================")
print("FINAL GOLDEN SET COMPARISON")
print("==============================")

print(results.to_string(index=False))

results.to_csv(
    "outputs/baseline_comparison.csv",
    index=False
)

print("\nSaved:")
print("outputs/baseline_comparison.csv")