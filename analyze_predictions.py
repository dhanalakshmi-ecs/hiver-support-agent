import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load data
df = pd.read_csv("data/processed/training_sample.csv")

X = df["customer_message"].astype(str)
y = df["intent"].astype(str)


# Same split as our classifier
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# Predictions
y_pred = model.predict(X_test_tfidf)


# Create results table
results = pd.DataFrame({
    "message": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred
})


# Keep only wrong predictions
errors = results[results["actual"] != results["predicted"]]


print("Total test examples:", len(results))
print("Incorrect predictions:", len(errors))

print("\nFirst 30 mistakes:")
print(errors.head(30).to_string(index=False))