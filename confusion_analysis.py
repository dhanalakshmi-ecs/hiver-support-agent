import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load training data
df = pd.read_csv("data/processed/training_sample.csv")

X = df["customer_message"].astype(str)
y = df["intent"].astype(str)

# Train/test split
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

# Model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Confusion matrix
labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nConfusion Matrix:")
print(cm_df)

print("\nMost common mistakes:")

mistakes = []

for actual in labels:
    for predicted in labels:
        if actual != predicted and cm_df.loc[actual, predicted] > 0:
            mistakes.append(
                (
                    actual,
                    predicted,
                    cm_df.loc[actual, predicted]
                )
            )

mistakes = sorted(
    mistakes,
    key=lambda x: x[2],
    reverse=True
)

for actual, predicted, count in mistakes:
    print(
        f"{actual} -> {predicted}: {count}"
    )