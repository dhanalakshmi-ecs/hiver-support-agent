import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load our labeled training data
df = pd.read_csv("data/processed/training_sample.csv")

# 2. Separate messages and labels
X = df["customer_message"]
y = df["intent"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Find the most common intent
majority_intent = y_train.value_counts().idxmax()

print("Majority intent:", majority_intent)

# 5. Predict the same intent for every test message
y_pred = [majority_intent] * len(y_test)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("\nBaseline Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)