import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load golden evaluation set
df = pd.read_csv("data/golden/labeling_sample.csv")

# Actual labels
y_true = df["intent"]

# Majority class
majority_class = y_true.value_counts().idxmax()

# Predict the same class for every example
y_pred = [majority_class] * len(y_true)

# Results
accuracy = accuracy_score(y_true, y_pred)

print("===================================")
print("MAJORITY CLASS BASELINE")
print("===================================")

print("Majority class:", majority_class)
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))