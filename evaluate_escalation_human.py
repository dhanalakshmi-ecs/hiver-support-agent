import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from escalation import decide_escalation


# Load the 50 human-reviewed examples
path = "data/golden/escalation_sample.csv"
df = pd.read_csv(path)


# Apply the improved AI escalation policy
def get_ai_decision(row):
    message = str(row["customer_message"])
    intent = str(row["intent"])

    # For now, use confidence = 1.0 because this evaluation
    # is testing the escalation rules themselves.
    confidence = 1.0

    result = decide_escalation(
        message,
        intent,
        confidence
    )

    return result["decision"]


df["ai_decision"] = df.apply(get_ai_decision, axis=1)


# Human labels
y_true = df["escalation_label"].str.strip()

# AI predictions
y_pred = df["ai_decision"].str.strip()


# Metrics
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    pos_label="ESCALATE",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label="ESCALATE",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label="ESCALATE",
    zero_division=0
)


print("\n===== IMPROVED ESCALATION EVALUATION =====")

print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")


print("\n===== HUMAN LABELS =====")
print(y_true.value_counts())


print("\n===== AI DECISIONS =====")
print(y_pred.value_counts())


# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=["AUTO-HANDLE", "ESCALATE"]
)

print("\n===== CONFUSION MATRIX =====")
print("Rows = Human")
print("Columns = AI")
print()
print("                 AI AUTO    AI ESCALATE")
print(f"Human AUTO       {cm[0][0]:8d}    {cm[0][1]:8d}")
print(f"Human ESCALATE   {cm[1][0]:8d}    {cm[1][1]:8d}")


# Find disagreements
df["correct"] = y_true == y_pred

disagreements = df[df["correct"] == False]

print(f"\n===== DISAGREEMENTS: {len(disagreements)} =====")

for _, row in disagreements.iterrows():
    print("\nTweet ID:", row["customer_tweet_id"])
    print("Message:", row["customer_message"])
    print("Human:", row["escalation_label"])
    print("AI:", row["ai_decision"])


# Save evaluation results
output_path = "outputs/escalation_human_evaluation.csv"

df.to_csv(output_path, index=False)

print("\nResults saved to:")
print(output_path)