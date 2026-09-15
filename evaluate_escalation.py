import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from escalation import decide_escalation


# --------------------------------------------------
# 1. Load golden predictions
# --------------------------------------------------

print("Loading golden intent predictions...")

df = pd.read_csv(
    "outputs/golden_intent_predictions.csv"
)

print(
    "Golden examples:",
    len(df)
)


# --------------------------------------------------
# 2. Define expected escalation policy
# --------------------------------------------------

def expected_escalation(message, gold_intent):

    message = str(message).lower()

    high_risk_keywords = [
        "fraud",
        "hacked",
        "stolen",
        "unauthorized",
        "someone used my card",
        "someone used my account",
        "identity theft",
        "charged twice",
        "charged me without",
        "payment fraud"
    ]

    # High-risk issues should go to humans
    for keyword in high_risk_keywords:
        if keyword in message:
            return "ESCALATE"

    # Account-specific problems should go to humans
    if gold_intent == "account_issue":
        return "ESCALATE"

    # Ambiguous / other issues should go to humans
    if gold_intent == "other":
        return "ESCALATE"

    # Routine known issues can potentially be auto-handled
    return "AUTO-HANDLE"


# --------------------------------------------------
# 3. Run escalation policy
# --------------------------------------------------

expected = []
predicted = []

records = []

for _, row in df.iterrows():

    customer_message = str(
        row["customer_message"]
    )

    predicted_intent = str(
        row["predicted_intent"]
    )

    gold_intent = str(
        row["intent"]
    )

    confidence = float(
        row["confidence"]
    )

    # Expected/reference decision
    expected_decision = expected_escalation(
        customer_message,
        gold_intent
    )

    # Actual system decision
    result = decide_escalation(
        customer_message,
        predicted_intent,
        confidence
    )

    predicted_decision = result["decision"]

    expected.append(
        expected_decision
    )

    predicted.append(
        predicted_decision
    )

    records.append({
        "customer_message": customer_message,
        "gold_intent": gold_intent,
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "expected_decision": expected_decision,
        "predicted_decision": predicted_decision,
        "reason": result["reason"]
    })


# --------------------------------------------------
# 4. Calculate metrics
# --------------------------------------------------

accuracy = accuracy_score(
    expected,
    predicted
)

precision = precision_score(
    expected,
    predicted,
    pos_label="ESCALATE",
    zero_division=0
)

recall = recall_score(
    expected,
    predicted,
    pos_label="ESCALATE",
    zero_division=0
)

f1 = f1_score(
    expected,
    predicted,
    pos_label="ESCALATE",
    zero_division=0
)


# --------------------------------------------------
# 5. Print results
# --------------------------------------------------

print("\n========================================")
print("ESCALATION POLICY EVALUATION")
print("========================================")

print(
    "Golden examples:",
    len(df)
)

print("\nRESULTS")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)


# --------------------------------------------------
# 6. Decision distributions
# --------------------------------------------------

print("\nExpected escalation distribution:")

print(
    pd.Series(expected).value_counts()
)


print("\nPredicted escalation distribution:")

print(
    pd.Series(predicted).value_counts()
)


# --------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    expected,
    predicted,
    labels=[
        "AUTO-HANDLE",
        "ESCALATE"
    ]
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual AUTO-HANDLE",
        "Actual ESCALATE"
    ],
    columns=[
        "Predicted AUTO-HANDLE",
        "Predicted ESCALATE"
    ]
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm_df)


# --------------------------------------------------
# 8. Save detailed results
# --------------------------------------------------

results_df = pd.DataFrame(records)

results_df.to_csv(
    "outputs/escalation_evaluation.csv",
    index=False
)

print(
    "\nSaved detailed results to:"
)

print(
    "outputs/escalation_evaluation.csv"
)


# --------------------------------------------------
# 9. Show disagreements
# --------------------------------------------------

disagreements = results_df[
    results_df["expected_decision"]
    != results_df["predicted_decision"]
]

print("\n========================================")
print("DISAGREEMENTS")
print("========================================")

print(
    "Total disagreements:",
    len(disagreements)
)

print("\nFirst 10 disagreements:")

print(
    disagreements[
        [
            "customer_message",
            "gold_intent",
            "predicted_intent",
            "confidence",
            "expected_decision",
            "predicted_decision",
            "reason"
        ]
    ].head(10).to_string(index=False)
)