import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


print("Loading golden intent predictions...")

df = pd.read_csv("outputs/golden_intent_predictions.csv")

print(f"Golden examples: {len(df)}")

print("\nAvailable columns:")
print(df.columns.tolist())


# ----------------------------------------
# HIGH-RISK KEYWORDS
# ----------------------------------------

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


def has_high_risk(message):

    message = str(message).lower()

    return any(
        keyword in message
        for keyword in high_risk_keywords
    )


# ----------------------------------------
# EXPECTED ESCALATION POLICY
# ----------------------------------------

def expected_escalation(row):

    message = str(row["customer_message"])

    # IMPORTANT:
    # "intent" is the true/golden intent
    gold_intent = str(row["intent"])

    # High-risk situations
    if has_high_risk(message):
        return "ESCALATE"

    # Account problems should go to a human
    if gold_intent == "account_issue":
        return "ESCALATE"

    # "Other" is uncertain and should be reviewed
    if gold_intent == "other":
        return "ESCALATE"

    return "AUTO-HANDLE"


df["expected_decision"] = df.apply(
    expected_escalation,
    axis=1
)


# ----------------------------------------
# TEST DIFFERENT THRESHOLDS
# ----------------------------------------

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]


results = []


for threshold in thresholds:

    predictions = []

    for _, row in df.iterrows():

        confidence = float(row["confidence"])

        if confidence < threshold:
            decision = "ESCALATE"
        else:
            decision = "AUTO-HANDLE"

        predictions.append(decision)


    y_true = df["expected_decision"]

    y_pred = predictions


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


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


    auto_handle_count = predictions.count(
        "AUTO-HANDLE"
    )

    escalate_count = predictions.count(
        "ESCALATE"
    )


    results.append({
        "threshold": threshold,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auto_handle": auto_handle_count,
        "escalate": escalate_count
    })


# ----------------------------------------
# DISPLAY RESULTS
# ----------------------------------------

results_df = pd.DataFrame(results)


print("\n========================================")
print("ESCALATION THRESHOLD ANALYSIS")
print("========================================")

print(
    results_df.to_string(index=False)
)


# ----------------------------------------
# SAVE RESULTS
# ----------------------------------------

output_file = (
    "outputs/escalation_threshold_analysis.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print("\nSaved results to:")
print(output_file)