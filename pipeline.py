
from intent_predictor import predict_intent
from retriever import retrieve_similar_cases
from response_generator import generate_response
from escalation import decide_escalation


def run_pipeline(customer_message):
    """
    Run the complete Hiver AI support-agent pipeline.

    Flow:
    1. Predict customer intent
    2. Retrieve similar historical Amazon cases
    3. Decide AUTO-HANDLE vs ESCALATE
    4. Generate a reply only when safe to auto-handle
    """

    customer_message = customer_message.strip()

    if not customer_message:
        raise ValueError("Customer message cannot be empty.")

    print("\n" + "=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 70)

    print("\nCustomer message:")
    print(customer_message)

    # --------------------------------------------------
    # STEP 1: Predict intent
    # --------------------------------------------------

    print("\n[1] Predicting customer intent...")

    predicted_intent, confidence = predict_intent(
        customer_message
    )

    print("Predicted intent:", predicted_intent)
    print("Confidence:", round(confidence, 3))

    # --------------------------------------------------
    # STEP 2: Retrieve similar historical cases
    # --------------------------------------------------

    print("\n[2] Searching historical Amazon conversations...")

    similar_cases = retrieve_similar_cases(
        customer_message,
        top_k=3
    )

    print("Retrieved", len(similar_cases), "similar cases.")

    # --------------------------------------------------
    # STEP 3: Decide whether human support is required
    # --------------------------------------------------

    print("\n[3] Checking whether human support is required...")

    decision = decide_escalation(
        customer_message,
        predicted_intent,
        confidence
    )

    print("Decision:", decision["decision"])
    print("Reason:", decision["reason"])

    # --------------------------------------------------
    # STEP 4: Generate response only for AUTO-HANDLE
    # --------------------------------------------------

    if decision["decision"] == "AUTO-HANDLE":

        print("\n[4] Generating AI support response...")

        response = generate_response(
            customer_message,
            predicted_intent,
            similar_cases
        )

    else:

        print("\n[4] Skipping AI response generation.")

        response = (
            "This issue should be reviewed by a human Amazon "
            "support agent because it requires additional investigation."
        )

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print("\nIntent:")
    print(predicted_intent)

    print("\nIntent confidence:")
    print(round(confidence, 3))

    print("\nSuggested Reply:")
    print(response)

    print("\nDecision:")
    print(decision["decision"])

    print("\nReason:")
    print(decision["reason"])

    # --------------------------------------------------
    # Return structured result
    # --------------------------------------------------

    return {
        "customer_message": customer_message,
        "intent": predicted_intent,
        "confidence": float(confidence),
        "similar_cases": similar_cases,
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
        "reply": response
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    customer_message = input(
        "\nEnter customer message:\n"
    )

    run_pipeline(customer_message)

