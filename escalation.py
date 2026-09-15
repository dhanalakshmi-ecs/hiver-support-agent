def decide_escalation(message, intent, confidence):
    message_lower = str(message).lower()

    # 1. HIGH-RISK SECURITY / ACCOUNT / PAYMENT CONCERNS
    high_risk_keywords = [
        "fraud",
        "fraudulent",
        "hacked",
        "hack",
        "stolen",
        "unauthorized",
        "unauthorised",
        "without my permission",
        "someone used my account",
        "someone accessed my account",
        "someone got into my account",
        "someone has accessed my account",
        "someone has access to my account",
        "account was compromised",
        "account compromised",
        "account has been compromised",
        "my account was hacked",
        "my account has been hacked",
        "identity theft",
        "someone used my card",
        "charged twice",
        "charged me without",
        "payment fraud",
        "payment information was changed",
        "payment info was changed",
        "changed my payment information",
        "changed my payment info",
        "scam",
        "not a subscriber",
        "don't recognize this charge",
        "do not recognize this charge",
        "unrecognized charge",
        "unrecognised charge",
        "unauthorized transaction",
        "unauthorised transaction",
        "someone changed my password",
        "password was changed",
        "email was changed",
        "phone number was changed"
    ]

    # 2. CUSTOMER EXPLICITLY REQUESTS HUMAN SUPPORT
    human_request_keywords = [
        "call me",
        "phone call",
        "speak to someone",
        "speak to a person",
        "human",
        "agent",
        "representative",
        "contact me"
    ]

    # 3. PROBLEM IS REPEATED OR UNRESOLVED
    unresolved_keywords = [
        "still not",
        "still hasn't",
        "still have not",
        "still showing",
        "not resolved",
        "not fixed",
        "already contacted",
        "already filled",
        "multiple times",
        "4th time",
        "5x",
        "nobody came",
        "no response",
        "no one responded",
        "failed to",
        "false promise",
        "promised",
        "unresolved",
        "isn't the first time",
        "second time",
        "2nd time",
        "again"
    ]

    # 4. URGENT REQUESTS
    urgency_keywords = [
        "asap",
        "immediately",
        "urgent",
        "urgently",
        "right away",
        "immediate resolution",
        "resolve this asap",
        "need this resolved"
    ]

    # 5. SERIOUS DELIVERY / PRODUCT PROBLEMS
    serious_keywords = [
        "damaged",
        "broken",
        "left outside",
        "left in rain",
        "threw",
        "chucked",
        "wrong product",
        "wrong item",
        "wrong edition",
        "package damaged"
    ]

    # 6. STRONG CUSTOMER DISSATISFACTION
    frustration_keywords = [
        "idiot",
        "bullshit",
        "fucking",
        "unprofessional",
        "ridiculous",
        "worst service",
        "terrible service",
        "don't care",
        "not acceptable",
        "serious complaint",
        "blatant lying",
        "lying",
        "liar",
        "screwup",
        "screw up",
        "customer service is not capable"
    ]

    # ---------------------------------------------------------
    # ESCALATION RULES
    # Priority matters: security comes first.
    # ---------------------------------------------------------

    # Rule 1: Security / payment risk
    if any(k in message_lower for k in high_risk_keywords):
        return {
            "decision": "ESCALATE",
            "reason": "High-risk security or payment concern."
        }

    # Rule 2: Account-related issues
    if intent == "account_issue":
        return {
            "decision": "ESCALATE",
            "reason": "Account-related issue requires human verification."
        }

    # Rule 3: Customer explicitly wants a human
    if any(k in message_lower for k in human_request_keywords):
        return {
            "decision": "ESCALATE",
            "reason": "Customer explicitly requested human support."
        }

    # Rule 4: Repeated / unresolved issue
    if any(k in message_lower for k in unresolved_keywords):
        return {
            "decision": "ESCALATE",
            "reason": "Issue appears unresolved or repeated."
        }

    # Rule 5: Urgent issue
    if any(k in message_lower for k in urgency_keywords):
        return {
            "decision": "ESCALATE",
            "reason": "Customer indicates an urgent need for resolution."
        }

    # Rule 6: Serious product / delivery problem
    if any(k in message_lower for k in serious_keywords):
        return {
            "decision": "ESCALATE",
            "reason": "Potentially serious delivery or product issue."
        }

    # Rule 7: Strong dissatisfaction
    if any(k in message_lower for k in frustration_keywords):
        return {
            "decision": "ESCALATE",
            "reason": "Strong customer dissatisfaction requires human handling."
        }

    # Rule 8: Low classifier confidence
    if confidence < 0.25:
        return {
            "decision": "ESCALATE",
            "reason": "Low intent-classification confidence."
        }

    # Otherwise automatically handle
    return {
        "decision": "AUTO-HANDLE",
        "reason": "Routine issue with no strong escalation signal."
    }