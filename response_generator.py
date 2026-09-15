import os
from openai import OpenAI


# Check API key
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY not found. Please set your API key first."
    )

client = OpenAI()


def generate_response(customer_message, predicted_intent, similar_cases):

    # Build historical context
    historical_context = ""

    for i, case in enumerate(similar_cases, start=1):

        # Pandas row
        if hasattr(case, "get"):
            customer_text = case.get("customer_message", "")
            amazon_reply = case.get("amazon_reply", "")
            similarity = case.get("similarity", "")

        # Tuple/list
        elif isinstance(case, (list, tuple)):
            customer_text = case[0] if len(case) > 0 else ""
            amazon_reply = case[1] if len(case) > 1 else ""
            similarity = case[2] if len(case) > 2 else ""

        else:
            customer_text = str(case)
            amazon_reply = ""
            similarity = ""

        historical_context += f"""
Historical Case {i}
Customer: {customer_text}
Amazon Reply: {amazon_reply}
Similarity: {similarity}
"""


    prompt = f"""
You are an AI customer-support assistant for Amazon.

Your task is to draft a helpful customer-support response.

Customer message:
{customer_message}

Predicted intent:
{predicted_intent}

Here are similar historical Amazon support conversations:

{historical_context}

Instructions:

1. Answer the customer's actual problem.
2. Use the historical conversations as evidence for the type of response Amazon typically gives.
3. Do not invent order numbers, refunds, delivery dates, or account information.
4. Do not claim that you performed an action that you cannot actually perform.
5. If the issue requires account-specific investigation, politely direct the customer to Amazon support.
6. Keep the response concise and professional.
7. Do not mention that you are an AI.
8. Do not mention the historical cases.

Generate only the customer-facing reply.
"""


    print("\nGenerating AI response...")

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text


# Test this file independently
if __name__ == "__main__":

    print("=" * 70)
    print("AMAZON AI SUPPORT RESPONSE GENERATOR")
    print("=" * 70)

    from retriever import retrieve_similar_cases

    customer_message = input(
        "\nEnter customer message:\n"
    )

    predicted_intent = input(
        "\nEnter predicted intent:\n"
    )

    similar_cases = retrieve_similar_cases(
        customer_message,
        top_k=3
    )

    try:

        answer = generate_response(
            customer_message,
            predicted_intent,
            similar_cases
        )

        print("\n" + "=" * 70)
        print("AI GENERATED RESPONSE")
        print("=" * 70)

        print(answer)

    except Exception as e:

        print("\nERROR:")
        print(e)