import os
import pandas as pd

from retriever import retrieve_similar_cases
from response_generator import generate_response


INPUT_FILE = "data/golden/labeling_sample.csv"
OUTPUT_FILE = "outputs/reply_evaluation_samples.csv"

SAMPLE_SIZE = 30
TOP_K = 3


def format_historical_examples(retrieved_cases):
    """
    Convert retrieved DataFrame rows into readable historical examples.
    """

    examples = []

    for i, (_, case) in enumerate(
        retrieved_cases.iterrows(), start=1
    ):

        examples.append(
            f"""Historical Case {i}
Customer: {case['customer_message']}
Amazon Reply: {case['amazon_reply']}
Similarity: {float(case['similarity']):.3f}"""
        )

    return "\n\n".join(examples)


def main():

    print("=" * 70)
    print("REPLY GENERATION EVALUATION SAMPLE")
    print("=" * 70)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found."
        )

    # Load golden dataset
    df = pd.read_csv(INPUT_FILE)

    print(f"\nGolden examples available: {len(df)}")

    # Select 30 examples
    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=42
    ).copy()

    # Output columns
    results = []

    for index, row in sample.iterrows():

        customer_message = str(row["customer_message"])
        intent = str(row["intent"])

        print("\n" + "-" * 70)
        print(f"Example {len(results) + 1}/{len(sample)}")
        print(f"Customer: {customer_message}")
        print(f"Intent: {intent}")

        # Retrieve historical Amazon cases
        retrieved_cases = retrieve_similar_cases(
            customer_message,
            top_k=TOP_K
        )

        # Convert retrieved cases into readable text
        historical_examples = format_historical_examples(
            retrieved_cases
        )

        print("\nRetrieved historical cases:")
        print(historical_examples)

        # Generate response
        generated_reply = generate_response(
            customer_message,
            intent,
            retrieved_cases
        )

        print("\nGenerated reply:")
        print(generated_reply)

        results.append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_message": customer_message,
            "intent": intent,
            "historical_examples": historical_examples,
            "generated_reply": generated_reply
        })

        # Save after every example
        # This protects progress if the API fails later.
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )

        print("\nSaved progress.")

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print(f"Saved {len(results)} examples to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()