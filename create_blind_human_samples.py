import os
import pandas as pd

from retriever import retrieve_similar_cases
from response_generator import generate_response


INPUT_FILE = "data/golden/labeling_sample.csv"
EXISTING_FILE = "outputs/reply_evaluation_samples.csv"
OUTPUT_FILE = "outputs/blind_human_reply_samples.csv"

SAMPLE_SIZE = 6
TOP_K = 3


def format_historical_examples(retrieved_cases):
    examples = []

    for i, (_, case) in enumerate(retrieved_cases.iterrows(), start=1):
        examples.append(
            f"""Historical Case {i}
Customer: {case['customer_message']}
Amazon Reply: {case['amazon_reply']}
Similarity: {float(case['similarity']):.3f}"""
        )

    return "\n\n".join(examples)


def main():

    print("=" * 70)
    print("BLIND HUMAN REPLY EVALUATION SAMPLE")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found.")

    golden = pd.read_csv(INPUT_FILE)
    existing = pd.read_csv(EXISTING_FILE)

    used_ids = set(existing["customer_tweet_id"].astype(str))

    available = golden[
        ~golden["customer_tweet_id"].astype(str).isin(used_ids)
    ].copy()

    print(f"\nGolden examples: {len(golden)}")
    print(f"Already evaluated: {len(used_ids)}")
    print(f"Available for blind evaluation: {len(available)}")

    if len(available) < SAMPLE_SIZE:
        raise RuntimeError("Not enough unused golden examples available.")

    # Fixed sample so that a retry always refers to the same six examples.
    sample = available.sample(
        n=SAMPLE_SIZE,
        random_state=2026
    ).copy()

    # ---------------------------------------------------------
    # RESUME FROM EXISTING BLIND RESULTS
    # ---------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):
        saved = pd.read_csv(OUTPUT_FILE)

        if len(saved) > 0:
            completed_ids = set(
                saved["customer_tweet_id"].astype(str)
            )

            print(
                f"Existing blind results found: "
                f"{len(completed_ids)}"
            )

            # Keep only examples belonging to our fixed six-example sample
            sample_ids = set(
                sample["customer_tweet_id"].astype(str)
            )

            saved = saved[
                saved["customer_tweet_id"].astype(str).isin(sample_ids)
            ].copy()

            completed_ids = set(
                saved["customer_tweet_id"].astype(str)
            )

            sample = sample[
                ~sample["customer_tweet_id"].astype(str).isin(
                    completed_ids
                )
            ].copy()

            results = saved.to_dict("records")

        else:
            results = []

    else:
        results = []

    print(
        f"Already generated in blind set: {len(results)}"
    )
    print(
        f"Remaining to generate: {len(sample)}"
    )

    # ---------------------------------------------------------
    # GENERATE ONLY MISSING EXAMPLES
    # ---------------------------------------------------------

    for number, (_, row) in enumerate(
        sample.iterrows(),
        start=len(results) + 1
    ):

        customer_message = str(row["customer_message"])
        intent = str(row["intent"])

        print("\n" + "-" * 70)
        print(f"Example {number}/{SAMPLE_SIZE}")
        print(f"Customer: {customer_message}")
        print(f"Intent: {intent}")

        retrieved_cases = retrieve_similar_cases(
            customer_message,
            top_k=TOP_K
        )

        historical_examples = format_historical_examples(
            retrieved_cases
        )

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