import pandas as pd

from retriever import retrieve_similar_cases


INPUT_FILE = "outputs/reply_evaluation_samples.csv"
OUTPUT_FILE = "outputs/reply_evaluation_samples.csv"

TOP_K = 3


def format_historical_examples(retrieved_cases):

    examples = []

    for i, (_, case) in enumerate(
        retrieved_cases.iterrows(),
        start=1
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
    print("REPAIRING HISTORICAL EXAMPLES")
    print("=" * 70)

    # Load existing evaluation samples
    df = pd.read_csv(INPUT_FILE)

    print(f"\nExisting examples: {len(df)}")

    repaired_examples = []

    for index, row in df.iterrows():

        customer_message = str(row["customer_message"])

        print(
            f"\nProcessing {index + 1}/{len(df)}"
        )

        # Retrieve historical cases locally
        retrieved_cases = retrieve_similar_cases(
            customer_message,
            top_k=TOP_K
        )

        historical_examples = format_historical_examples(
            retrieved_cases
        )

        repaired_examples.append(historical_examples)

    # Replace ONLY the broken column
    df["historical_examples"] = repaired_examples

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("REPAIR COMPLETE")
    print("=" * 70)

    print(f"\nSaved: {OUTPUT_FILE}")

    print("\nFirst example:")
    print(df.iloc[0]["historical_examples"])


if __name__ == "__main__":
    main()