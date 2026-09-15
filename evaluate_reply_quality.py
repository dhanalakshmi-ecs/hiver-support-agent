import os
import json
import pandas as pd
from openai import OpenAI


# -------------------------------------------------
# 1. Check API key
# -------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY not found."
    )

client = OpenAI()


# -------------------------------------------------
# 2. Files
# -------------------------------------------------

input_file = "outputs/reply_evaluation_samples.csv"
output_file = "outputs/reply_quality_evaluation.csv"


# -------------------------------------------------
# 3. Settings
# -------------------------------------------------

BATCH_SIZE = 5


# -------------------------------------------------
# 4. Load input
# -------------------------------------------------

df = pd.read_csv(input_file)

print("Total examples:", len(df))


# -------------------------------------------------
# 5. Load previous results
# -------------------------------------------------

if os.path.exists(output_file):

    results_df = pd.read_csv(output_file)

    if len(results_df) > 0:
        completed_ids = set(
            results_df["customer_tweet_id"].astype(str)
        )
    else:
        completed_ids = set()

    print("Already evaluated:", len(completed_ids))

else:

    results_df = pd.DataFrame(
        columns=[
            "customer_tweet_id",
            "customer_message",
            "intent",
            "generated_reply",
            "judge_result"
        ]
    )

    completed_ids = set()

    print("No previous evaluations found.")


# -------------------------------------------------
# 6. LLM judge
# -------------------------------------------------

def judge_batch(batch):

    prompt_parts = []

    for i, row in enumerate(batch, start=1):

        prompt_parts.append(
            f"""
EXAMPLE {i}

Customer message:
{row["customer_message"]}

Expected intent:
{row["intent"]}

Historical Amazon support examples:
{row["historical_examples"]}

AI-generated response:
{row["generated_reply"]}
"""
        )

    prompt = f"""
You are evaluating AI customer-support responses for Amazon.

Evaluate each example independently.

For every example, score these five criteria from 1 to 5:

1. Correctness
Does the response correctly address the customer's problem?

2. Relevance
Is the response focused on the customer's actual issue?

3. Historical grounding
Does the response follow the type of solution or guidance
shown in the historical Amazon examples?

4. Helpfulness
Does it provide a useful and realistic next step?

5. No hallucination
Does it avoid inventing facts, actions, refunds, dates,
order information, or unsupported promises?

Important:
- Evaluate only the response for that example.
- Do not assume the AI actually performed an action.
- Do not give credit merely because the response sounds polite.
- Use the historical examples when judging grounding.
- Be strict and consistent.

Return ONLY valid JSON.

Required format:

{{
  "evaluations": [
    {{
      "example": 1,
      "correctness": 1,
      "relevance": 1,
      "grounding": 1,
      "helpfulness": 1,
      "no_hallucination": 1,
      "reason": "short explanation"
    }}
  ]
}}

There must be exactly {len(batch)} evaluations.

Here are the examples:

{"".join(prompt_parts)}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    text = response.output_text.strip()

    # Remove markdown code fences if the model adds them
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    return json.loads(text)


# -------------------------------------------------
# 7. Prepare remaining examples
# -------------------------------------------------

remaining = df[
    ~df["customer_tweet_id"].astype(str).isin(
        completed_ids
    )
].copy()

print(
    "Remaining examples:",
    len(remaining)
)


# -------------------------------------------------
# 8. Process in batches
# -------------------------------------------------

for start in range(
    0,
    len(remaining),
    BATCH_SIZE
):

    batch_df = remaining.iloc[
        start:start + BATCH_SIZE
    ]

    batch = []

    for _, row in batch_df.iterrows():

        batch.append({
            "customer_tweet_id": str(
                row["customer_tweet_id"]
            ),
            "customer_message": str(
                row["customer_message"]
            ),
            "intent": str(
                row["intent"]
            ),
            "historical_examples": str(
                row["historical_examples"]
            ),
            "generated_reply": str(
                row["generated_reply"]
            )
        })

    print("\n" + "=" * 70)

    print(
        f"Evaluating batch "
        f"{start + 1}-"
        f"{start + len(batch)} "
        f"of {len(remaining)}"
    )

    print("=" * 70)

    try:

        judge_output = judge_batch(batch)

        evaluations = judge_output["evaluations"]

        if len(evaluations) != len(batch):

            raise ValueError(
                "Judge returned an incorrect number "
                "of evaluations."
            )

        new_rows = []

        for item, evaluation in zip(
            batch,
            evaluations
        ):

            judge_result = (
                f"Correctness: "
                f"{evaluation['correctness']}\n"
                f"Relevance: "
                f"{evaluation['relevance']}\n"
                f"Grounding: "
                f"{evaluation['grounding']}\n"
                f"Helpfulness: "
                f"{evaluation['helpfulness']}\n"
                f"No_Hallucination: "
                f"{evaluation['no_hallucination']}\n"
                f"Reason: "
                f"{evaluation['reason']}"
            )

            new_rows.append({
                "customer_tweet_id":
                    item["customer_tweet_id"],

                "customer_message":
                    item["customer_message"],

                "intent":
                    item["intent"],

                "generated_reply":
                    item["generated_reply"],

                "judge_result":
                    judge_result
            })

        new_result_df = pd.DataFrame(
            new_rows
        )

        results_df = pd.concat(
            [
                results_df,
                new_result_df
            ],
            ignore_index=True
        )

        results_df.to_csv(
            output_file,
            index=False
        )

        completed_ids.update(
            item["customer_tweet_id"]
            for item in batch
        )

        print(
            f"Batch saved successfully."
        )

        print(
            f"Total completed: "
            f"{len(completed_ids)}/{len(df)}"
        )

    except Exception as e:

        print("\nEvaluation stopped.")
        print("Reason:")
        print(e)

        print(
            f"\nSuccessfully evaluated so far: "
            f"{len(completed_ids)}/{len(df)}"
        )

        print(
            "\nYour progress has been saved."
        )

        print(
            "Run this script again after the "
            "API limit resets."
        )

        break


# -------------------------------------------------
# 9. Final status
# -------------------------------------------------

print("\n" + "=" * 70)
print("REPLY EVALUATION STATUS")
print("=" * 70)

print(
    f"Completed: "
    f"{len(completed_ids)}/{len(df)}"
)

print(
    f"Remaining: "
    f"{len(df) - len(completed_ids)}"
)

print(
    f"\nResults saved to:\n"
    f"{output_file}"
)