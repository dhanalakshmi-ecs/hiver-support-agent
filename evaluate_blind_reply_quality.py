import os
import json
import pandas as pd
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

# -------------------------------------------------
# 1. Check API key
# -------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found.")

client = OpenAI()

# -------------------------------------------------
# 2. Files
# -------------------------------------------------

input_file = "outputs/blind_human_reply_samples.csv"
output_file = "outputs/blind_llm_judge_evaluation.csv"

# -------------------------------------------------
# 3. Load blind samples
# -------------------------------------------------

df = pd.read_csv(input_file)

print("Blind samples:", len(df))

# -------------------------------------------------
# 4. LLM judge
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

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    return json.loads(text)


# -------------------------------------------------
# 5. Run judge
# -------------------------------------------------

batch = []

for _, row in df.iterrows():

    batch.append({
        "customer_tweet_id": str(row["customer_tweet_id"]),
        "customer_message": str(row["customer_message"]),
        "intent": str(row["intent"]),
        "historical_examples": str(row["historical_examples"]),
        "generated_reply": str(row["generated_reply"])
    })

print("\nRunning LLM judge on the 6 blind samples...")

judge_output = judge_batch(batch)

evaluations = judge_output["evaluations"]

if len(evaluations) != len(batch):
    raise ValueError("Judge returned incorrect number of evaluations.")

# -------------------------------------------------
# 6. Save LLM results
# -------------------------------------------------

rows = []

for item, evaluation in zip(batch, evaluations):

    rows.append({
        "customer_tweet_id": item["customer_tweet_id"],
        "customer_message": item["customer_message"],
        "intent": item["intent"],
        "generated_reply": item["generated_reply"],
        "Correctness": evaluation["correctness"],
        "Relevance": evaluation["relevance"],
        "Grounding": evaluation["grounding"],
        "Helpfulness": evaluation["helpfulness"],
        "No_Hallucination": evaluation["no_hallucination"],
        "Reason": evaluation["reason"]
    })

judge_df = pd.DataFrame(rows)

judge_df.to_csv(output_file, index=False)

print("\nLLM judge results saved to:")
print(output_file)

print("\nLLM JUDGE SCORES")
print("=" * 70)
print(
    judge_df[
        [
            "customer_tweet_id",
            "Correctness",
            "Relevance",
            "Grounding",
            "Helpfulness",
            "No_Hallucination"
        ]
    ].to_string(index=False)
)

# -------------------------------------------------
# 7. Human ratings
# -------------------------------------------------

human_ratings = {
    "69715": [5, 5, 4, 5, 5],
    "86233": [5, 5, 4, 5, 5],
    "26906": [4, 5, 4, 4, 5],
    "21393": [3, 4, 2, 3, 2],
    "74845": [5, 5, 4, 3, 5],
    "113742": [5, 4, 4, 5, 5]
}

dimensions = [
    "Correctness",
    "Relevance",
    "Grounding",
    "Helpfulness",
    "No_Hallucination"
]

# -------------------------------------------------
# 8. Compare human vs LLM judge
# -------------------------------------------------

comparison_rows = []

for _, row in judge_df.iterrows():

    sample_id = str(row["customer_tweet_id"])
    human = human_ratings[sample_id]

    for i, dimension in enumerate(dimensions):

        llm_score = int(row[dimension])
        human_score = int(human[i])

        comparison_rows.append({
            "customer_tweet_id": sample_id,
            "dimension": dimension,
            "human_score": human_score,
            "llm_score": llm_score,
            "exact_agreement": human_score == llm_score
        })

comparison_df = pd.DataFrame(comparison_rows)

# -------------------------------------------------
# 9. Agreement metrics
# -------------------------------------------------

agreement_rows = []

for dimension in dimensions:

    x = comparison_df[
        comparison_df["dimension"] == dimension
    ]

    human_scores = x["human_score"].tolist()
    llm_scores = x["llm_score"].tolist()

    exact_agreement = x["exact_agreement"].mean()

    # Unweighted Cohen's kappa
    kappa = cohen_kappa_score(
        human_scores,
        llm_scores
    )

    agreement_rows.append({
        "dimension": dimension,
        "samples": len(x),
        "exact_agreement": round(exact_agreement, 3),
        "cohen_kappa": round(kappa, 3)
    })

agreement_df = pd.DataFrame(agreement_rows)

agreement_file = "outputs/blind_human_llm_agreement.csv"

agreement_df.to_csv(
    agreement_file,
    index=False
)

print("\n" + "=" * 70)
print("HUMAN vs LLM JUDGE AGREEMENT")
print("=" * 70)

print(
    agreement_df.to_string(index=False)
)

print("\nAgreement results saved to:")
print(agreement_file)