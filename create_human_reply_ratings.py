import pandas as pd
import re

INPUT_FILE = "outputs/reply_quality_evaluation.csv"
OUTPUT_FILE = "outputs/human_reply_ratings.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only rows that have an LLM judge result
df = df[df["judge_result"].notna()].copy()


def extract_score(text, field):
    """Extract a score like 'Correctness: 4' from judge_result."""
    match = re.search(
        rf"{field}\s*:\s*([1-5])",
        str(text),
        re.IGNORECASE
    )
    return int(match.group(1)) if match else None


# Extract the LLM judge scores
df["llm_correctness"] = df["judge_result"].apply(
    lambda x: extract_score(x, "Correctness")
)

df["llm_relevance"] = df["judge_result"].apply(
    lambda x: extract_score(x, "Relevance")
)

df["llm_grounding"] = df["judge_result"].apply(
    lambda x: extract_score(x, "Grounding")
)

df["llm_helpfulness"] = df["judge_result"].apply(
    lambda x: extract_score(x, "Helpfulness")
)

df["llm_no_hallucination"] = df["judge_result"].apply(
    lambda x: extract_score(x, "No_Hallucination")
)


# Create empty columns for your human ratings
df["human_correctness"] = ""
df["human_relevance"] = ""
df["human_grounding"] = ""
df["human_helpfulness"] = ""
df["human_no_hallucination"] = ""
df["human_notes"] = ""


result = df[
    [
        "customer_tweet_id",
        "customer_message",
        "intent",
        "generated_reply",

        "llm_correctness",
        "llm_relevance",
        "llm_grounding",
        "llm_helpfulness",
        "llm_no_hallucination",

        "human_correctness",
        "human_relevance",
        "human_grounding",
        "human_helpfulness",
        "human_no_hallucination",
        "human_notes",
    ]
]

result.to_csv(OUTPUT_FILE, index=False)

print("Human rating file created successfully!")
print(f"File: {OUTPUT_FILE}")
print(f"Rows to review: {len(result)}")
print()
print("LLM scores extracted:")
print(result[
    [
        "llm_correctness",
        "llm_relevance",
        "llm_grounding",
        "llm_helpfulness",
        "llm_no_hallucination"
    ]
].to_string(index=False))

print()
print("Open the CSV and fill the human_* columns with scores from 1 to 5.")