import pandas as pd
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/processed/amazon_clean.csv"
EMBEDDING_PATH = "outputs/amazon_embeddings.npy"


print("Loading Amazon historical conversations...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["customer_message", "amazon_reply"]
).reset_index(drop=True)

print("Historical conversations:", len(df))


print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Load existing embeddings if available
# --------------------------------------------------

if os.path.exists(EMBEDDING_PATH):

    print("Loading saved embeddings...")

    embeddings = np.load(EMBEDDING_PATH)

    print("Embeddings loaded:", embeddings.shape)


else:

    print("Creating embeddings for the first time...")

    customer_texts = (
        df["customer_message"]
        .astype(str)
        .tolist()
    )

    embeddings = model.encode(
        customer_texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(
        EMBEDDING_PATH,
        embeddings
    )

    print("Embeddings saved to:", EMBEDDING_PATH)

    print("Embeddings shape:", embeddings.shape)


# --------------------------------------------------
# Retrieval function
# --------------------------------------------------

def retrieve_similar_cases(query, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[top_indices].copy()

    results["similarity"] = scores[top_indices]

    return results[
        [
            "customer_message",
            "amazon_reply",
            "similarity"
        ]
    ]


# --------------------------------------------------
# Test the retriever
# --------------------------------------------------

if __name__ == "__main__":

    query = input(
        "\nEnter a customer message:\n"
    )

    results = retrieve_similar_cases(
        query,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("TOP HISTORICAL SIMILAR CASES")
    print("=" * 70)

    for _, row in results.iterrows():

        print("\nCustomer:")
        print(row["customer_message"])

        print("\nAmazon reply:")
        print(row["amazon_reply"])

        print(
            "\nSimilarity:",
            round(row["similarity"], 3)
        )

        print("-" * 70)