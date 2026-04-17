"""
ColPali Retriever
=================
Embeds a natural-language query with ColPali and retrieves the top-k
most relevant document pages from Qdrant using MaxSim scoring.

From the article:
    "The system compares the query embeddings to document patches using
    a MaxSim (Maximum Similarity) mechanism, which is a hallmark of the
    ColBERT/ColPali late-interaction strategy."
"""

import torch

from pipeline.config import QDRANT_COLLECTION_NAME, TOP_K


class ColPaliRetriever:
    """
    Retrieves the most relevant document pages for a given query.

    Uses MaxSim late-interaction scoring against multi-vector embeddings
    stored in Qdrant.
    """

    def __init__(self, qdrant_client, model, processor,
                 collection_name: str = QDRANT_COLLECTION_NAME):
        self.client = qdrant_client
        self.model = model
        self.processor = processor
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embed the query and retrieve the top-k most relevant pages.

        From the article:
        - "The user's text query is encoded using the same model to ensure
          it resides in the same latent space as the document images."
        - "The system compares the query embeddings to document patches
          using MaxSim."

        Args:
            query: Natural language query string
            top_k: Number of results to return

        Returns:
            List of dicts with keys: document, page, image_path, score
        """
        from qdrant_client import models

        # Embed the query
        with torch.inference_mode():
            processed_query = self.processor.process_queries([query]).to(self.model.device)
            query_embedding = self.model(**processed_query)

        # Search with MaxSim
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].cpu().float().tolist(),
            limit=top_k,
            search_params=models.SearchParams(hnsw_ef=128, exact=False),
        )

        # Format results
        retrieved = []
        for point in results.points:
            retrieved.append({
                "document": point.payload["document"],
                "page": point.payload["page"],
                "chunk_id": point.payload.get("chunk_id", ""),
                "chunk_type": point.payload.get("chunk_type", "Chunk"),
                "text": point.payload.get("text", ""),
                "image_path": point.payload["image_path"],
                "score": point.score,
            })

        return retrieved
