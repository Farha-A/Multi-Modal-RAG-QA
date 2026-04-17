"""
Qdrant Indexer
==============
Embeds document page images with ColPali and upserts the resulting
multi-vector embeddings into a persistent local Qdrant collection.

From the article:
    "Stores the resulting visual embeddings.  Unlike standard dense
    embeddings, these are multi-vector, requiring a database capable
    of handling late-interaction scoring."

Collection config: size=128, COSINE distance, MAX_SIM comparator.
"""

import os
from uuid import uuid4

import torch
from PIL import Image
from tqdm import tqdm

from pipeline.config import BASE_DIR, QDRANT_COLLECTION_NAME


class QdrantIndexer:
    """
    Indexes document page embeddings into Qdrant.

    Uses a persistent local Qdrant instance (no Docker/server needed).
    """

    def __init__(self, collection_name: str = QDRANT_COLLECTION_NAME):
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        db_path = BASE_DIR / "Data" / "qdrant_db"
        os.makedirs(db_path, exist_ok=True)
        self.client = QdrantClient(path=str(db_path))  # Persistent local storage
        self._collection_created = self.client.collection_exists(self.collection_name)

    def create_collection(self):
        """
        Create the Qdrant collection with multi-vector configuration.

        From the article & reference implementation:
        - Vector size = 128 (ColQwen2 patch embedding dimension)
        - Distance = COSINE
        - Multi-vector comparator = MAX_SIM (late-interaction scoring)
        """
        from qdrant_client import models

        if self._collection_created:
            return

        print(f"\n  Creating Qdrant collection: '{self.collection_name}'")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                on_disk=False,
            ),
            on_disk_payload=False,
        )
        self._collection_created = True
        print(f"  Collection created (128-dim, COSINE, MaxSim)")

    def index_pages(self, model, processor, pages: list[dict], batch_size: int = 2):
        """
        Embed document page images and upsert them into Qdrant.

        From the article: "The core engine splits the document image into
        patches and uses a Vision Transformer to generate contextualized,
        multi-vector embeddings."

        Args:
            model: ColQwen2 model
            processor: ColQwen2 processor
            pages: List of dicts from PDFImageConverter.convert_all()
            batch_size: Number of images to embed at once (small for CPU)
        """
        from qdrant_client import models

        self.create_collection()

        print(f"\n{'='*60}")
        print(f"  Indexing {len(pages)} pages into Qdrant")
        print(f"  Batch size: {batch_size} (optimized for CPU)")
        print(f"{'='*60}")

        total_batches = (len(pages) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(0, len(pages), batch_size),
                              total=total_batches,
                              desc="  Embedding & indexing"):
            batch_pages = pages[batch_idx:batch_idx + batch_size]

            # Load images
            batch_images = []
            for page in batch_pages:
                img = Image.open(page["image_path"]).convert("RGB")
                batch_images.append(img)

            # Generate multi-vector embeddings
            with torch.inference_mode():
                processed = processor.process_images(batch_images).to(model.device)
                embeddings = model(**processed)

            # Upsert into Qdrant
            points = []
            for offset, (page, embedding) in enumerate(zip(batch_pages, embeddings)):
                vector = embedding.cpu().float().numpy().tolist()
                point = models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "document": page["pdf_name"],
                        "page": page["page_num"],
                        "chunk_id": page.get("chunk_id", ""),
                        "chunk_type": page.get("chunk_type", ""),
                        "text": page.get("text", ""),
                        "image_path": page["image_path"],
                    },
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        # Report stats
        collection_info = self.client.get_collection(self.collection_name)
        print(f"  ✓ Indexed {collection_info.points_count} page embeddings")

    def get_client(self):
        """Return the Qdrant client for querying."""
        return self.client
