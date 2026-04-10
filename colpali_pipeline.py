"""
ColPali Multi-Modal RAG Pipeline
=================================
Based on "The King of Multi-Modal RAG: ColPali" by Juan Ovalle.

Architecture:
    1. PDF → Image Converter   (pdf2image, 300 DPI)
    2. Image Storage            (local disk)
    3. ColPali Model            (ColQwen2 via colpali-engine, CPU mode)
    4. Vector Database          (Qdrant in-memory, multi-vector MaxSim)
    5. Inference Pipeline       (query → retrieval → Gemini generation)

Usage:
    python colpali_pipeline.py index      # Index all PDFs in Data/PDFs/
    python colpali_pipeline.py query      # Interactive query mode
    python colpali_pipeline.py pipeline   # Full pipeline: index then query
"""

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

load_dotenv()

BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "Data" / "PDFs"
IMAGE_DIR = BASE_DIR / "Data" / "images"

COLPALI_MODEL_NAME = os.getenv("COLPALI_MODEL_NAME", "vidore/colqwen2-v1.0-hf")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "docbank_colpali")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TOP_K = int(os.getenv("TOP_K", "3"))


# ═════════════════════════════════════════════════════════════════════════════
# 1. ColPali Model Loader (CPU-Focused)
# ═════════════════════════════════════════════════════════════════════════════

class ColPaliModelLoader:
    """
    Loads ColQwen2 model and processor from HuggingFace.

    From the article: ColPali uses a Vision Language Model to generate
    multi-vector embeddings directly from document images, allowing the
    system to "see" the document rather than relying on OCR.

    Configured for CPU inference (no GPU required).
    """

    def __init__(self, model_name: str = COLPALI_MODEL_NAME):
        self.model_name = model_name
        self.device = "cpu"
        # Use float32 on CPU (bfloat16/float16 are GPU-optimized)
        self.dtype = torch.float32
        self.model = None
        self.processor = None

    def load(self):
        """Load both model and processor."""
        print(f"\n{'='*60}")
        print(f"  Loading ColPali Model: {self.model_name}")
        print(f"  Device: {self.device} | Dtype: {self.dtype}")
        print(f"{'='*60}")

        start = time.time()
        self._load_model()
        self._load_processor()
        elapsed = time.time() - start

        print(f"  Model loaded in {elapsed:.1f}s")
        return self.model, self.processor

    def _load_model(self):
        """
        Load the document retrieval model.
        Uses AutoModel to support any architecture (like ColSmol or ColQwen2).
        """
        if "colsmol" in self.model_name.lower() or "idefics" in self.model_name.lower():
            from colpali_engine.models import ColIdefics3 as ModelClass
        else:
            from transformers import AutoModel as ModelClass

        print("  Loading model weights (this may take a few minutes on CPU)...")
        self.model = ModelClass.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        ).eval()

    def _load_processor(self):
        """Load the processor for tokenizing queries and images."""
        if "colsmol" in self.model_name.lower() or "idefics" in self.model_name.lower():
            from colpali_engine.models import ColIdefics3Processor as ProcessorClass
        else:
            from transformers import AutoProcessor as ProcessorClass

        print("  Loading processor...")
        self.processor = ProcessorClass.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. PDF → Image Converter
# ═════════════════════════════════════════════════════════════════════════════

class PDFImageConverter:
    """
    Converts PDF pages to high-resolution images.

    From the article: "Uses pdf2image (with optimized DPI settings) to convert
    PDF pages into high-resolution images. This preserves the visual integrity
    of the original file."

    Output images are saved as JPEG files at 300 DPI.
    """

    def __init__(self, pdf_dir: Path = PDF_DIR, image_dir: Path = IMAGE_DIR, dpi: int = 300):
        self.pdf_dir = pdf_dir
        self.image_dir = image_dir
        self.dpi = dpi

    def convert_all(self) -> list[dict]:
        """
        Convert all PDFs in the source directory to page images.

        Returns:
            List of dicts with keys: pdf_name, page_num, image_path
        """
        from pdf2image import convert_from_path

        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"  No PDF files found in {self.pdf_dir}")
            return []

        print(f"\n{'='*60}")
        print(f"  Converting {len(pdf_files)} PDFs to images ({self.dpi} DPI)")
        print(f"{'='*60}")

        all_pages = []

        for pdf_path in tqdm(pdf_files, desc="  Processing PDFs"):
            pdf_name = pdf_path.stem
            output_dir = self.image_dir / pdf_name
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=self.dpi,
                    fmt="jpeg",
                    thread_count=4,
                )

                for page_num, img in enumerate(images, start=1):
                    img_path = output_dir / f"page_{page_num}.jpeg"
                    img.save(img_path, "JPEG", quality=95)

                    all_pages.append({
                        "pdf_name": pdf_name,
                        "page_num": page_num,
                        "image_path": str(img_path),
                    })

            except Exception as e:
                print(f"\n  ✗ Error converting {pdf_name}: {e}")

        print(f"  Converted {len(all_pages)} pages from {len(pdf_files)} PDFs")
        return all_pages


# ═════════════════════════════════════════════════════════════════════════════
# 3. Qdrant Indexer (In-Memory, Multi-Vector MaxSim)
# ═════════════════════════════════════════════════════════════════════════════

class QdrantIndexer:
    """
    Indexes document page embeddings into Qdrant.

    From the article: "Stores the resulting visual embeddings. Unlike standard
    dense embeddings, these are multi-vector, requiring a database capable of
    handling late-interaction scoring."

    Uses an in-memory Qdrant instance (no Docker/server needed).
    Collection config: size=128, COSINE distance, MAX_SIM comparator.
    """

    def __init__(self, collection_name: str = QDRANT_COLLECTION_NAME):
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")  # In-memory for local usage
        self._collection_created = False

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


# ═════════════════════════════════════════════════════════════════════════════
# 4. ColPali Retriever
# ═════════════════════════════════════════════════════════════════════════════

class ColPaliRetriever:
    """
    Retrieves the most relevant document pages for a given query.

    From the article: "The system compares the query embeddings to document
    patches using a MaxSim (Maximum Similarity) mechanism, which is a
    hallmark of the ColBERT/ColPali late-interaction strategy."
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
                "image_path": point.payload["image_path"],
                "score": point.score,
            })

        return retrieved


# ═════════════════════════════════════════════════════════════════════════════
# 5. Gemini Generator (Multimodal LLM for Answer Generation)
# ═════════════════════════════════════════════════════════════════════════════

class GeminiGenerator:
    """
    Generates answers using Google Gemini with retrieved page images.

    From the article: "The top-k relevant document images are retrieved and
    passed directly to a Multimodal LLM to generate the final,
    context-aware answer."

    Adapted from the reference implementation (which used Claude Sonnet)
    to use Google Gemini instead.
    """

    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = GEMINI_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None

    def _ensure_client(self):
        """Lazily initialize the Gemini client."""
        if self.client is not None:
            return

        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, query: str, retrieved_pages: list[dict]) -> str:
        """
        Generate an answer using Gemini with the retrieved page images.

        Args:
            query: The user's question
            retrieved_pages: List of dicts from ColPaliRetriever.retrieve()

        Returns:
            Generated answer string
        """
        self._ensure_client()

        # Build the multimodal prompt
        prompt_parts = []

        # System instruction
        prompt_parts.append(
            "You are a helpful document analysis assistant. "
            "You have been given document page images that were retrieved as "
            "the most relevant pages for the user's query. "
            "Analyze these images carefully and provide a detailed, accurate "
            "answer based ONLY on the information visible in these pages.\n\n"
            "Retrieved document pages:\n"
        )

        # Add each retrieved page image
        for i, page in enumerate(retrieved_pages, 1):
            img_path = page["image_path"]
            prompt_parts.append(
                f"\n--- Page {i}: {page['document']}, Page {page['page']} "
                f"(score: {page['score']:.4f}) ---\n"
            )

            # Load and add the image
            img = Image.open(img_path).convert("RGB")
            prompt_parts.append(img)

        # Add the query
        prompt_parts.append(f"\n\nUser Query: {query}\n\nProvide a detailed answer:")

        # Generate using google-genai SDK
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt_parts,
        )
        return response.text


# ═════════════════════════════════════════════════════════════════════════════
# 6. RAG Pipeline Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Full ColPali RAG pipeline orchestrator.

    Ties together all 5 components from the article:
    1. PDF → Image conversion
    2. Local image storage
    3. ColPali embedding + Qdrant indexing
    4. Query-time retrieval with MaxSim
    5. Answer generation with Gemini
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.indexer = None
        self.retriever = None
        self.generator = None
        self._model_loaded = False

    def _load_model(self):
        """Load the ColPali model (lazy, only when needed)."""
        if self._model_loaded:
            return

        loader = ColPaliModelLoader()
        self.model, self.processor = loader.load()
        self._model_loaded = True

    def index(self):
        """
        Full indexing pipeline:
        1. Convert all PDFs to images
        2. Load ColPali model
        3. Embed all page images
        4. Store in Qdrant
        """
        print("\n" + "═" * 60)
        print("       ColPali RAG Pipeline — INDEXING")
        print("═" * 60)

        # Step 1: Convert PDFs to images
        converter = PDFImageConverter()
        pages = converter.convert_all()

        if not pages:
            print("\n  No pages to index. Exiting.")
            return None

        # Step 2: Load model
        self._load_model()

        # Step 3 & 4: Embed and index
        self.indexer = QdrantIndexer()
        self.indexer.index_pages(self.model, self.processor, pages)

        # Setup retriever
        self.retriever = ColPaliRetriever(
            qdrant_client=self.indexer.get_client(),
            model=self.model,
            processor=self.processor,
        )

        # Setup generator
        self.generator = GeminiGenerator()

        print(f"\n  Pipeline ready! {len(pages)} pages indexed.")
        return pages

    def query(self, query_text: str, top_k: int = TOP_K, generate: bool = True) -> dict:
        """
        Query the pipeline:
        1. Embed the query with ColPali
        2. Retrieve top-k pages from Qdrant (MaxSim)
        3. (Optional) Generate answer with Gemini

        Args:
            query_text: Natural language query
            top_k: Number of pages to retrieve
            generate: Whether to generate an answer with Gemini

        Returns:
            Dict with 'retrieved_pages' and optionally 'answer'
        """
        if self.retriever is None:
            raise RuntimeError("Pipeline not indexed. Run index() first.")

        print(f"\n{'─'*60}")
        print(f"  Query: {query_text}")
        print(f"{'─'*60}")

        # Retrieve
        start = time.time()
        retrieved = self.retriever.retrieve(query_text, top_k=top_k)
        retrieval_time = time.time() - start

        print(f"\n  Retrieved {len(retrieved)} pages in {retrieval_time:.2f}s:")
        for i, page in enumerate(retrieved, 1):
            print(f"    {i}. {page['document']} — Page {page['page']} "
                  f"(score: {page['score']:.4f})")

        result = {"retrieved_pages": retrieved}

        # Generate
        if generate:
            try:
                print(f"\n  Generating answer with {GEMINI_MODEL}...")
                start = time.time()
                answer = self.generator.generate(query_text, retrieved)
                gen_time = time.time() - start

                print(f"\n  {'─'*56}")
                print(f"  Answer (generated in {gen_time:.1f}s):")
                print(f"  {'─'*56}")
                print(f"  {answer}")
                result["answer"] = answer
            except ValueError as e:
                print(f"\n  Skipping generation: {e}")
            except Exception as e:
                print(f"\n  Generation error: {e}")

        return result

    def interactive_query(self, top_k: int = TOP_K, generate: bool = True):
        """
        Interactive query loop — type questions and get answers.

        Commands:
            'quit' or 'exit' — stop the loop
            'top_k N' — change the number of results
            'gen on/off' — toggle answer generation
        """
        print("\n" + "═" * 60)
        print("       ColPali RAG Pipeline — INTERACTIVE QUERY")
        print("═" * 60)
        print("  Type your questions below. Commands:")
        print("    'quit'/'exit'  — stop")
        print("    'top_k N'      — change number of results")
        print("    'gen on/off'   — toggle Gemini generation")
        print("═" * 60)

        current_top_k = top_k
        current_gen = generate

        while True:
            try:
                user_input = input("\n  ❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("  Goodbye!")
                break

            if user_input.lower().startswith("top_k "):
                try:
                    current_top_k = int(user_input.split()[1])
                    print(f"  top_k set to {current_top_k}")
                except (IndexError, ValueError):
                    print("  Usage: top_k N")
                continue

            if user_input.lower() in ("gen on", "gen off"):
                current_gen = user_input.lower() == "gen on"
                print(f"  ✓ Generation {'enabled' if current_gen else 'disabled'}")
                continue

            self.query(user_input, top_k=current_top_k, generate=current_gen)


# ═════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ColPali Multi-Modal RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python colpali_pipeline.py index       # Index all PDFs
  python colpali_pipeline.py query       # Interactive query mode
  python colpali_pipeline.py pipeline    # Full pipeline (index + query)
        """,
    )
    parser.add_argument(
        "command",
        choices=["index", "query", "pipeline"],
        help="Command to run: 'index' to index PDFs, 'query' for interactive querying, "
             "'pipeline' for full index-then-query flow",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of pages to retrieve (default: {TOP_K})",
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Disable Gemini answer generation (retrieval-only mode)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Batch size for embedding (default: 2, keep small for CPU)",
    )

    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.command == "index":
        pipeline.index()
        print("\n  Done! Re-run with 'query' command to search your documents.")

    elif args.command == "query":
        # For query-only mode, we need to index first
        print("  Note: Indexing required before querying (in-memory Qdrant).")
        pages = pipeline.index()
        if pages:
            pipeline.interactive_query(
                top_k=args.top_k,
                generate=not args.no_generate,
            )

    elif args.command == "pipeline":
        pages = pipeline.index()
        if pages:
            pipeline.interactive_query(
                top_k=args.top_k,
                generate=not args.no_generate,
            )


if __name__ == "__main__":
    main()
