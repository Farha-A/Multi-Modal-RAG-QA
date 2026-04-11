"""
RAG Pipeline Orchestrator
=========================
Ties together all pipeline components into a single, high-level API:

1. PDF → Image conversion    (``ingestion.PDFImageConverter``)
2. Local image storage
3. ColPali embedding + Qdrant indexing  (``indexer.QdrantIndexer``)
4. Query-time retrieval with MaxSim    (``retrieval.ColPaliRetriever``)
5. Answer generation with Groq         (``generation.GroqGenerator``)

Typical usage::

    pipeline = RAGPipeline()
    pipeline.index()              # one-time ingestion
    result = pipeline.query("What is …?")
"""

import time

from pipeline.config import GROQ_MODEL, TOP_K
from pipeline.generation import GroqGenerator
from pipeline.indexer import QdrantIndexer
from pipeline.ingestion import PDFImageConverter
from pipeline.model_loader import ColPaliModelLoader
from pipeline.retrieval import ColPaliRetriever


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

    def load_pipeline(self) -> bool:
        """Loads the pre-existing Qdrant collection and initializes the retriever if available."""
        if self.indexer is None:
            self.indexer = QdrantIndexer()
        if self.indexer.client.collection_exists(self.indexer.collection_name):
            try:
                info = self.indexer.client.get_collection(self.indexer.collection_name)
                if info.points_count > 0:
                    self._load_model()
                    self.retriever = ColPaliRetriever(
                        qdrant_client=self.indexer.get_client(),
                        model=self.model,
                        processor=self.processor,
                    )
                    self.generator = GroqGenerator()
                    return True
            except Exception as e:
                print(f"  Error loading existing database: {e}")
        return False

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
        if self.indexer is None:
            self.indexer = QdrantIndexer()
        self.indexer.index_pages(self.model, self.processor, pages)

        # Setup retriever
        self.retriever = ColPaliRetriever(
            qdrant_client=self.indexer.get_client(),
            model=self.model,
            processor=self.processor,
        )

        # Setup generator
        self.generator = GroqGenerator()

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
                print(f"\n  Generating answer with {GROQ_MODEL}...")
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
                result["error"] = str(e)
            except Exception as e:
                print(f"\n  Generation error: {e}")
                result["error"] = str(e)

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
                print(f"  Generation {'enabled' if current_gen else 'disabled'}")
                continue

            self.query(user_input, top_k=current_top_k, generate=current_gen)
