"""
ColPali Multi-Modal RAG Pipeline
=================================
A modular pipeline for multi-modal retrieval-augmented generation on
PDF document collections.

Architecture
------------
1. **Ingestion**   — ``PDFImageConverter``:  PDF → page images (300 DPI)
2. **Indexing**    — ``QdrantIndexer``:      ColPali embeddings → Qdrant
3. **Retrieval**   — ``ColPaliRetriever``:   query → MaxSim → top-k pages
4. **Generation**  — ``GroqGenerator``:      pages + query → LLM answer
5. **Orchestrator**— ``RAGPipeline``:        ties 1-4 together

Quick start::

    from pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.index()
    result = rag.query("What is …?")
"""

from pipeline.generation import GroqGenerator
from pipeline.indexer import QdrantIndexer
from pipeline.ingestion import PDFImageConverter
from pipeline.model_loader import ColPaliModelLoader
from pipeline.rag_pipeline import RAGPipeline
from pipeline.retrieval import ColPaliRetriever

__all__ = [
    "ColPaliModelLoader",
    "PDFImageConverter",
    "QdrantIndexer",
    "ColPaliRetriever",
    "GroqGenerator",
    "RAGPipeline",
]
