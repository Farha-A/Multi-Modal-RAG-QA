"""
Multi-Modal Benchmark Evaluation Suite
======================================
Contains predefined queries across multiple modalities (Text, Table, Figure, Layout)
to evaluate the retrieval and generation performance of the ColPali pipeline.
"""

import time
from typing import Dict, Generator, Any

BENCHMARK_QUERIES = [
    {
        "modality": "Text",
        "query": "What methodology was used in the experiments described in the paper?",
        "description": "General text retrieval for methodology."
    },
    {
        "modality": "Text",
        "query": "What are the key conclusions or findings of the study?",
        "description": "General text retrieval for conclusions."
    },
    {
        "modality": "Table",
        "query": "Based on the comparison tables, which method performs best and what are its quantitative results?",
        "description": "Requires extracting structural tabular information."
    },
    {
        "modality": "Table",
        "query": "What datasets or hyperparameters are listed in the data summary tables?",
        "description": "Extracting descriptive info from parameter tables."
    },
    {
        "modality": "Figure",
        "query": "What does the architecture diagram or system overview figure illustrate?",
        "description": "Visual semantic understanding of a block diagram or flowchart."
    },
    {
        "modality": "Figure",
        "query": "Describe the main trends shown in the experimental graphs or performance plots.",
        "description": "Visual understanding of a data distribution or line graph."
    },
    {
        "modality": "Layout",
        "query": "How is the references or bibliography section structured?",
        "description": "Requires identifying page layout for references."
    }
]

def run_benchmark(pipeline: Any, top_k: int = 3, generate: bool = True) -> Generator[Dict, None, None]:
    """
    Executes the benchmark queries against the provided pipeline.
    Yields results one by one to allow for incremental progress reporting in UIs.
    """
    if pipeline.retriever is None:
        raise RuntimeError("Pipeline is not indexed. Cannot run benchmark.")

    for item in BENCHMARK_QUERIES:
        query_text = item["query"]
        modality = item["modality"]
        
        result = {
            "query": query_text,
            "modality": modality,
            "description": item["description"],
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "retrieved_pages": [],
            "answer": None,
            "error": None
        }

        try:
            # Measure retrieval
            start_retrieval = time.time()
            retrieved = pipeline.retriever.retrieve(query_text, top_k=top_k)
            result["retrieval_time"] = time.time() - start_retrieval
            result["retrieved_pages"] = retrieved

            # Measure generation if enabled
            if generate and hasattr(pipeline, 'generator') and pipeline.generator:
                start_generation = time.time()
                answer = pipeline.generator.generate(query_text, retrieved)
                result["generation_time"] = time.time() - start_generation
                result["answer"] = answer
                result["has_citation"] = "[" in answer and "]" in answer
        except Exception as e:
            result["error"] = str(e)

        yield result
