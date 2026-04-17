# 📄 ColPali Multi-Modal RAG Pipeline

  A visual retrieval-augmented generation system that *sees* your documents — no OCR required.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?logo=streamlit&logoColor=white)
  ![ColPali](https://img.shields.io/badge/ColPali-ColSmol-blueviolet)
  ![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?logo=qdrant&logoColor=white)
  ![Groq](https://img.shields.io/badge/Groq-Llama_3.2_Vision-orange)

---

## Overview

This project implements an end-to-end **Multi-Modal Retrieval-Augmented Generation (RAG)** pipeline that queries PDF document collections using visual embeddings instead of traditional text extraction. Built on the [ColPali](https://arxiv.org/abs/2407.01449) architecture, the system generates multi-vector embeddings from semantic visual chunks (tables, images, text blocks), enabling semantically rich and fine-grained retrieval that precisely captures complex layouts.

### Why Visual RAG?

Traditional RAG pipelines rely on OCR and text extraction, which often **lose critical visual context** — tables break apart, figures are ignored, and complex layouts are flattened. This pipeline enhances visual RAG by using PyMuPDF to structurally parse documents and extract targeted chunks as distinct, high-resolution visual segments, preserving full visual fidelity while improving retrieval precision.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
│                    "What is shown in Table 3?"                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. INGESTION  (PDFImageConverter)                              │
│     PDF files ──► Semantic visual chunks (Tables, Text, Images) │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. INDEXING  (QdrantIndexer + ColPaliModelLoader)              │
│     Visual chunks ──► colSmol-256M multi-vector embeddings ──► Qdrant │
│     (128-dim patches, COSINE distance, MaxSim comparator)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. RETRIEVAL  (ColPaliRetriever)                               │
│     Query embedding ──► MaxSim late-interaction search ──►      │
│     Top-k most relevant visual chunks                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. GENERATION  (GroqGenerator)                                 │
│     Retrieved visual chunks + query ──► Groq Llama 3.2 Vision  │
│     ──► Context-aware natural language answer                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```text
.
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── .env                        # API keys & configuration (not tracked)
├── Data/
│   ├── data_download.ipynb     # Notebook to download DocBank PDFs from arXiv
│   ├── PDFs/                   # Source PDF documents
│   ├── images/                 # Converted page images (auto-generated)
│   └── qdrant_db/              # Persistent Qdrant vector database (auto-generated)
└── pipeline/
    ├── __init__.py             # Package exports & quick-start docs
    ├── config.py               # Centralised configuration & environment variables
    ├── ingestion.py            # PDF → JPEG image conversion (300 DPI)
    ├── model_loader.py         # ColQwen2 / ColSmol model & processor loader
    ├── indexer.py              # Qdrant collection creation & embedding upsert
    ├── retrieval.py            # MaxSim query-time retrieval
    ├── generation.py           # Groq Llama 3.2 Vision answer generation
    └── rag_pipeline.py         # Orchestrator tying all components together
```

---

## Key Components

| Module | Class | Responsibility |
| --- | --- | --- |
| `ingestion.py` | `PDFImageConverter` | Converts PDFs into semantic visual chunks (Tables, Text, Images) using `PyMuPDF` |
| `model_loader.py` | `ColPaliModelLoader` | Loads ColSmol model & processor from HuggingFace for CPU inference |
| `indexer.py` | `QdrantIndexer` | Creates a Qdrant collection with multi-vector config and upserts ColPali embeddings |
| `retrieval.py` | `ColPaliRetriever` | Embeds a text query with ColPali and retrieves top-k chunks via MaxSim scoring |
| `generation.py` | `GroqGenerator` | Sends retrieved visual chunks + query to Groq's Llama 3.2 Vision for answer generation |
| `rag_pipeline.py` | `RAGPipeline` | Orchestrates the full index → retrieve → generate workflow |
| `config.py` | — | Centralises env vars, paths, model names, and default constants |

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Groq API Key** — get one at [console.groq.com](https://console.groq.com)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Farha-A/Multi-Modal-RAG-QA.git
   cd Multi-Modal-RAG-QA
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key_here

   # Optional overrides (defaults shown)
   COLPALI_MODEL_NAME=vidore/colSmol-256M
   QDRANT_COLLECTION_NAME=docbank_colpali
   GROQ_MODEL=llama-3.2-11b-vision-preview
   TOP_K=3
   ```

### Preparing the Data

Place your PDF files in the `Data/PDFs/` directory, or use the provided notebook to download sample papers from arXiv:

```bash
jupyter notebook Data/data_download.ipynb
```

The notebook downloads 30 random papers from the [DocBank](https://doc-analysis.github.io/docbank-page/) subset of arXiv (January 2014 papers).

---

## Usage

### Streamlit Web App (Recommended)

Launch the interactive web interface:

```bash
streamlit run app.py
```

Then in the browser:

1. Click **"Index Documents"** in the sidebar to ingest your PDFs (one-time step)
2. Type a question in the chat input or explore the **Evaluation Suite** tab
3. View the generated answer alongside the retrieved source visual chunks

> **Note:** The first run will download the ColSmol model from HuggingFace (~400 MB). Subsequent runs load from cache.

### Python API

```python
from pipeline import RAGPipeline

# Initialise and index
rag = RAGPipeline()
rag.index()  # semantic chunking, embeds chunks, stores in Qdrant

# Query
result = rag.query("What methodology was used in the experiments?")
print(result["answer"])
print(result["retrieved_pages"])
```

### Interactive CLI

```python
from pipeline import RAGPipeline

rag = RAGPipeline()
rag.index()
rag.interactive_query()

# Commands available in the REPL:
#   quit / exit    — stop
#   top_k N        — change number of retrieved pages
#   gen on / off   — toggle answer generation
```

---

## Configuration

All settings are centralised in `pipeline/config.py` and can be overridden via environment variables or the `.env` file:

| Variable | Default | Description |
| --- | --- | --- |
| `COLPALI_MODEL_NAME` | `vidore/colSmol-256M` | HuggingFace model ID for the ColPali vision encoder |
| `QDRANT_COLLECTION_NAME` | `docbank_colpali` | Name of the Qdrant vector collection |
| `GROQ_API_KEY` | — | API key for Groq inference (required for generation) |
| `GROQ_MODEL` | `llama-3.2-11b-vision-preview` | Groq model used for answer generation |
| `TOP_K` | `3` | Number of chunks to retrieve per query |

---

## Technical Details

### Embedding Model

The pipeline uses **ColSmol** (`vidore/colSmol-256M`) by default — a Vision Language Model fine-tuned for document retrieval. It generates **128-dimensional multi-vector embeddings** where each vector corresponds to a visual patch of the document image. This enables the late-interaction **MaxSim** scoring mechanism inherited from the ColBERT family.

The model loader also supports **ColQwen2** / **ColIdefics3** variants for lighter-weight inference.

### Vector Database

**Qdrant** runs as a persistent local instance (no Docker or external server needed). The collection is configured with:

- **Vector size:** 128
- **Distance metric:** Cosine similarity
- **Multi-vector comparator:** MaxSim (late-interaction scoring)

The database is stored on disk at `Data/qdrant_db/` and persists across application restarts.

### Inference

All model inference runs on **CPU** with `float32` precision. While this avoids any GPU dependency, indexing large document collections will be slower. The Streamlit app caches the loaded model using `@st.cache_resource` to avoid reloading on each interaction.
