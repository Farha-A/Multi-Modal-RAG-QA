"""
ColPali Multi-Modal RAG — Streamlit Application
================================================
Interactive web interface for querying PDF document collections
using ColPali embeddings and Groq Llama 3.2 Vision.

Usage::

    streamlit run app.py
"""

import streamlit as st
from PIL import Image

from pipeline import RAGPipeline
from pipeline.config import TOP_K
from pipeline.benchmark import run_benchmark, BENCHMARK_QUERIES


def main():
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="ColPali Multi-Modal RAG", page_icon="📄", layout="wide")
    st.title("ColPali Multi-Modal RAG Pipeline")
    st.markdown("Query your PDF documents using ColPali embeddings and Groq Llama 3.2 Vision.")

    @st.cache_resource(show_spinner=False)
    def get_pipeline():
        p = RAGPipeline()
        p.load_pipeline()
        return p
        
    pipeline = get_pipeline()
    
    if "indexed" not in st.session_state:
        st.session_state.indexed = pipeline.retriever is not None

    with st.sidebar:
        st.header("Configuration")
        top_k = st.number_input("Top K Results", min_value=1, max_value=10, value=TOP_K)
        generate = st.checkbox("Generate Answer with Groq", value=True)
        
        st.markdown("---")
        if st.button("Index Documents", width="stretch"):
            with st.spinner("Indexing PDFs... This might take a while."):
                pages = pipeline.index()
                if pages is not None:
                    st.session_state.indexed = True
                    st.success(f"Successfully indexed {len(pages)} chunks!")
                else:
                    st.warning("No chunks found to index. Check your Data/PDFs directory.")
                    
    if not st.session_state.indexed:
        st.info("⬅️ Please index your documents from the sidebar to start querying.")
        return
        
    tab1, tab2 = st.tabs(["Chat & Query", "Evaluation Suite"])
    
    with tab1:
        query_text = st.chat_input("Ask a question about your documents...")
        
        if query_text:
            st.chat_message("user").write(query_text)
            
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating answer..."):
                    result = pipeline.query(query_text, top_k=top_k, generate=generate)
                    
                if "answer" in result:
                    st.write(result["answer"])
                elif "error" in result:
                    st.error(f"**API Error:** {result['error']}")
                    
                if result.get("retrieved_pages"):
                    st.markdown("### Retrieved Chunks")
                    cols = st.columns(len(result["retrieved_pages"]))
                    for col, page in zip(cols, result["retrieved_pages"]):
                        with col:
                            img = Image.open(page["image_path"])
                            chunk_type = page.get("chunk_type", "Chunk")
                            caption = f"{page['document']} (P{page['page']}) | {chunk_type} | Score: {page['score']:.4f}"
                            st.image(img, caption=caption, width='stretch')
                            if page.get("text"):
                                with st.expander("View Text"):
                                    st.write(page["text"])

    with tab2:
        st.header("Multi-Modal Benchmark Suite")
        st.markdown("Run a standardized set of queries to evaluate retrieval and answer generation across Text, Table, Figure, and Layout modalities.")
        
        if st.button("Run Benchmark Suite", type="primary"):
            st.info(f"Running {len(BENCHMARK_QUERIES)} benchmark queries. Generation enabled: `{generate}`.")
            progress_bar = st.progress(0)
            
            for idx, result in enumerate(run_benchmark(pipeline, top_k=top_k, generate=generate)):
                progress = (idx + 1) / len(BENCHMARK_QUERIES)
                progress_bar.progress(progress)
                
                with st.expander(f"[{result['modality']}] {result['query']}", expanded=True):
                    st.write(f"**Description:** {result['description']}")
                    st.markdown(f"**Retrieval Time:** `{result['retrieval_time']:.2f}s` | **Generation Time:** `{result['generation_time']:.2f}s`")
                    
                    if result["error"]:
                        st.error(f"Error: {result['error']}")
                    elif result["answer"]:
                        st.info(f"**Answer:** {result['answer']}")
                        if result.get("has_citation"):
                            st.success("✅ Proper Source Citation Detected")
                        else:
                            st.warning("⚠️ No clear source citation found in the answer")
                        
                    if result["retrieved_pages"]:
                        st.markdown("**Top Retrieved Chunks:**")
                        cols = st.columns(min(3, len(result["retrieved_pages"])))
                        for i, (col, page) in enumerate(zip(cols, result["retrieved_pages"][:3])):
                            with col:
                                img = Image.open(page["image_path"])
                                chunk_type = page.get("chunk_type", "Chunk")
                                st.image(img, caption=f"Rank {i+1} | {chunk_type} | Score: {page['score']:.4f}", width='stretch')
                                if page.get("text"):
                                    with st.expander("Text"):
                                        st.write(page["text"])
                                
            st.success("Benchmark Complete!")


if __name__ == "__main__":
    main()
