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
                    st.success(f"Successfully indexed {len(pages)} pages!")
                else:
                    st.warning("No pages found to index. Check your Data/PDFs directory.")
                    
    if not st.session_state.indexed:
        st.info("⬅️ Please index your documents from the sidebar to start querying.")
        return
        
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
                st.markdown("### Retrieved Pages")
                cols = st.columns(len(result["retrieved_pages"]))
                for col, page in zip(cols, result["retrieved_pages"]):
                    with col:
                        img = Image.open(page["image_path"])
                        st.image(img, caption=f"{page['document']} (Page {page['page']}) | Score: {page['score']:.4f}", width='stretch')


if __name__ == "__main__":
    main()
