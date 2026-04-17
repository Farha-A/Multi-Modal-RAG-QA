"""
Answer Generation (Groq Multimodal LLM)
========================================
Sends retrieved document page images to Groq's multimodal LLM
(e.g. Llama 3.2 Vision) together with the user query, and returns
a context-aware answer.

From the article:
    "The top-k relevant document images are retrieved and passed
    directly to a Multimodal LLM to generate the final,
    context-aware answer."
"""

import base64

from pipeline.config import GROQ_API_KEY, GROQ_MODEL


class GroqGenerator:
    """
    Generates answers using Groq with retrieved page images.

    The Groq client is lazily initialised on the first ``generate()`` call.
    """

    def __init__(self, api_key: str = GROQ_API_KEY, model_name: str = GROQ_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None

    def _ensure_client(self):
        """Lazily initialize the Groq client."""
        if self.client is not None:
            return

        from groq import Groq
        self.client = Groq(api_key=self.api_key)

    def generate(self, query: str, retrieved_pages: list[dict]) -> str:
        """
        Generate an answer using Groq Llama 3.2 Vision with the retrieved page images.

        Args:
            query: The user's natural-language question.
            retrieved_pages: List of dicts (from ColPaliRetriever) containing
                             at least an ``image_path`` key.

        Returns:
            The generated answer string.
        """
        self._ensure_client()

        # Build the multimodal prompt content
        content = []

        # System instruction & Query
        content.append({
            "type": "text",
            "text": (
                "You are a helpful document analysis assistant. "
                "You have been given specific document structural chunks (text blocks, tables, images) that were retrieved as "
                "the most relevant sections for the user's query. "
                "Analyze these visual chunks carefully and provide a detailed, accurate "
                "answer based ONLY on the information visible in them.\n\n"
                "IMPORTANT: You MUST include source attributions for all information used in your answer. "
                "Use the provided metadata to cite explicitly in the format: "
                "`[Document Name, Page X, Chunk Type]`. "
                f"\n\nUser Query: {query}\n\nProvide a detailed answer:"
            )
        })

        # Add each retrieved page image encoded as base64
        for page in retrieved_pages:
            img_path = page["image_path"]
            doc_name = page.get("document", "Unknown")
            page_num = page.get("page", "Unknown")
            chunk_type = page.get("chunk_type", "Unknown")
            text_snippet = page.get("text", "")
            
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            # Prepend metadata reference for this image chunk
            content.append({
                "type": "text",
                "text": f"--- CHUNK METADATA ---\nDocument: {doc_name}\nPage: {page_num}\nType: {chunk_type}\nExtracted Text Snippet: {text_snippet}\n"
            })
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        # Generate using groq SDK
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
        )
        return response.choices[0].message.content
