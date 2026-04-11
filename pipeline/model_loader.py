"""
ColPali Model Loader
====================
Loads the ColQwen2 / ColSmol vision-language model and its processor
from HuggingFace, configured for **CPU inference** (float32).

From the article:
    ColPali uses a Vision Language Model to generate multi-vector
    embeddings directly from document images, allowing the system to
    "see" the document rather than relying on OCR.
"""

import time

import torch

from pipeline.config import COLPALI_MODEL_NAME


class ColPaliModelLoader:
    """
    Loads ColQwen2 model and processor from HuggingFace.

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
