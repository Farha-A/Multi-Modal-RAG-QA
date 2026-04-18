"""
PDF Ingestion (PDF → Multi-modal Semantic Chunks)
=================================================
Converts every page of every PDF into semantic chunks (text blocks,
tables, images) using PyMuPDF4LLM with ``pymupdf_layout`` for
GNN-based page layout analysis.

Extracts text/table metadata and saves high-resolution JPEGs
for each structural chunk to be indexed by ColPali.
"""

from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm
from tqdm import tqdm

from pipeline.config import IMAGE_DIR, PDF_DIR


class PDFImageConverter:
    """
    Converts PDF pages into semantic chunks (Images, Tables, Paragraphs).

    Uses ``pymupdf4llm`` with the ``pymupdf_layout`` GNN engine for
    improved layout analysis (titles, headers/footers, tables, pictures).
    Output images are saved as JPEG files at 300 DPI, along with metadata.
    """

    # Map pymupdf_layout box classes to our chunk types
    _CLASS_MAP = {
        "text": "Text",
        "table": "Table",
        "picture": "Image/Chart",
    }

    def __init__(self, pdf_dir: Path = PDF_DIR, image_dir: Path = IMAGE_DIR, dpi: int = 300):
        self.pdf_dir = pdf_dir
        self.image_dir = image_dir
        self.dpi = dpi

    def convert_all(self) -> list[dict]:
        """
        Convert all PDFs into semantic chunk images and metadata.

        Returns:
            List of dicts with keys: pdf_name, page_num, chunk_id, image_path, text, chunk_type
        """
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"  No PDF files found in {self.pdf_dir}")
            return []

        print(f"\n{'='*60}")
        print(f"  Chunking {len(pdf_files)} PDFs (PyMuPDF-Layout GNN, {self.dpi} DPI)")
        print(f"{'='*60}")

        all_chunks = []

        for pdf_path in tqdm(pdf_files, desc="  Processing PDFs"):
            pdf_name = pdf_path.stem
            output_dir = self.image_dir / pdf_name
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                all_chunks.extend(
                    self._process_pdf(pdf_path, pdf_name, output_dir)
                )
            except Exception as e:
                print(f"\n  ✗ Error parsing {pdf_name}: {e}")

        print(f"  Extracted {len(all_chunks)} semantic chunks from {len(pdf_files)} PDFs")
        return all_chunks

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _process_pdf(self, pdf_path: Path, pdf_name: str, output_dir: Path) -> list[dict]:
        """Run pymupdf4llm layout analysis and extract chunk images."""

        # page_chunks=True → list of dicts (one per page)
        # Each dict contains: text (markdown), page_boxes (layout bboxes)
        # pymupdf_layout is automatically active (GNN-based analysis)
        page_dicts = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            show_progress=False,
        )

        # Open document with raw fitz for rendering chunk images
        doc = fitz.open(pdf_path)
        chunks: list[dict] = []

        for page_dict in page_dicts:
            page_num = page_dict["metadata"]["page_number"]  # 1-based
            page = doc[page_num - 1]
            md_text = page_dict.get("text", "")
            page_boxes = page_dict.get("page_boxes", [])

            for box in page_boxes:
                box_class = box["class"]           # "text", "table", "picture", …
                bbox = box["bbox"]                 # [x0, y0, x1, y1]
                pos_start, pos_end = box["pos"]    # char offsets into md_text
                box_idx = box["index"]

                chunk_type = self._CLASS_MAP.get(box_class)
                if chunk_type is None:
                    # Skip classes we don't track (headers, footers, …)
                    continue

                # Extract text content for this box from the markdown
                text = md_text[pos_start:pos_end].strip()

                # Filter out negligible text blocks (by word count)
                if chunk_type == "Text" and len(text.split()) < 30:
                    continue

                # Build a fitz.Rect and validate
                rect = fitz.Rect(bbox)
                if rect.width < 10 or rect.height < 10:
                    continue

                # Expand slightly for context, clamp to page
                context_bbox = rect + fitz.Rect(-5, -5, 5, 5)
                context_bbox.intersect(page.rect)

                # Render chunk region to JPEG
                pix = page.get_pixmap(dpi=self.dpi, clip=context_bbox)
                chunk_id = f"page_{page_num}_box_{box_idx}"
                img_path = output_dir / f"{chunk_id}.jpeg"
                pix.save(str(img_path))

                chunks.append({
                    "pdf_name": pdf_name,
                    "page_num": page_num,
                    "chunk_id": chunk_id,
                    "image_path": str(img_path),
                    "text": text,
                    "chunk_type": chunk_type,
                })

        doc.close()
        return chunks
