"""
PDF Ingestion (PDF → Multi-modal Semantic Chunks)
=================================================
Converts every page of every PDF into semantic chunks (text blocks,
tables, images) using PyMuPDF.

Extracts text/table metadata and saves high-resolution JPEGs
for each structural chunk to be indexed by ColPali.
"""

from pathlib import Path
import fitz  # PyMuPDF
from tqdm import tqdm

from pipeline.config import IMAGE_DIR, PDF_DIR


class PDFImageConverter:
    """
    Converts PDF pages into semantic chunks (Images, Tables, Paragraphs).

    Output images are saved as JPEG files at 300 DPI, along with metadata.
    """

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
        print(f"  Chunking {len(pdf_files)} PDFs (Semantic Extraction, {self.dpi} DPI)")
        print(f"{'='*60}")

        all_chunks = []

        for pdf_path in tqdm(pdf_files, desc="  Processing PDFs"):
            pdf_name = pdf_path.stem
            output_dir = self.image_dir / pdf_name
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                doc = fitz.open(pdf_path)
                for page_num_0 in range(len(doc)):
                    page_num = page_num_0 + 1
                    page = doc[page_num_0]
                    
                    # 1. Extract tables
                    tables = page.find_tables()
                    table_bboxes = []
                    if tables.tables:
                        for idx, tab in enumerate(tables.tables):
                            bbox = fitz.Rect(tab.bbox)
                            table_bboxes.append(bbox)
                            
                            # Expand slightly for context
                            context_bbox = bbox + fitz.Rect(-5, -5, 5, 5)
                            context_bbox.intersect(page.rect)
                            
                            pix = page.get_pixmap(dpi=self.dpi, clip=context_bbox)
                            chunk_id = f"page_{page_num}_table_{idx}"
                            img_path = output_dir / f"{chunk_id}.jpeg"
                            pix.save(str(img_path))
                            
                            # Extract text from table
                            # tab.extract() returns a list of lists (rows of columns)
                            extracted = tab.extract()
                            table_text = "\n".join([" | ".join([str(c) if c else "" for c in row]) for row in extracted if row])
                            
                            all_chunks.append({
                                "pdf_name": pdf_name,
                                "page_num": page_num,
                                "chunk_id": chunk_id,
                                "image_path": str(img_path),
                                "text": table_text.strip(),
                                "chunk_type": "Table"
                            })

                    # 2. Extract layout blocks
                    blocks = page.get_text("blocks") # x0, y0, x1, y1, text, block_type, block_no
                    for idx, b in enumerate(blocks):
                        x0, y0, x1, y1, text, block_type, block_no = b
                        rect = fitz.Rect(x0, y0, x1, y1)
                        
                        # Skip if this block is inside any table
                        is_in_table = False
                        for t_bbox in table_bboxes:
                            intersect = rect.intersect(t_bbox)
                            # If overlap area is significant, skip
                            if not intersect.is_empty:
                                is_in_table = True
                                break
                        
                        if is_in_table:
                            continue
                            
                        chunk_type = "Image/Chart" if block_type == 1 else "Text"
                        text = text.strip()
                        
                        # Filter out negligible text blocks
                        if chunk_type == "Text" and len(text) < 15:
                            continue
                            
                        # Ensure bbox is valid and visible
                        if rect.width < 10 or rect.height < 10:
                            continue
                            
                        context_bbox = rect + fitz.Rect(-5, -5, 5, 5)
                        context_bbox.intersect(page.rect)
                        
                        pix = page.get_pixmap(dpi=self.dpi, clip=context_bbox)
                        chunk_id = f"page_{page_num}_block_{idx}"
                        img_path = output_dir / f"{chunk_id}.jpeg"
                        pix.save(str(img_path))
                        
                        all_chunks.append({
                            "pdf_name": pdf_name,
                            "page_num": page_num,
                            "chunk_id": chunk_id,
                            "image_path": str(img_path),
                            "text": text,
                            "chunk_type": chunk_type
                        })

                doc.close()
            except Exception as e:
                print(f"\n  ✗ Error parsing {pdf_name}: {e}")

        print(f"  Extracted {len(all_chunks)} semantic chunks from {len(pdf_files)} PDFs")
        return all_chunks
