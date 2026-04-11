"""
PDF Ingestion (PDF → Image Conversion)
=======================================
Converts every page of every PDF in the source directory to a
high-resolution JPEG image (300 DPI by default).

From the article:
    "Uses pdf2image (with optimized DPI settings) to convert PDF pages
    into high-resolution images.  This preserves the visual integrity
    of the original file."
"""

from pathlib import Path

from tqdm import tqdm

from pipeline.config import IMAGE_DIR, PDF_DIR


class PDFImageConverter:
    """
    Converts PDF pages to high-resolution images.

    Output images are saved as JPEG files at 300 DPI.
    """

    def __init__(self, pdf_dir: Path = PDF_DIR, image_dir: Path = IMAGE_DIR, dpi: int = 300):
        self.pdf_dir = pdf_dir
        self.image_dir = image_dir
        self.dpi = dpi

    def convert_all(self) -> list[dict]:
        """
        Convert all PDFs in the source directory to page images.

        Returns:
            List of dicts with keys: pdf_name, page_num, image_path
        """
        from pdf2image import convert_from_path

        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"  No PDF files found in {self.pdf_dir}")
            return []

        print(f"\n{'='*60}")
        print(f"  Converting {len(pdf_files)} PDFs to images ({self.dpi} DPI)")
        print(f"{'='*60}")

        all_pages = []

        for pdf_path in tqdm(pdf_files, desc="  Processing PDFs"):
            pdf_name = pdf_path.stem
            output_dir = self.image_dir / pdf_name
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                images = convert_from_path(
                    pdf_path,
                    dpi=self.dpi,
                    fmt="jpeg",
                    thread_count=4,
                )

                for page_num, img in enumerate(images, start=1):
                    img_path = output_dir / f"page_{page_num}.jpeg"
                    img.save(img_path, "JPEG", quality=95)

                    all_pages.append({
                        "pdf_name": pdf_name,
                        "page_num": page_num,
                        "image_path": str(img_path),
                    })

            except Exception as e:
                print(f"\n  ✗ Error converting {pdf_name}: {e}")

        print(f"  Converted {len(all_pages)} pages from {len(pdf_files)} PDFs")
        return all_pages
