import logging
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_file(file_path: Path) -> List[Document]:
    # PyMuPDF ile tek bir PDF dosyasını okuma
    docs: List[Document] = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path.name),
                            "page": page_num + 1,
                        },
                    )
                )
        return docs
    except Exception as exc:
        logger.error(f"Error reading PDF file {file_path.name}: {exc}")
        raise exc
def load_all_pdfs(data_dir: Path) -> List[Document]:
    # Belirtilen dizindeki tüm PDF dosyalarını tarar ve toplu doküman listesi döner
    all_docs: List[Document] = []
    successful_files: List[str] = []
    failed_files: List[str] = []

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in directory: {data_dir}")
        return all_docs

    for pdf_path in pdf_files:
        try:
            docs = load_pdf_file(pdf_path)
            all_docs.extend(docs)
            successful_files.append(pdf_path.name)
            logger.info(f"Loaded {len(docs)} page(s) from {pdf_path.name}")
        except Exception as exc:
            failed_files.append(pdf_path.name)
            logger.error(f"Skipping corrupt or unreadable file {pdf_path.name}: {exc}")

    logger.info(
        f"PDF ingestion completed: {len(successful_files)} succeeded, "
        f"{len(failed_files)} failed out of {len(pdf_files)} total files."
    )
    return all_docs
