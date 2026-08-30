import logging
from pathlib import Path
from typing import List, Union
import pymupdf4llm
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_file(file_path: Union[str, Path]) -> List[Document]:
    # PyMuPDF4LLM ile PDF'i sayfa bazlı Markdown formatında okuma
    target_path = Path(file_path)
    docs: List[Document] = []
    
    try:
        # page_chunks=True her sayfayı ayrı bir sözlük olarak döndürür
        page_data_list = pymupdf4llm.to_markdown(
            doc=str(target_path),
            page_chunks=True
        )

        for page_idx, page_data in enumerate(page_data_list):
            text = page_data.get("text", "")
            
            # Sayfa numarası kütüphane çıktısında yoksa index üzerinden 1 tabanlı hesaplanır
            page_number = page_data.get("page", page_idx + 1)
            
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(target_path.name),
                            "page": page_number,
                        },
                    )
                )
        return docs
    except Exception as exc:
        logger.error("Error reading PDF file %s: %s", target_path.name, exc)
        raise exc


def load_all_pdfs(data_dir: Union[str, Path]) -> List[Document]:
    # Belirtilen dizindeki tüm PDF dosyalarını tarar ve toplu doküman listesi döner
    target_dir = Path(data_dir)
    all_docs: List[Document] = []
    successful_files: List[str] = []
    failed_files: List[str] = []

    if not target_dir.exists() or not target_dir.is_dir():
        logger.warning("Data directory does not exist or is not a directory: %s", target_dir)
        return all_docs

    pdf_files = list(target_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in directory: %s", target_dir)
        return all_docs

    for pdf_path in pdf_files:
        try:
            docs = load_pdf_file(pdf_path)
            all_docs.extend(docs)
            successful_files.append(pdf_path.name)
            logger.info("Loaded %d page(s) from %s", len(docs), pdf_path.name)
        except Exception as exc:
            failed_files.append(pdf_path.name)
            logger.error("Skipping corrupt or unreadable file %s: %s", pdf_path.name, exc)

    logger.info(
        "PDF ingestion completed: %d succeeded, %d failed out of %d total files.",
        len(successful_files),
        len(failed_files),
        len(pdf_files),
    )
    return all_docs