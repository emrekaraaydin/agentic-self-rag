import logging
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE: int = 1600
CHUNK_OVERLAP: int = 250

HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]


def get_markdown_header_splitter() -> MarkdownHeaderTextSplitter:
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )


def get_recursive_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    # İki aşamalı parçalama: Markdown başlık hiyerarşisi + Karakter limiti denetimi
    if not documents:
        logger.warning("Empty document list provided for splitting.")
        return []

    header_splitter = get_markdown_header_splitter()
    text_splitter = get_recursive_text_splitter()
    processed_chunks: List[Document] = []

    for doc in documents:
        # 1. Aşama: Sayfa metnini başlıklara göre ayır
        header_splits = header_splitter.split_text(doc.page_content)

        # Başlık içermeyen sayfalar için ham dokümanı koru
        if not header_splits:
            header_splits = [doc]
        else:
            for split_doc in header_splits:
                # Orijinal dokümanın metadata bilgilerini (source, page) yeni parçalara aktar
                for key, value in doc.metadata.items():
                    if key not in split_doc.metadata:
                        split_doc.metadata[key] = value

        # 2. Aşama: Başlık bloklarını CHUNK_SIZE sınırına göre alt parçalara böl
        sub_chunks = text_splitter.split_documents(header_splits)

        for idx, chunk in enumerate(sub_chunks):
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["chunk_index"] = idx
            processed_chunks.append(chunk)

    logger.info("Split %d document(s) into %d chunk(s).", len(documents), len(processed_chunks))
    return processed_chunks