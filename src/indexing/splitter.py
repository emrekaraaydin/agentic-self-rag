"""Header-aware two-stage Markdown text splitter with structural breadcrumb injection."""

import logging
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)

# --- Chunking Parameters ---
CHUNK_SIZE: int = 1600
CHUNK_OVERLAP: int = 250

HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]


# --- Splitter Factories ---
def get_markdown_header_splitter() -> MarkdownHeaderTextSplitter:
    """Return a splitter that segments text along Markdown heading levels."""
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

def get_recursive_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return a character splitter enforcing CHUNK_SIZE and CHUNK_OVERLAP bounds."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )


# --- Two-Stage Chunking Pipeline ---
def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split pages by Markdown headers, enforce chunk size limits, and prepend heading breadcrumbs."""
    if not documents:
        logger.warning("Empty document list provided for splitting.")
        return []

    header_splitter = get_markdown_header_splitter()
    text_splitter = get_recursive_text_splitter()
    processed_chunks: List[Document] = []

    for doc in documents:
        # Stage 1: Split page content along Markdown headings
        header_splits = header_splitter.split_text(doc.page_content)

        # Preserve pages that have no Markdown headings as a single split
        if not header_splits:
            header_splits = [doc]
        else:
            for split_doc in header_splits:
                # Propagate source and page metadata to every heading block
                for key, value in doc.metadata.items():
                    if key not in split_doc.metadata:
                        split_doc.metadata[key] = value

        # Stage 2: Sub-divide heading blocks according to CHUNK_SIZE
        sub_chunks = text_splitter.split_documents(header_splits)

        for chunk in sub_chunks:
            chunk.metadata["chunk_id"] = str(uuid.uuid4())

            # Stage 3: Prepend the active heading hierarchy as context breadcrumbs
            active_headers = [
                str(chunk.metadata[header_key]).strip()
                for _, header_key in HEADERS_TO_SPLIT_ON
                if header_key in chunk.metadata and chunk.metadata[header_key]
            ]

            if active_headers:
                header_context = " > ".join(active_headers)
                deepest_header = active_headers[-1]

                # Prefix heading hierarchy when the chunk body does not start with its deepest header
                if not chunk.page_content.lstrip().startswith(deepest_header):
                    chunk.page_content = f"[{header_context}]\n{chunk.page_content}"

            processed_chunks.append(chunk)

    logger.info("Split %d document(s) into %d chunk(s).", len(documents), len(processed_chunks))
    return processed_chunks