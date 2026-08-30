import logging
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    # Dokümanları bölme ve her parçaya benzersiz metadata ekleme
    if not documents:
        logger.warning("Empty document list provided for splitting.")
        return []

    splitter = get_text_splitter()
    processed_chunks: list[Document] = []

    for doc in documents:
        # Tekil dokümanı parçala (üst dokümanın metadata'sı otomatik kopyalanır)
        doc_chunks = splitter.split_documents([doc])
        
        for idx, chunk in enumerate(doc_chunks):
            # Her parçaya benzersiz id ve sıralı indeks atanır
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["chunk_index"] = idx
            processed_chunks.append(chunk)

    logger.info("Split %d document(s) into %d chunk(s).", len(documents), len(processed_chunks))
    return processed_chunks