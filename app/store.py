
from dataclasses import dataclass


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    embedding: list[float]


class DocumentStore:
    def __init__(self) -> None:
        self._documents: dict[str, list[StoredChunk]] = {}

    def add_document(
        self,
        doc_id: str,
        chunks: list[StoredChunk],
    ) -> None:
        """Add a new document or replace an existing one."""
        self._documents[doc_id] = chunks

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document. Return False if it does not exist."""
        if doc_id not in self._documents:
            return False

        del self._documents[doc_id]
        return True

    def get_document(
        self,
        doc_id: str,
    ) -> list[StoredChunk] | None:
        return self._documents.get(doc_id)

    def get_all_chunks(self) -> list[tuple[str, StoredChunk]]:
        """Return all chunks with their document IDs."""
        return [
            (doc_id, chunk)
            for doc_id, chunks in self._documents.items()
            for chunk in chunks
        ]

    def list_documents(self) -> list[dict[str, int | str]]:
        return [
            {
                "id": doc_id,
                "chunks": len(chunks),
            }
            for doc_id, chunks in sorted(self._documents.items())
        ]

    def is_empty(self) -> bool:
        return not self._documents
