
from app import llm
from app.chunking import chunk_text
from app.schemas import (
    AskResponse,
    IngestDocument,
    SourceInfo,
)
from app.store import DocumentStore, StoredChunk


class IngestionService:
    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def ingest(
        self,
        documents: list[IngestDocument],
    ) -> int:
        """Ingest documents and return the number of chunks created."""
        total_chunks = 0

        for document in documents:
            texts = chunk_text(document.text)

            chunks = [
                StoredChunk(
                    chunk_id=f"{document.id}:{index}",
                    text=text,
                    embedding=llm.embed(text),
                )
                for index, text in enumerate(texts)
            ]

            self.store.add_document(document.id, chunks)
            total_chunks += len(chunks)

        return total_chunks


class RetrievalService:
    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    @staticmethod
    def similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """Compute dot product of normalized vectors."""
        return sum(
            a * b for a, b in zip(vector_a, vector_b)
        )

    def ask(
        self,
        question: str,
        top_k: int,
    ) -> AskResponse:
        question_vector = llm.embed(question)

        scored_chunks = []

        for doc_id, chunk in self.store.get_all_chunks():
            score = self.similarity(
                question_vector,
                chunk.embedding,
            )

            scored_chunks.append(
                SourceInfo(
                    doc_id=doc_id,
                    chunk_id=chunk.chunk_id,
                    score=score,
                    text=chunk.text,
                )
            )

        scored_chunks.sort(
            key=lambda source: source.score,
            reverse=True,
        )

        sources = scored_chunks[:top_k]

        answer = llm.generate(
            question,
            [source.text for source in sources],
        )

        return AskResponse(
            answer=answer,
            sources=sources,
        )
