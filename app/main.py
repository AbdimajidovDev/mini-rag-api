
from fastapi import FastAPI, HTTPException, Response, status

from app import llm  # noqa: F401
from app.chunking import chunk_text  # noqa: F401
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentsResponse,
    IngestRequest,
    IngestResponse,
)
from app.services import IngestionService, RetrievalService
from app.store import DocumentStore


def create_app() -> FastAPI:
    """Create a new FastAPI application with an isolated in-memory store."""

    app = FastAPI(
        title="Mini RAG API",
        description="A lightweight retrieval-augmented question-answering API.",
        version="1.0.0",
    )

    # Each application instance gets its own storage.
    store = DocumentStore()

    ingestion_service = IngestionService(store)
    retrieval_service = RetrievalService(store)

    app.state.store = store

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/ingest",
        response_model=IngestResponse,
        status_code=status.HTTP_200_OK,
    )
    def ingest(payload: IngestRequest) -> IngestResponse:
        chunks_count = ingestion_service.ingest(payload.documents)

        return IngestResponse(
            ingested=len(payload.documents),
            chunks=chunks_count,
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
    )
    def ask(payload: AskRequest) -> AskResponse:
        if store.is_empty():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents have been ingested.",
            )

        return retrieval_service.ask(
            question=payload.question,
            top_k=payload.top_k,
        )

    @app.get(
        "/documents",
        response_model=DocumentsResponse,
    )
    def list_documents() -> DocumentsResponse:
        return DocumentsResponse(
            documents=store.list_documents(),
        )

    @app.delete(
        "/documents/{doc_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_document(doc_id: str) -> Response:
        deleted = store.delete_document(doc_id)

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_id}' not found.",
            )

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
