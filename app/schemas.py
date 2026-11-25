
from pydantic import BaseModel, Field, field_validator


class IngestDocument(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("id", "text")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return value


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return value


class IngestResponse(BaseModel):
    ingested: int
    chunks: int


class DocumentInfo(BaseModel):
    id: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class SourceInfo(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
