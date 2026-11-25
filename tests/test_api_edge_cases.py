
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_empty_documents_rejected(client):
    response = client.post(
        "/ingest",
        json={"documents": []},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "document",
    [
        {"id": "", "text": "Valid text"},
        {"id": "doc1", "text": ""},
        {"id": "   ", "text": "Valid text"},
        {"id": "doc1", "text": "   "},
    ],
)
def test_invalid_documents_rejected(client, document):
    response = client.post(
        "/ingest",
        json={"documents": [document]},
    )
    assert response.status_code == 422


def test_ask_without_documents_returns_404(client):
    response = client.post(
        "/ask",
        json={"question": "What is Python?"},
    )
    assert response.status_code == 404


@pytest.mark.parametrize("top_k", [0, 11, -1])
def test_invalid_top_k_rejected(client, top_k):
    response = client.post(
        "/ask",
        json={
            "question": "What is Python?",
            "top_k": top_k,
        },
    )
    assert response.status_code == 422


def test_empty_question_rejected(client):
    response = client.post(
        "/ask",
        json={"question": "   "},
    )
    assert response.status_code == 422


def test_document_replacement(client):
    client.post(
        "/ingest",
        json={
            "documents": [
                {"id": "doc1", "text": "Old document content."}
            ]
        },
    )

    response = client.post(
        "/ingest",
        json={
            "documents": [
                {"id": "doc1", "text": "New document content."}
            ]
        },
    )

    assert response.status_code == 200

    documents = client.get("/documents").json()["documents"]

    assert documents == [{"id": "doc1", "chunks": 1}]

    result = client.post(
        "/ask",
        json={"question": "New document content"},
    )

    assert result.status_code == 200
    assert result.json()["sources"][0]["text"] == (
        "New document content."
    )


def test_delete_document(client):
    client.post(
        "/ingest",
        json={
            "documents": [
                {"id": "doc1", "text": "Python programming language."}
            ]
        },
    )

    response = client.delete("/documents/doc1")

    assert response.status_code == 204
    assert response.content == b""

    assert client.get("/documents").json() == {
        "documents": []
    }

    assert client.post(
        "/ask",
        json={"question": "Python"},
    ).status_code == 404


def test_delete_nonexistent_document(client):
    response = client.delete("/documents/unknown")

    assert response.status_code == 404


def test_app_instances_are_isolated():
    client1 = TestClient(create_app())
    client2 = TestClient(create_app())

    client1.post(
        "/ingest",
        json={
            "documents": [
                {"id": "doc1", "text": "Python is a language."}
            ]
        },
    )

    assert len(
        client1.get("/documents").json()["documents"]
    ) == 1

    assert client2.get("/documents").json() == {
        "documents": []
    }


def test_documents_are_sorted(client):
    client.post(
        "/ingest",
        json={
            "documents": [
                {"id": "zebra", "text": "Zebra document."},
                {"id": "apple", "text": "Apple document."},
                {"id": "banana", "text": "Banana document."},
            ]
        },
    )

    documents = client.get("/documents").json()["documents"]

    assert [doc["id"] for doc in documents] == [
        "apple",
        "banana",
        "zebra",
    ]
