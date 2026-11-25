
import pytest

from app.chunking import chunk_text


def test_basic_chunking():
    text = " ".join(f"w{i}" for i in range(12))

    assert chunk_text(text, size=5, overlap=2) == [
        "w0 w1 w2 w3 w4",
        "w3 w4 w5 w6 w7",
        "w6 w7 w8 w9 w10",
        "w9 w10 w11",
    ]


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_text_smaller_than_chunk_size():
    assert chunk_text("hello world", size=5, overlap=2) == [
        "hello world"
    ]


def test_exact_chunk_size():
    assert chunk_text("a b c d e", size=5, overlap=2) == [
        "a b c d e"
    ]


@pytest.mark.parametrize(
    "size, overlap",
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("hello world", size, overlap)
