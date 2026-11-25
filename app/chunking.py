
def chunk_text(text: str, size: int = 50, overlap: int = 10) -> list[str]:
    """Split text into overlapping chunks of words."""

    if size <= 0:
        raise ValueError("size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = text.split()

    if not words:
        return []

    chunks = []
    step = size - overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + size]

        chunks.append(" ".join(chunk))

        if start + size >= len(words):
            break

    return chunks
