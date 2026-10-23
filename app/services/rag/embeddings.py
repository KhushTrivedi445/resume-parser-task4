import hashlib
import re
from math import sqrt

EMBEDDING_DIMENSION = 256
TOKEN_RE = re.compile(r"[a-z0-9+#.]+", re.I)


def embed_text(text: str) -> list[float]:
    """Create a deterministic signed feature-hash embedding without external models."""
    vector = [0.0] * EMBEDDING_DIMENSION
    for token in TOKEN_RE.findall(text.casefold()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions")
    return sum(a * b for a, b in zip(left, right))
