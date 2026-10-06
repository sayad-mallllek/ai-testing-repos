"""Transparent lexical RAG: overlapping chunks, TF-IDF vectors, cosine ranking."""

import math
import re
from collections import Counter

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_playground.models import KnowledgeChunk, KnowledgeDocument

MAX_CHUNKS = 1000  # Deliberately a small-store implementation; do not silently truncate.
STOP_WORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or the to with".split()
)


def tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP_WORDS]


def chunk_text(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def rank_texts(query: str, texts: list[str], limit: int = 3) -> list[tuple[int, float]]:
    """A vector is a dict {word: weight}; no opaque embedding API is involved."""
    if not texts or limit < 1:
        return []
    counts = [Counter(tokenize(text)) for text in texts]
    document_frequency = Counter(word for count in counts for word in count)
    idf = {
        word: math.log((1 + len(texts)) / (1 + frequency)) + 1
        for word, frequency in document_frequency.items()
    }

    def vector(count: Counter) -> dict[str, float]:
        weights = {
            word: (1 + math.log(frequency)) * idf[word]
            for word, frequency in count.items()
            if word in idf
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {word: weight / norm for word, weight in weights.items()} if norm else {}

    query_vector = vector(Counter(tokenize(query)))
    scores = []
    for index, count in enumerate(counts):
        doc_vector = vector(count)
        score = sum(weight * doc_vector.get(word, 0) for word, weight in query_vector.items())
        if score > 0:
            scores.append((index, score))
    return sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:limit]


async def ingest_document(
    session: AsyncSession, *, slug: str, title: str, content: str, source: str
) -> KnowledgeDocument:
    document = await session.scalar(select(KnowledgeDocument).where(KnowledgeDocument.slug == slug))
    if document is None:
        document = KnowledgeDocument(slug=slug, title=title, content=content, source=source)
        session.add(document)
        await session.flush()
    else:
        document.title, document.content, document.source = title, content, source
        await session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id)
        )
    session.add_all(
        KnowledgeChunk(document_id=document.id, position=position, content=text)
        for position, text in enumerate(chunk_text(content))
    )
    await session.flush()
    return document


async def retrieve_knowledge(session: AsyncSession, query: str, limit: int = 3) -> dict:
    rows = (
        await session.execute(
            select(KnowledgeChunk, KnowledgeDocument)
            .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
            .order_by(KnowledgeDocument.slug, KnowledgeChunk.position)
            .limit(MAX_CHUNKS + 1)
        )
    ).all()
    if len(rows) > MAX_CHUNKS:
        return {
            "error": "knowledge_capacity",
            "message": "Use an indexed retriever above 1000 chunks.",
        }
    # Titles help short policy queries; returned content remains the exact stored chunk.
    texts = [f"{document.title}\n{chunk.content}" for chunk, document in rows]
    matches = []
    for index, score in rank_texts(query, texts, limit):
        chunk, document = rows[index]
        matches.append(
            {
                "citation": f"store:{document.slug}#{chunk.position}",
                "title": document.title,
                "source": document.source,
                "content": chunk.content,
                "score": round(score, 6),
            }
        )
    return {"query": query, "method": "tf-idf-cosine", "matches": matches}
