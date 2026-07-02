from dataclasses import dataclass, field


@dataclass
class ExtractedPage:
    url: str
    domain: str
    title: str
    text: str
    ad_ratio: float
    crawl_date: str  # ISO 8601 date, e.g. "2026-07-01"
    simhash: int


@dataclass
class Chunk:
    url: str
    title: str
    domain: str
    chunk_text: str
    embedding: list[float]
    ad_ratio: float
    domain_quality: float
    crawl_date: str
    score: float = 0.0


@dataclass
class Answer:
    text: str
    citations: list[int]
    sources: list["Chunk"]
    related_questions: list[str] = field(default_factory=list)
