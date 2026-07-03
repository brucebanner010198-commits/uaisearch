import time
from dataclasses import dataclass, field


@dataclass
class IngestStats:
    """Throughput counters for one ingest run (crawl cycle or backfill)."""

    started: float = field(default_factory=time.perf_counter)
    pages: int = 0
    chunks: int = 0

    def summary(self, index_docs: int | None = None) -> str:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        parts = [
            f"pages={self.pages}",
            f"chunks={self.chunks}",
            f"elapsed_s={elapsed:.1f}",
            f"pages_per_s={self.pages / elapsed:.2f}",
            f"chunks_per_s={self.chunks / elapsed:.2f}",
        ]
        if index_docs is not None:
            parts.append(f"index_docs={index_docs}")
        return "ingest-stats " + " ".join(parts)
