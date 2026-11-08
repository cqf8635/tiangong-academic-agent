"""Query-dependent scores using the existing RAG models; no model loading here."""
from __future__ import annotations

import math
import os
import threading
from typing import Any, Dict, List

_SCORING_LOCK = threading.Lock()

def score_evidence(query: str, evidence: List[Dict[str, Any]], rag_engine=None) -> List[Dict[str, Any]]:
    """Return scored copies in input order, without source-based confidence bonuses.

    Reranker outputs keep the same scale as local RAG's ``predict`` scores.
    Use the configured metric; unavailable or failed models stop scoring.
    ``score_method`` exposes which metric was available; these are relevance
    measures, not calibrated probabilities of factual correctness.
    """
    if not evidence:
        return []
    query = (query or "").strip()
    if not query:
        raise ValueError("相关性评分缺少检索词")
    texts = []
    for item in evidence:
        title = str(item.get("doc_name") or item.get("title") or "").strip()
        body = str(item.get("summary") or item.get("text") or "").strip()
        texts.append(f"{title}\n{body}".strip())

    if any(not text for text in texts):
        raise ValueError("候选证据缺少可评分内容")
    scores = None
    method = "embedding_cosine"
    reranker = (getattr(rag_engine, "reranker", None)
                if os.getenv("PAPER_USE_RERANKER", "0") in {"1", "true", "True"} else None)
    if os.getenv("PAPER_USE_RERANKER", "0") in {"1", "true", "True"} and reranker is None:
        raise RuntimeError("已启用重排，但重排模型不可用")
    if reranker is not None:
        try:
            with _SCORING_LOCK:
                try:
                    predictions = reranker.predict([[query, text] for text in texts], batch_size=8)
                except TypeError:
                    predictions = reranker.predict([[query, text] for text in texts])
            if hasattr(predictions, "tolist"):
                predictions = predictions.tolist()
            if isinstance(predictions, (int, float)):
                predictions = [predictions]
            values = [float(value) for value in predictions]
            if len(values) != len(texts) or not all(math.isfinite(v) for v in values):
                raise ValueError("invalid reranker scores")
            scores, method = values, "reranker"
        except Exception as exc:
            if "CUDA out of memory" in str(exc):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    with _SCORING_LOCK:
                        try:
                            predictions = reranker.predict([[query, text] for text in texts], batch_size=2)
                        except TypeError:
                            predictions = reranker.predict([[query, text] for text in texts])
                    if hasattr(predictions, "tolist"):
                        predictions = predictions.tolist()
                    if isinstance(predictions, (int, float)):
                        predictions = [predictions]
                    values = [float(value) for value in predictions]
                    if len(values) == len(texts) and all(math.isfinite(v) for v in values):
                        scores, method = values, "reranker"
                except Exception:
                    pass
            if scores is None:
                raise RuntimeError(f"证据重排失败: {exc}") from exc

    embedding_model = getattr(rag_engine, "embedding_model", None)
    if scores is None and query and embedding_model is not None:
        try:
            with _SCORING_LOCK:
                vectors = embedding_model.encode([query, *texts], normalize_embeddings=True)
            vectors = [[float(value) for value in vector] for vector in vectors]
            if len(vectors) != len(texts) + 1 or not vectors[0]:
                raise ValueError("invalid embedding count")
            reference = vectors[0]
            values = []
            for vector in vectors[1:]:
                if len(vector) != len(reference) or not all(math.isfinite(v) for v in [*reference, *vector]):
                    raise ValueError("invalid embedding vector")
                norm = math.sqrt(sum(v * v for v in reference) * sum(v * v for v in vector))
                if not norm:
                    raise ValueError("zero embedding vector")
                similarity = sum(a * b for a, b in zip(reference, vector)) / norm
                values.append(max(0.0, min(1.0, similarity)))
            scores, method = values, "embedding_cosine"
        except Exception as exc:
            raise RuntimeError(f"证据向量评分失败: {exc}") from exc

    if scores is None:
        raise RuntimeError("语义评分模型不可用，未使用词面评分代替")

    return [
        {**item, "score": score, "score_method": method}
        for item, text, score in zip(evidence, texts, scores)
    ]
