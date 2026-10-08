import math
from collections.abc import Iterable
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Protocol, TypedDict

from mai_agent.catalog import load_products, normalize_product_reference
from mai_agent.skills.product.embeddings import get_product_embeddings


class EmbeddingsClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class ProductChunk(TypedDict):
    product_id: str
    product_name: str
    section: str
    content: str


class SearchResult(ProductChunk):
    score: float


def build_product_chunks(products: list[dict[str, Any]] | None = None) -> list[ProductChunk]:
    chunks: list[ProductChunk] = []
    for product in load_products() if products is None else products:
        label = f"{product['name']} ({product['id']})"
        sections = {
            "overview": (
                f"Sản phẩm {label}. Danh mục: {product['category']}. "
                f"Phù hợp: {', '.join(product['skin_types'])}. "
                f"Giá: {product['price_vnd']} VND."
            ),
            "ingredients": f"Thành phần của {label}: {', '.join(product['ingredients'])}.",
            "benefits": f"Công dụng của {label}: {', '.join(product['benefits'])}.",
            "usage": f"Cách dùng {label}: {product['usage']}",
        }
        chunks.extend(
            {
                "product_id": product["id"],
                "product_name": product["name"],
                "section": section,
                "content": content,
            }
            for section, content in sections.items()
        )
    return chunks


def resolve_product_references(references: Iterable[str]) -> dict[str, list[Any]]:
    """Map mã/tên gần đúng về SKU trước khi retrieval."""
    products = load_products()
    resolved: list[dict[str, Any]] = []
    unresolved: list[str] = []
    seen: set[str] = set()

    for reference in references:
        normalized_reference = normalize_product_reference(reference)
        if not normalized_reference:
            continue

        exact = next(
            (
                product
                for product in products
                if normalized_reference == normalize_product_reference(product["id"])
                or normalized_reference == normalize_product_reference(product["name"])
            ),
            None,
        )
        if exact is None:
            substring_matches = [
                product
                for product in products
                if normalized_reference in normalize_product_reference(product["name"])
            ]
            if len(substring_matches) == 1:
                exact = substring_matches[0]
            elif len(substring_matches) > 1:
                unresolved.append(reference)
                continue

        if exact is None and products:
            candidates: list[tuple[float, dict[str, Any]]] = []
            for product in products:
                normalized_name = normalize_product_reference(product["name"])
                score = SequenceMatcher(None, normalized_reference, normalized_name).ratio()
                candidates.append((score, product))
            candidates.sort(key=lambda item: item[0], reverse=True)
            score, exact = candidates[0]
            runner_up_score = candidates[1][0] if len(candidates) > 1 else 0.0
            if score < 0.65 or score - runner_up_score < 0.08:
                exact = None

        if exact is None:
            unresolved.append(reference)
        elif exact["id"] not in seen:
            resolved.append({"id": exact["id"], "name": exact["name"]})
            seen.add(exact["id"])

    return {"resolved": resolved, "unresolved": unresolved}


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors phải có cùng số chiều.")
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class ProductVectorStore:
    """In-memory vector index cho catalog nhỏ, dùng Gemini Embedding."""

    def __init__(self, embeddings: EmbeddingsClient) -> None:
        self.embeddings = embeddings
        self.chunks = build_product_chunks()
        self.vectors: list[list[float]] | None = None

    def _ensure_index(self) -> None:
        if self.vectors is None:
            self.vectors = self.embeddings.embed_documents(
                [chunk["content"] for chunk in self.chunks]
            )

    def search(
        self,
        query: str,
        *,
        product_ids: list[str] | None = None,
        top_k: int = 3,
    ) -> list[SearchResult]:
        normalized_query = query.strip()
        if not normalized_query or top_k <= 0:
            return []

        self._ensure_index()
        query_vector = self.embeddings.embed_query(normalized_query)
        allowed_ids = {product_id.upper() for product_id in product_ids or []}
        candidates: list[SearchResult] = []

        for chunk, vector in zip(self.chunks, self.vectors or [], strict=True):
            if allowed_ids and chunk["product_id"].upper() not in allowed_ids:
                continue
            candidates.append(
                {
                    **chunk,
                    "score": round(_cosine_similarity(query_vector, vector), 6),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        if not allowed_ids:
            return candidates[:top_k]

        # Khi so sánh nhiều SKU, giữ evidence cho từng sản phẩm thay vì để một SKU lấn át.
        per_product = math.ceil(top_k / len(allowed_ids))
        balanced: list[SearchResult] = []
        for product_id in product_ids or []:
            product_results = [
                item for item in candidates if item["product_id"].upper() == product_id.upper()
            ]
            balanced.extend(product_results[:per_product])
        return balanced[:top_k]


@lru_cache
def get_product_vector_store() -> ProductVectorStore:
    return ProductVectorStore(get_product_embeddings())
