import math
from collections.abc import Iterable
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Protocol, TypedDict

from mai_agent.catalog import (
    load_products,
    normalize_product_reference,
    product_alias_candidates,
)
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
            "ingredients": (
                f"Các thành phần nổi bật được catalog liệt kê cho {label}: "
                f"{', '.join(product['ingredients'])}. Đây không phải bảng INCI đầy đủ."
            ),
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

        alias_ids = product_alias_candidates(reference)
        if alias_ids:
            alias_products = [product for product in products if product["id"] in alias_ids]
            if len(alias_products) == 1:
                candidate = alias_products[0]
                reference_terms = set(normalized_reference.split())
                candidate_terms = set(
                    normalize_product_reference(
                        " ".join(
                            [candidate["id"], candidate["name"], candidate["category"]]
                        )
                    ).split()
                )
                # Do not let a short alias swallow a meaningful qualifier that points
                # to a different or unavailable variant.
                coverage = len(reference_terms & candidate_terms) / len(reference_terms)
                exact = candidate if coverage >= 2 / 3 else None
                if exact is None:
                    unresolved.append(reference)
                    continue
            else:
                unresolved.append(reference)
                continue
        else:
            exact = None

        exact = exact or next(
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

    @staticmethod
    def _lexical_score(query: str, chunk: ProductChunk) -> float:
        normalized_query = normalize_product_reference(query)
        query_terms = set(normalized_query.split())
        content_terms = set(normalize_product_reference(chunk["content"]).split())
        overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
        section_cues = {
            "overview": ("gia", "price", "cost", "spf", "skin", "da"),
            "ingredients": ("thanh phan", "hoat chat", "ingredient"),
            "benefits": ("cong dung", "benefit", "hieu qua"),
            "usage": ("cach dung", "dung", "buoc", "use", "apply", "dap"),
        }
        cue_bonus = 0.5 if any(
            cue in normalized_query for cue in section_cues.get(chunk["section"], ())
        ) else 0.0
        return round(overlap + cue_bonus, 6)

    def _lexical_search(
        self,
        query: str,
        *,
        allowed_ids: set[str],
        top_k: int,
    ) -> list[SearchResult]:
        candidates = [
            {**chunk, "score": self._lexical_score(query, chunk)}
            for chunk in self.chunks
            if not allowed_ids or chunk["product_id"].upper() in allowed_ids
        ]
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:top_k]

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

        allowed_ids = {product_id.upper() for product_id in product_ids or []}
        # Exact catalog resolution is stronger than semantic similarity. It also
        # keeps product facts available when the configured embedding endpoint is
        # temporarily unavailable or rejects a model name.
        if allowed_ids:
            return self._lexical_search(
                normalized_query,
                allowed_ids=allowed_ids,
                top_k=top_k,
            )

        try:
            self._ensure_index()
            query_vector = self.embeddings.embed_query(normalized_query)
        except Exception:
            return self._lexical_search(
                normalized_query,
                allowed_ids=set(),
                top_k=top_k,
            )
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
