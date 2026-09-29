from mai_agent.skills.product.vector_store import (
    ProductVectorStore,
    build_product_chunks,
    resolve_product_references,
)


class FakeEmbeddings:
    keywords = ["khô", "dầu", "phục hồi", "làm sáng", "thành phần", "cách dùng"]

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(keyword)) for keyword in cls.keywords] + [1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_semantic_chunks_never_mix_product_ids() -> None:
    chunks = build_product_chunks()

    assert chunks
    assert {chunk["section"] for chunk in chunks} == {
        "overview",
        "ingredients",
        "benefits",
        "usage",
    }
    assert all(chunk["product_id"] in chunk["content"] for chunk in chunks)


def test_product_name_resolution_handles_typo_and_rejects_ambiguity() -> None:
    resolved = resolve_product_references(["toner cap am oa hydrating tone"])
    ambiguous = resolve_product_references(["serum"])

    assert resolved["resolved"][0]["id"] == "OA003"
    assert ambiguous["resolved"] == []
    assert ambiguous["unresolved"] == ["serum"]


def test_comparison_retrieval_keeps_evidence_for_each_product() -> None:
    store = ProductVectorStore(FakeEmbeddings())

    results = store.search(
        "So sánh công dụng phục hồi và làm sáng",
        product_ids=["OA004", "OA005"],
        top_k=4,
    )

    assert {result["product_id"] for result in results} == {"OA004", "OA005"}
