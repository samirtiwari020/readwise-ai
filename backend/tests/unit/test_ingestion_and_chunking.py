from pathlib import Path

import pymupdf

from app.services.ingestion.adaptive_chunker import AdaptiveChunker
from app.services.ingestion.pdf_parser import iter_parsed_pages, profile_pdf
from app.services.retrieval.vector_store import ChromaVectorStore


def _build_structured_pdf(pdf_path: Path) -> None:
    """Create a realistic 4-page PDF with TOC, headers/footers, prose, list, and table."""
    doc = pymupdf.open()

    # Page 1: Chapter 1 narrative + repeating header/footer
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((50, 35), "Distributed Systems Handbook", fontsize=9)
    p1.insert_text((50, 110), "Chapter 1: Vector Retrieval", fontsize=18)
    p1.insert_text(
        (50, 150),
        "Dense vector embeddings project semantic meaning into continuous spaces. "
        "Page-aware chunking preserves structural boundaries across large books so "
        "every claim traces cleanly back to its exact origin page.",
        fontsize=11,
    )
    p1.insert_text((280, 760), "Page 1", fontsize=9)

    # Page 2: Subsection + bullet list
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((50, 35), "Distributed Systems Handbook", fontsize=9)
    p2.insert_text((50, 100), "1.1 Hybrid Search Benefits", fontsize=14)
    p2.insert_text(
        (50, 135),
        "- Combines dense cosine similarity with sparse BM25 lexical matching\n"
        "- Preserves exact theorem names, acronyms, and identifiers\n"
        "- Boosts recall across multi-document queries\n"
        "- Feeds high-precision candidates to cross-encoder rerankers",
        fontsize=11,
    )
    p2.insert_text((280, 760), "Page 2", fontsize=9)

    # Page 3: Chapter 2 + table-like rows
    p3 = doc.new_page(width=612, height=792)
    p3.insert_text((50, 35), "Distributed Systems Handbook", fontsize=9)
    p3.insert_text((50, 100), "Chapter 2: Agentic Orchestration", fontsize=18)
    p3.insert_text(
        (50, 145),
        "Model | Dimensions | Use Case\n"
        "bge-small | 384 | Fast CPU retrieval\n"
        "bge-base | 768 | Balanced accuracy\n"
        "bge-reranker | 1 | Pairwise relevance scoring",
        fontsize=11,
    )
    p3.insert_text((280, 760), "Page 3", fontsize=9)

    # Page 4: Chapter 2 continuation
    p4 = doc.new_page(width=612, height=792)
    p4.insert_text((50, 35), "Distributed Systems Handbook", fontsize=9)
    p4.insert_text(
        (50, 110),
        "Query decomposition splits multi-hop questions into targeted sub-queries "
        "executed per document scope, preventing longer books from crowding out shorter ones.",
        fontsize=11,
    )
    p4.insert_text((280, 760), "Page 4", fontsize=9)

    doc.set_toc(
        [
            [1, "Chapter 1: Vector Retrieval", 1],
            [2, "1.1 Hybrid Search Benefits", 2],
            [1, "Chapter 2: Agentic Orchestration", 3],
        ]
    )
    doc.save(str(pdf_path))
    doc.close()


def test_pdf_parser_strips_headers_footers_and_adaptive_chunker_preserves_pages(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "handbook.pdf"
    _build_structured_pdf(pdf_path)

    profile = profile_pdf(pdf_path)
    assert profile.page_count == 4
    assert len(profile.toc) == 3
    assert profile.toc[0]["title"] == "Chapter 1: Vector Retrieval"

    pages = list(iter_parsed_pages(pdf_path, profile))
    assert len(pages) == 4
    combined_clean_text = "\n".join(b.text for p in pages for b in p.blocks)
    assert "Distributed Systems Handbook" not in combined_clean_text
    assert "Page 1" not in combined_clean_text
    assert "Page 4" not in combined_clean_text

    chunker = AdaptiveChunker(
        document_id="doc-1",
        document_title="Distributed Systems Handbook",
        toc=profile.toc,
        min_tokens=10,
        target_tokens=120,
        max_tokens=200,
        overlap_tokens=15,
    )
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) >= 3

    # Chapter 1 chunks stay on pages 1-2; Chapter 2 chunks start on page 3
    ch1_chunks = [c for c in chunks if c.chapter == "Chapter 1: Vector Retrieval"]
    ch2_chunks = [c for c in chunks if c.chapter == "Chapter 2: Agentic Orchestration"]
    assert ch1_chunks and ch2_chunks
    assert max(c.page_end for c in ch1_chunks) <= 2
    assert min(c.page_start for c in ch2_chunks) >= 3

    # Dense list/table classification detected accurately
    chunk_types = {c.chunk_type for c in chunks}
    assert "list" in chunk_types or "table" in chunk_types
    assert "[Distributed Systems Handbook | Chapter 1: Vector Retrieval" in ch1_chunks[0].contextual_text


def test_chroma_vector_store_filtering_and_deletion(tmp_path: Path) -> None:
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma", collection_name="test_chunks")
    store.upsert_chunks(
        ids=["doc-a:0", "doc-a:1", "doc-b:0"],
        documents=["alpha p1", "alpha p10", "beta p2"],
        embeddings=[[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.95, 0.05, 0.0]],
        metadatas=[
            {"document_id": "doc-a", "page_start": 1, "page_end": 1},
            {"document_id": "doc-a", "page_start": 10, "page_end": 11},
            {"document_id": "doc-b", "page_start": 2, "page_end": 2},
        ],
    )
    assert store.count() == 3

    # Filter by single document_id
    res_a = store.query([1.0, 0.0, 0.0], top_k=5, document_ids=["doc-a"])
    assert {r["metadata"]["document_id"] for r in res_a} == {"doc-a"}

    # Filter by document_id + page range
    res_p10 = store.query([1.0, 0.0, 0.0], top_k=5, document_ids=["doc-a"], page_start=9, page_end=12)
    assert [r["chroma_id"] for r in res_p10] == ["doc-a:1"]

    # Delete removes only target document chunks
    store.delete_document_chunks("doc-a")
    assert store.count() == 1
    res_after = store.query([1.0, 0.0, 0.0], top_k=5)
    assert [r["chroma_id"] for r in res_after] == ["doc-b:0"]

