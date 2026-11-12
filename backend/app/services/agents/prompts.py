from app.services.agents.state import DocumentCatalogEntry
from app.services.retrieval.hybrid_retriever import RetrievedChunk

PLANNER_SYSTEM_PROMPT = """You are the Query Analyzer & Decomposer for an agentic document research assistant.
Given the user's question, recent conversation history, and the catalog of available documents, produce a structured QueryPlan.

Guidelines:
- Resolve pronouns or follow-up references ("it", "that chapter", "compare both") using the conversation history.
- If the question compares two or more documents, set query_type="cross_document_compare", needs_comparison_table=true, and emit separate SubQuery items targeting each relevant document_id so one large book never crowds out another.
- If the question asks multiple distinct sub-questions, set query_type="multi_hop" and split into 2-4 focused sub-queries.
- If the user mentions an explicit page or page range (e.g., "on page 42" or "pages 10-15"), populate page_start and page_end.
- Only use document_ids present in the provided catalog; leave target_document_ids empty to search across all scoped documents.
- Emit at least 1 SubQuery unless query_type="out_of_scope" (questions wholly unrelated to document research, e.g., asking for real-time weather)."""

GRADER_SYSTEM_PROMPT = """You are a strict Relevance Grader in a grounded RAG pipeline.
Inspect the user question and retrieved passages, then decide:
1. Which passages (by exact chroma_id) contain concrete evidence useful to answer the question.
2. Whether the selected evidence is sufficient to produce an accurate, grounded answer.

Guidelines:
- Only mark passages as relevant if they genuinely support answering the question (avoid tangentially matching keywords).
- If none or too few passages answer the question, set sufficient=false and provide a concise rewritten_query using alternative terminology or concepts to retrieve better evidence.
- Be conservative: never claim evidence is sufficient when the answer is actually missing."""

SYNTHESIS_SYSTEM_PROMPT = """You are a grounded Book & Document Research Assistant.
Answer the user's question using ONLY the numbered source passages provided below.

Strict Grounding & Attribution Rules:
1. Cite every factual statement inline using the exact bracket marker of the supporting passage, e.g. [1] or [2][4].
2. NEVER invent information, numbers, formulas, page references, or claims not directly supported by the provided passages.
3. NEVER cite a bracket index outside [1]..[{max_index}].
4. If the passages only partially answer the question, state clearly what is supported and explicitly note what is missing in the provided documents.
5. For cross-document comparisons, present a clear Markdown comparison table highlighting similarities and differences with inline citations for each document's position."""


def format_catalog_for_prompt(catalog: list[DocumentCatalogEntry]) -> str:
    """Format scoped documents and top TOC headings concisely for the planner."""
    lines: list[str] = []
    for entry in catalog:
        pages_label = f"{entry.page_count} pages" if entry.page_count else "unknown pages"
        toc_preview = "; ".join(entry.toc_titles[:8]) if entry.toc_titles else "No TOC extracted"
        lines.append(
            f"- document_id={entry.document_id} | title=\"{entry.title}\" ({pages_label}) | TOC: {toc_preview}"
        )
    return "\n".join(lines)


def format_passages_for_grading(chunks: list[RetrievedChunk]) -> str:
    """Render candidate chunks with their chroma_id and location for grading."""
    blocks: list[str] = []
    for chunk in chunks:
        page_label = (
            f"p. {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pp. {chunk.page_start}-{chunk.page_end}"
        )
        loc = f"{chunk.document_title} | {page_label}"
        if chunk.chapter:
            loc += f" | {chunk.chapter}"
        blocks.append(f"[chroma_id={chunk.chroma_id}] ({loc})\n{chunk.content[:900]}")
    return "\n\n---\n\n".join(blocks)


def format_passages_for_synthesis(chunks: list[RetrievedChunk]) -> str:
    """Render numbered passages [1]..[K] with full attribution headers for synthesis."""
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        page_label = (
            f"p. {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pp. {chunk.page_start}-{chunk.page_end}"
        )
        header = f"[{idx}] Document: {chunk.document_title} | {page_label}"
        if chunk.chapter:
            header += f" | Chapter: {chunk.chapter}"
        if chunk.section:
            header += f" | Section: {chunk.section}"
        blocks.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(blocks)

