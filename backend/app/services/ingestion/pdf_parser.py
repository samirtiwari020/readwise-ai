import re
import statistics
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

MONOSPACE_FLAG = 1 << 3
BOLD_FLAG = 1 << 4


@dataclass(slots=True)
class TextBlock:
    """Structured layout block extracted from a single PDF page."""

    text: str
    page_number: int
    block_type: str  # "heading" | "narrative" | "list" | "table" | "code"
    font_size: float
    is_bold: bool
    heading_level: int | None = None


@dataclass(slots=True)
class ParsedPage:
    """Cleaned blocks and heading cues for a single 1-indexed PDF page."""

    page_number: int
    blocks: list[TextBlock] = field(default_factory=list)
    has_text: bool = True


@dataclass(slots=True)
class ParsedPdfMetadata:
    """Document-level metadata and Table of Contents extracted from a PDF."""

    page_count: int
    pdf_title: str | None
    author: str | None
    subject: str | None
    toc: list[dict]
    body_font_size: float
    repeating_edge_patterns: set[str]


def _normalize_edge_text(text: str) -> str:
    """Normalize numbers and whitespace to match repeating page headers/footers."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\d+", "#", collapsed).lower()


def _is_standalone_page_number(text: str) -> bool:
    """Return True if text looks like an isolated header/footer page number."""
    cleaned = text.strip()
    return bool(re.fullmatch(r"(?:page\s*)?\d{1,5}(?:\s*/\s*\d{1,5})?", cleaned, re.IGNORECASE))


def _sample_page_indices(page_count: int, max_samples: int = 80) -> list[int]:
    """Return evenly spaced 0-based page indices for fast profiling of 1,000+ page PDFs."""
    if page_count <= max_samples:
        return list(range(page_count))
    step = page_count / max_samples
    return sorted({min(int(i * step), page_count - 1) for i in range(max_samples)})


def _extract_block_text_and_style(block: dict) -> tuple[str, float, bool, bool]:
    """Extract normalized text, dominant font size, bold ratio, and monospace ratio."""
    lines_text: list[str] = []
    font_sizes: list[float] = []
    bold_chars = 0
    mono_chars = 0
    total_chars = 0

    for line in block.get("lines", []):
        span_parts: list[str] = []
        for span in line.get("spans", []):
            txt = span.get("text", "")
            if not txt:
                continue
            span_parts.append(txt)
            char_len = len(txt.strip())
            if char_len > 0:
                total_chars += char_len
                size = float(span.get("size", 11.0))
                font_sizes.append(size)
                flags = int(span.get("flags", 0))
                font_name = str(span.get("font", "")).lower()
                if (flags & BOLD_FLAG) or "bold" in font_name:
                    bold_chars += char_len
                if (flags & MONOSPACE_FLAG) or any(
                    hint in font_name for hint in ("mono", "courier", "consolas", "code")
                ):
                    mono_chars += char_len
        joined = "".join(span_parts).rstrip()
        if joined:
            lines_text.append(joined)

    full_text = "\n".join(lines_text).strip()
    dominant_size = statistics.median(font_sizes) if font_sizes else 11.0
    is_bold = total_chars > 0 and (bold_chars / total_chars) >= 0.65
    is_mono = total_chars > 0 and (mono_chars / total_chars) >= 0.65
    return full_text, dominant_size, is_bold, is_mono


def _classify_block(
    text: str,
    font_size: float,
    body_font_size: float,
    is_bold: bool,
    is_mono: bool,
) -> tuple[str, int | None]:
    """Classify block into heading, code, list, table, or narrative."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    word_count = len(text.split())

    # Heading heuristics: significantly larger font or short bold standalone title
    if word_count <= 20 and len(lines) <= 3:
        if font_size >= body_font_size * 1.35:
            return "heading", 1
        if font_size >= body_font_size * 1.16 or (is_bold and font_size >= body_font_size * 1.05):
            return "heading", 2

    if is_mono and len(lines) >= 2:
        return "code", None

    # List detection (bullets or numbered items on majority of lines)
    list_lines = sum(
        1 for ln in lines if re.match(r"^\s*(?:[-*•◦▪]|\d+[.)]|\([a-zA-Z0-9]+\))\s+", ln)
    )
    if len(lines) >= 2 and (list_lines / len(lines)) >= 0.5:
        return "list", None

    # Table-like multi-column / numeric alignment detection
    tabular_lines = sum(
        1 for ln in lines if ("|" in ln or re.search(r"\S\s{3,}\S", ln))
    )
    if len(lines) >= 2 and (tabular_lines / len(lines)) >= 0.5:
        return "table", None

    return "narrative", None


def profile_pdf(pdf_path: Path) -> ParsedPdfMetadata:
    """Profile a PDF for TOC, body font size, and repeating header/footer patterns."""
    with pymupdf.open(str(pdf_path)) as doc:
        page_count = doc.page_count
        raw_meta = doc.metadata or {}
        raw_toc = doc.get_toc(simple=True) or []
        toc = [
            {
                "level": int(entry[0]),
                "title": str(entry[1]).strip(),
                "page": max(1, int(entry[2])),
            }
            for entry in raw_toc
            if len(entry) >= 3 and str(entry[1]).strip()
        ]

        sampled_indices = _sample_page_indices(page_count)
        font_samples: list[float] = []
        edge_counter: Counter[str] = Counter()

        for idx in sampled_indices:
            page = doc.load_page(idx)
            page_height = float(page.rect.height or 792.0)
            page_dict = page.get_text("dict", sort=True)
            seen_edges_on_page: set[str] = set()

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                text, font_size, _is_bold, _is_mono = _extract_block_text_and_style(block)
                if not text:
                    continue
                if len(text.split()) >= 8:
                    font_samples.append(font_size)

                bbox = block.get("bbox", (0, 0, 0, 0))
                y0, y1 = float(bbox[1]), float(bbox[3])
                if y0 < page_height * 0.10 or y1 > page_height * 0.90:
                    norm = _normalize_edge_text(text)
                    if norm and len(norm) <= 120:
                        seen_edges_on_page.add(norm)

            for norm in seen_edges_on_page:
                edge_counter[norm] += 1

        body_font_size = statistics.median(font_samples) if font_samples else 11.0
        threshold = max(3, int(len(sampled_indices) * 0.35))
        repeating_edges = {
            pattern for pattern, count in edge_counter.items() if count >= threshold
        }

        return ParsedPdfMetadata(
            page_count=page_count,
            pdf_title=(raw_meta.get("title") or "").strip() or None,
            author=(raw_meta.get("author") or "").strip() or None,
            subject=(raw_meta.get("subject") or "").strip() or None,
            toc=toc,
            body_font_size=body_font_size,
            repeating_edge_patterns=repeating_edges,
        )


def iter_parsed_pages(pdf_path: Path, profile: ParsedPdfMetadata) -> Iterator[ParsedPage]:
    """Stream cleaned, layout-classified text blocks page-by-page (1-indexed)."""
    with pymupdf.open(str(pdf_path)) as doc:
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            page_height = float(page.rect.height or 792.0)
            page_dict = page.get_text("dict", sort=True)
            blocks: list[TextBlock] = []

            for raw_block in page_dict.get("blocks", []):
                if raw_block.get("type") != 0:
                    continue
                text, font_size, is_bold, is_mono = _extract_block_text_and_style(raw_block)
                if not text:
                    continue

                bbox = raw_block.get("bbox", (0, 0, 0, 0))
                y0, y1 = float(bbox[1]), float(bbox[3])
                in_margin = y0 < page_height * 0.10 or y1 > page_height * 0.90
                if in_margin:
                    if _is_standalone_page_number(text):
                        continue
                    if _normalize_edge_text(text) in profile.repeating_edge_patterns:
                        continue

                block_type, heading_level = _classify_block(
                    text=text,
                    font_size=font_size,
                    body_font_size=profile.body_font_size,
                    is_bold=is_bold,
                    is_mono=is_mono,
                )
                blocks.append(
                    TextBlock(
                        text=text,
                        page_number=page_idx + 1,
                        block_type=block_type,
                        font_size=font_size,
                        is_bold=is_bold,
                        heading_level=heading_level,
                    )
                )

            yield ParsedPage(
                page_number=page_idx + 1,
                blocks=blocks,
                has_text=len(blocks) > 0,
            )

