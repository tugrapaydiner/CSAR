"""Streamlit UI for the CSAR compression pipeline."""

from __future__ import annotations

import html
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import streamlit as st

from pipeline import CompressionConfig, compress_document_for_query, text_length
from query_views import generate_query_views


CACHE_DIR = Path(".csar_cache")

MODE_PRESETS = {
    "Aggressive": {
        "description": "Smallest output for direct factual lookups.",
        "config": {
            "half_block_size": 96,
            "sliding_window_chunks": 1,
            "sentence_keep_ratio": 0.35,
            "hca_max_words": 14,
            "top_k_ratio_override": 0.15,
            "raw_score_blend": 0.5,
        },
    },
    "Moderate": {
        "description": "Balanced compression for everyday question answering.",
        "config": {
            "half_block_size": 96,
            "sliding_window_chunks": 2,
            "sentence_keep_ratio": 0.5,
            "hca_max_words": 20,
            "top_k_ratio_override": 0.3,
            "raw_score_blend": 0.5,
        },
    },
    "Light": {
        "description": "Keeps more detail for broad or analytical questions.",
        "config": {
            "half_block_size": 128,
            "sliding_window_chunks": 3,
            "sentence_keep_ratio": 0.7,
            "hca_max_words": 28,
            "top_k_ratio_override": 0.45,
            "raw_score_blend": 0.35,
        },
    },
}

EXAMPLE_DOCUMENT = """Python began as a hobby programming project by Guido van Rossum at Centrum Wiskunde and Informatica in the Netherlands.
Van Rossum started implementation in December 1989 while looking for a successor to the ABC language.
Python 0.9.0 was released in February 1991 and already included classes, exceptions, functions, and core data types.
The language name Python came from Monty Python's Flying Circus rather than from the snake.
Python's design philosophy emphasizes readability, explicit code, and the idea that there should be one obvious way to do it.
The Python Software Foundation was created in 2001 to hold intellectual property and support the Python community.
CPython is the reference implementation of Python and is written primarily in the C programming language.
PyPy is an alternative Python implementation known for its just-in-time compiler.
Python 2 reached end of life on January 1 2020 after a long migration period toward Python 3."""


def main() -> None:
    st.set_page_config(
        page_title="CSAR Compression Workbench",
        page_icon="",
        layout="wide",
    )
    inject_styles()

    st.title("CSAR Compression Workbench")
    st.caption("Query-aware context compression with HCA summaries, CSA extraction, Sinkhorn balancing, and optional two-tier cache.")

    with st.sidebar:
        st.header("Compression")
        mode = st.radio(
            "Mode",
            list(MODE_PRESETS),
            index=1,
            captions=[MODE_PRESETS[name]["description"] for name in MODE_PRESETS],
        )
        use_cache = st.toggle("Use two-tier cache", value=True)
        auto_short_inputs = st.toggle("Auto-fit short contexts", value=True)

        with st.expander("Advanced controls"):
            base = MODE_PRESETS[mode]["config"]
            half_block_size = st.slider("Half-block token target", 16, 256, int(base["half_block_size"]), step=8)
            sliding_window_chunks = st.slider("Recent chunks kept verbatim", 0, 8, int(base["sliding_window_chunks"]))
            sentence_keep_ratio = st.slider("CSA sentence keep ratio", 0.1, 1.0, float(base["sentence_keep_ratio"]), step=0.05)
            hca_max_words = st.slider("HCA summary words", 4, 32, int(base["hca_max_words"]))
            top_k_ratio = st.slider("Top-k chunk ratio", 0.05, 1.0, float(base["top_k_ratio_override"]), step=0.05)
            abstain_threshold = st.slider("Abstain threshold", 0.0, 0.25, 0.05, step=0.01)
            raw_score_blend = st.slider("Raw score blend", 0.0, 1.0, float(base["raw_score_blend"]), step=0.05)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Input")
        query = st.text_input("Query", value="Who created Python?")
        if st.button("Load example document"):
            st.session_state["document"] = EXAMPLE_DOCUMENT
        document = st.text_area(
            "Long context",
            value=st.session_state.get("document", EXAMPLE_DOCUMENT),
            height=470,
            placeholder="Paste the long context you want to compress.",
        )
        st.session_state["document"] = document

    config = CompressionConfig(
        half_block_size=half_block_size,
        sliding_window_chunks=sliding_window_chunks,
        sentence_keep_ratio=sentence_keep_ratio,
        hca_max_words=hca_max_words,
        abstain_threshold=abstain_threshold,
        top_k_ratio_override=top_k_ratio,
        raw_score_blend=raw_score_blend,
    )

    with right:
        st.subheader("Output")
        run = st.button("Compress", type="primary", use_container_width=True)
        if run:
            if not document.strip() or not query.strip():
                st.error("Provide both a query and context.")
            else:
                render_result(document, query, config, use_cache, auto_short_inputs)
        else:
            st.info("Adjust the controls, then run compression.")


def render_result(
    document: str,
    query: str,
    config: CompressionConfig,
    use_cache: bool,
    auto_short_inputs: bool,
) -> None:
    original_words = text_length(document)
    short_mode_applied = False
    if auto_short_inputs and original_words < 300:
        config = replace(
            config,
            sliding_window_chunks=0,
            # Raised from 8 → 20: the extractive HCA summarizer aims for
            # grammatical complete sentences, not aggressive truncation.
            hca_max_words=min(config.hca_max_words, 20),
            top_k_ratio_override=min(config.top_k_ratio_override or 0.15, 0.15),
        )
        short_mode_applied = True

    # Phase 1 fix 4.1: catch pipeline exceptions and surface them rather than
    # leaking a raw stack trace into the right column.
    try:
        with st.spinner("Running V4 compression pipeline…"):
            started = perf_counter()
            result = compress_document_for_query(
                document,
                query,
                config=config,
                cache=CACHE_DIR if use_cache else None,
            )
            elapsed_ms = (perf_counter() - started) * 1000
    except Exception as error:  # noqa: BLE001 — user-facing surface
        st.error(f"Compression failed: {type(error).__name__}: {error}")
        return

    # Phase 1 fix 4.3: distinguish "everything filtered out" from a pipeline crash.
    if not result.compressed_text.strip():
        st.warning(
            "Compression returned an empty result. Try lowering the abstain threshold "
            "or raising the top-k chunk ratio in Advanced controls."
        )
        return

    compressed_words = text_length(result.compressed_text)
    reduction = max(0.0, 1.0 - result.compression_ratio)
    n_chunks = len(result.chunks)
    n_csa = len(result.selected_chunk_indices)
    n_sliding = config.sliding_window_chunks
    # Derived from the rendered text because CompressionResult does not expose them.
    n_summaries = result.compressed_text.count("[Summary:")
    n_omissions = result.compressed_text.count("[OMITTED:")

    try:
        complexity = generate_query_views(query).complexity
    except Exception:  # noqa: BLE001 — best-effort badge
        complexity = "unknown"

    # Headline metrics — the four numbers that prove the pipeline did its job.
    headline = st.columns(4)
    headline[0].metric("Original words", f"{original_words:,}")
    headline[1].metric("Compressed words", f"{compressed_words:,}")
    # Phase 1 fix 3.3: replace ambiguous "0.37x" with an unambiguous reduction %.
    headline[2].metric(
        "Reduction",
        f"{reduction:.0%}",
        help=f"Output is {result.compression_ratio:.0%} of the original.",
    )
    headline[3].metric("Runtime", f"{elapsed_ms:.0f} ms")

    if result.compression_ratio >= 1.0:
        st.warning(
            "Output is not smaller than input. Short contexts can expand because "
            "summaries and verbatim recent content have fixed overhead."
        )
    else:
        st.progress(
            min(1.0, reduction),
            text=f"{reduction:.0%} smaller — {result.compression_ratio:.0%} of original retained",
        )

    # Phase 1 fix 2.4 + 3.4: surface adaptive complexity AND the per-mechanism counts.
    badge_bits = [f"Detected query complexity: <strong>{html.escape(complexity)}</strong>"]
    if short_mode_applied:
        badge_bits.append("short-context mode applied")
    st.markdown(
        f'<div class="csar-badge-row">{" · ".join(badge_bits)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="csar-section-label">V4 mechanism breakdown</div>', unsafe_allow_html=True)
    breakdown = st.columns(5)
    breakdown[0].metric("Chunks", n_chunks)
    breakdown[1].metric("HCA summaries", n_summaries)
    breakdown[2].metric("CSA selected", n_csa)
    breakdown[3].metric("Recent verbatim", n_sliding)
    breakdown[4].metric("Omission markers", n_omissions)

    # Phase 2: render compressed text with styled mechanism markers instead of a
    # plain textarea so the V4 architecture is visible in the output itself.
    st.markdown('<div class="csar-section-label">Compressed context</div>', unsafe_allow_html=True)
    render_compressed_text(result.compressed_text)

    with st.expander("Show raw text (for copy/paste)"):
        st.text_area(
            "Raw compressed context",
            value=result.compressed_text,
            height=240,
            label_visibility="collapsed",
        )

    st.download_button(
        "Download compressed context",
        data=result.compressed_text,
        file_name="compressed_context.txt",
        mime="text/plain",
        use_container_width=True,
    )


def render_compressed_text(text: str) -> None:
    """Render compressed output with styled HCA / CSA / OMITTED / verbatim markers.

    Markers are conceptually important — they show the V4 mechanisms doing their
    work. Each is given its own visual treatment, with text labels and shapes
    (not color alone) so the distinction survives for color-blind users.
    """

    blocks: list[str] = []
    for raw_section in text.split("\n\n"):
        section = raw_section.strip("\n")
        if not section.strip():
            continue

        leading = section.lstrip()
        if leading.startswith("[OMITTED:"):
            inner = leading[len("[OMITTED:"):].rstrip("]").strip()
            blocks.append(
                '<div class="csar-omitted" role="note" aria-label="Omitted content marker">'
                '<span class="csar-omitted-icon" aria-hidden="true">⋯</span>'
                f'<span><strong>OMITTED</strong> · {html.escape(inner)}</span>'
                '</div>'
            )
        elif leading.startswith("[Recent content, verbatim:]"):
            blocks.append(
                '<div class="csar-section-header" role="heading" aria-level="3">'
                '▾ RECENT CONTENT — VERBATIM'
                '</div>'
            )
        elif leading.startswith("[Summary:"):
            lines = section.split("\n")
            head = lines[0].strip()
            summary_text = head[len("[Summary:"):].rstrip("]").strip()
            extraction = "\n".join(lines[1:]).strip()
            parts = [
                '<div class="csar-summary">'
                '<span class="csar-pill" aria-label="HCA summary">HCA</span> '
                f'{html.escape(summary_text)}'
                '</div>'
            ]
            if extraction:
                parts.append(
                    '<div class="csar-csa">'
                    '<span class="csar-pill csar-pill-csa" aria-label="CSA extraction">CSA</span> '
                    f'{html.escape(extraction).replace(chr(10), "<br>")}'
                    '</div>'
                )
            blocks.append("".join(parts))
        else:
            blocks.append(
                f'<div class="csar-verbatim">'
                f'{html.escape(section).replace(chr(10), "<br>")}'
                f'</div>'
            )

    rendered = '<div class="csar-compressed">' + "".join(blocks) + '</div>'
    st.markdown(rendered, unsafe_allow_html=True)


def inject_styles() -> None:
    # Palette: one accent (academic blue), one secondary (extraction green),
    # one warning (omission amber), neutral grays elsewhere. No gradients.
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            max-width: 1320px;
        }
        textarea {
            font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
            line-height: 1.45;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d7dee8;
            border-radius: 8px;
            padding: 0.55rem 0.75rem;
            background: #ffffff;
        }
        div[data-testid="stMetric"] label {
            color: #586272;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
        }
        .csar-section-label {
            margin: 1.1rem 0 0.4rem 0;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #4a5363;
        }
        .csar-badge-row {
            margin: 0.5rem 0 0.25rem 0;
            font-size: 0.85rem;
            color: #4a5363;
        }
        .csar-compressed {
            font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
            font-size: 0.92rem;
            line-height: 1.55;
            background: #fafbfc;
            border: 1px solid #d7dee8;
            border-radius: 8px;
            padding: 12px 14px;
            max-height: 520px;
            overflow-y: auto;
            color: #1a1f2b;
        }
        .csar-summary {
            background: #eef3fa;
            border-left: 3px solid #3b6db8;
            padding: 8px 12px;
            margin: 6px 0 0 0;
            border-radius: 4px 4px 0 0;
            color: #1a2b48;
        }
        .csar-csa {
            padding: 6px 12px 10px 22px;
            margin: 0 0 10px 0;
            color: #1a1f2b;
            background: #f3faf5;
            border-left: 3px solid #2d6a3e;
            border-radius: 0 0 4px 4px;
        }
        .csar-pill {
            display: inline-block;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #3b6db8;
            background: #ffffff;
            border: 1px solid #c0d0e8;
            border-radius: 999px;
            padding: 1px 7px;
            margin-right: 6px;
            vertical-align: 1px;
        }
        .csar-pill-csa {
            color: #2d6a3e;
            border-color: #b8d6c1;
        }
        .csar-omitted {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            margin: 12px 0;
            background: #fdf6e3;
            border: 1px solid #d8b76a;
            border-radius: 6px;
            color: #4a3805;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 0.85rem;
        }
        .csar-omitted-icon {
            font-size: 1.4em;
            line-height: 1;
            color: #8a6914;
        }
        .csar-section-header {
            margin: 16px 0 8px 0;
            padding: 6px 10px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            color: #2a3140;
            background: #eef0f4;
            border-left: 3px solid #4a5363;
            border-radius: 0 4px 4px 0;
        }
        .csar-verbatim {
            padding: 6px 12px;
            margin: 6px 0;
            color: #1a1f2b;
            background: #ffffff;
            border: 1px solid #e3e7ed;
            border-radius: 4px;
        }
        @media (max-width: 768px) {
            .csar-compressed { font-size: 0.86rem; max-height: 400px; }
            div[data-testid="stMetric"] { padding: 0.4rem 0.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
