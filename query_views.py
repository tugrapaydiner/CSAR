"""Deterministic query view generation."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


_logger = logging.getLogger(__name__)
_rewriter_failure_warned = False


def _warn_rewriter_failure_once(error: BaseException) -> None:
    """Log the first query-rewriter failure per process; stay silent thereafter."""

    global _rewriter_failure_warned
    if _rewriter_failure_warned:
        return
    _rewriter_failure_warned = True
    _logger.warning(
        "Query rewriter failed (%s: %s); "
        "falling back to the literal query for the remainder of this process.",
        type(error).__name__,
        error,
    )


ViewKind = Literal["literal", "rewritten", "entity", "question_type"]
Complexity = Literal["simple", "moderate", "complex"]
QueryType = Literal["factual", "analytical", "comparative", "procedural", "general"]
QuestionCategory = Literal["person", "location", "temporal", "numerical", "causal", "general"]


QUESTION_WORDS = ("who", "what", "when", "where", "why", "how")
ANALYSIS_WORDS = {
    "analyze",
    "analysis",
    "assess",
    "evaluate",
    "explain",
    "implication",
    "implications",
    "impact",
    "impacts",
    "reason",
    "reasons",
    "significance",
}
COMPARATIVE_WORDS = {
    "compare",
    "comparison",
    "contrast",
    "versus",
    "vs",
    "better",
    "worse",
    "difference",
    "differences",
    "similarities",
}
PROCEDURAL_WORDS = {"how", "steps", "process", "guide", "implement", "build", "create"}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
}


@dataclass(frozen=True)
class QueryView:
    kind: ViewKind
    text: str
    weight: float


@dataclass(frozen=True)
class QueryViewSet:
    query: str
    views: tuple[QueryView, ...]
    complexity: Complexity
    query_type: QueryType
    # All-aspect weights (literal, rewritten, entity, question_type, bm25, ...)
    # stored as a tuple of pairs to keep the dataclass frozen-friendly.
    aspect_weights: tuple[tuple[str, float], ...] = ()
    # Question category drives both head-weighting (Fix 2) and answer-span
    # boosting in CSA (Fix 3). "general" preserves prior behavior.
    category: QuestionCategory = "general"

    def get(self, kind: ViewKind) -> QueryView:
        for view in self.views:
            if view.kind == kind:
                return view
        raise KeyError(kind)

    @property
    def weights(self) -> dict[str, float]:
        if self.aspect_weights:
            return dict(self.aspect_weights)
        # Back-compat: callers that built QueryViewSet without aspect_weights
        # still see a view-only weight dict.
        return {view.kind: view.weight for view in self.views}


class QueryViewGenerator:
    """Generate deterministic literal, rewritten, entity, and question views."""

    _VIEW_ORDER: tuple[ViewKind, ...] = ("literal", "rewritten", "entity", "question_type")

    def __init__(
        self,
        *,
        num_views: int = 4,
        rewriter: Callable[[str], str] | None = None,
    ) -> None:
        if num_views < 1 or num_views > len(self._VIEW_ORDER):
            raise ValueError("num_views must be between 1 and 4")
        self.num_views = num_views
        self.rewriter = rewriter

    def generate(self, query: str) -> QueryViewSet:
        normalized_query = normalize_space(query)
        query_type = classify_query_type(normalized_query)
        complexity = classify_complexity(normalized_query)
        category = classify_question_category(normalized_query)
        # Fix 2: WH-categorized queries get answer-shaped weights; "general"
        # falls through to the existing per-query-type defaults.
        if category == "general":
            weights = view_weights_for_query_type(query_type)
        else:
            weights = weights_for_question_category(category)

        all_view_text = {
            "literal": normalized_query,
            "rewritten": rewrite_query(normalized_query, self.rewriter),
            "entity": entity_view(normalized_query),
            "question_type": question_type_view(normalized_query),
        }

        selected_kinds = self._VIEW_ORDER[: self.num_views]
        views = tuple(
            QueryView(kind=kind, text=all_view_text[kind], weight=weights[kind])
            for kind in selected_kinds
        )
        # Freeze the full weight dict (5 keys including bm25) onto the view set.
        aspect_weights = tuple(sorted(weights.items()))
        return QueryViewSet(
            query=normalized_query,
            views=views,
            complexity=complexity,
            query_type=query_type,
            aspect_weights=aspect_weights,
            category=category,
        )


def generate_query_views(
    query: str,
    *,
    num_views: int = 4,
    rewriter: Callable[[str], str] | None = None,
) -> QueryViewSet:
    return QueryViewGenerator(num_views=num_views, rewriter=rewriter).generate(query)


def normalize_space(text: str) -> str:
    return " ".join(text.strip().split())


def rewrite_query(query: str, rewriter: Callable[[str], str] | None = None) -> str:
    if rewriter is None:
        return query

    try:
        rewritten = normalize_space(rewriter(query))
    except Exception as error:
        _warn_rewriter_failure_once(error)
        return query
    return rewritten or query


def classify_query_type(query: str) -> QueryType:
    tokens = set(tokenize_lower(query))
    if tokens & COMPARATIVE_WORDS:
        return "comparative"
    if tokens & ANALYSIS_WORDS:
        return "analytical"
    if tokens & PROCEDURAL_WORDS and "how" in tokens:
        return "procedural"
    if any(query.lower().startswith(f"{word} ") for word in ("who", "when", "where")):
        return "factual"
    if query.endswith("?") and len(tokens) <= 8:
        return "factual"
    return "general"


def classify_complexity(query: str) -> Complexity:
    tokens = tokenize_lower(query)
    token_count = len(tokens)
    lowered = query.lower()
    score = 0

    if token_count > 18:
        score += 2
    elif token_count > 8:
        score += 1

    if set(tokens) & (ANALYSIS_WORDS | COMPARATIVE_WORDS):
        score += 1
    if count_clauses(lowered) >= 2:
        score += 1
    if len([word for word in QUESTION_WORDS if word in tokens]) > 1:
        score += 1

    if score >= 3:
        return "complex"
    if score >= 1:
        return "moderate"
    return "simple"


def classify_question_category(query: str) -> QuestionCategory:
    """Detect the answer shape from the question word (Fix 2 / Fix 3)."""

    lowered = query.lower()
    tokens = set(tokenize_lower(query))
    if "how" in tokens and any(token in tokens for token in ("long", "old")):
        return "temporal"
    if "how" in tokens and any(token in tokens for token in ("many", "much")):
        return "numerical"
    if any(token in tokens for token in ("who", "whom", "whose")):
        return "person"
    if "where" in tokens:
        return "location"
    if any(token in tokens for token in ("when",)):
        return "temporal"
    if "why" in tokens or "because" in lowered:
        return "causal"
    return "general"


def weights_for_question_category(category: QuestionCategory) -> dict[str, float]:
    """Per-aspect weights tuned to the expected answer shape (Fix 2).

    Entity- and BM25-heavy for fact-shaped queries (person/location/temporal/
    numerical), since the answer almost always sits in a sentence carrying the
    surface form of the entity. Causal queries lean on rewritten + literal
    semantic similarity because the answer phrase is rarely lexical.
    """

    if category in ("person", "location", "temporal", "numerical"):
        return {
            "literal": 0.10, "rewritten": 0.10, "entity": 0.40,
            "question_type": 0.10, "bm25": 0.40,
        }
    if category == "causal":
        return {
            "literal": 0.40, "rewritten": 0.40, "entity": 0.10,
            "question_type": 0.10, "bm25": 0.10,
        }
    # "general" handled by view_weights_for_query_type fall-through.
    return view_weights_for_query_type("general")


def view_weights_for_query_type(query_type: QueryType) -> dict[str, float]:
    """Per-aspect weights including the BM25 aspect (Fix 1)."""

    weights_by_type: dict[QueryType, dict[str, float]] = {
        "factual": {
            "literal": 0.25, "rewritten": 0.15, "entity": 0.40,
            "question_type": 0.20, "bm25": 0.50,
        },
        "analytical": {
            "literal": 0.20, "rewritten": 0.40, "entity": 0.15,
            "question_type": 0.25, "bm25": 0.50,
        },
        "comparative": {
            "literal": 0.20, "rewritten": 0.30, "entity": 0.20,
            "question_type": 0.30, "bm25": 0.50,
        },
        "procedural": {
            "literal": 0.25, "rewritten": 0.25, "entity": 0.15,
            "question_type": 0.35, "bm25": 0.50,
        },
        "general": {
            "literal": 0.35, "rewritten": 0.25, "entity": 0.20,
            "question_type": 0.20, "bm25": 0.50,
        },
    }
    return weights_by_type[query_type]


def entity_view(query: str) -> str:
    return " ".join(extract_query_entities(query))


def extract_query_entities(query: str) -> list[str]:
    quoted = extract_quoted_strings(query)
    capitalized = extract_capitalized_terms(query)
    noun_phrases = extract_content_phrases(query)
    return dedupe_preserving_order([*quoted, *capitalized, *noun_phrases])


def question_type_view(query: str) -> str:
    tokens = tokenize_lower(query)
    question_terms = [word for word in QUESTION_WORDS if word in tokens]
    content_terms = key_content_terms(query, limit=6)
    signature = dedupe_preserving_order([*question_terms, *content_terms])
    return " ".join(signature)


def extract_quoted_strings(query: str) -> list[str]:
    matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
    return [normalize_space(first or second) for first, second in matches if first or second]


def extract_capitalized_terms(query: str) -> list[str]:
    terms: list[str] = []
    current: list[str] = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_+-]*\b", query):
        if token.lower() in QUESTION_WORDS:
            continue
        current.append(token)
    if current:
        terms.extend(current)
    return terms


def extract_content_phrases(query: str) -> list[str]:
    lowered = query.lower()
    phrases: list[str] = []

    for pattern in (
        r"\b(?:of|about|regarding|for|on)\s+([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,3})",
        r"\b(?:created|invented|founded|built)\s+([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,2})",
    ):
        for match in re.finditer(pattern, lowered):
            phrase_tokens = [
                token
                for token in tokenize_lower(match.group(1))
                if token not in STOP_WORDS and token not in ANALYSIS_WORDS
            ]
            if phrase_tokens:
                phrases.append(" ".join(phrase_tokens))

    return phrases


def key_content_terms(query: str, *, limit: int) -> list[str]:
    terms = [
        token
        for token in tokenize_lower(query)
        if token not in STOP_WORDS and len(token) > 1
    ]
    return dedupe_preserving_order(terms)[:limit]


def tokenize_lower(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def count_clauses(lowered_query: str) -> int:
    separators = re.findall(r"[,;:]|\b(?:and|but|while|because|although|however)\b", lowered_query)
    return len(separators)


def dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = normalize_space(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped
