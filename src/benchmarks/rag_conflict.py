from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .common import answer_variants, contains_answer, load_json, normalize_text


CONDITIONS = ("no_context", "consistent_context", "conflicting_context")
STATES = ("pre", "post")

REQUIRED_FIELDS = (
    "case_id",
    "subject",
    "relation",
    "edit_prompt",
    "original_answer",
    "edited_answer",
    "query",
    "paraphrase_queries",
    "consistent_context",
    "conflicting_context",
)


def load_records(path: str) -> list[dict[str, Any]]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"RAG conflict file must contain a JSON list: {path}")
    for idx, record in enumerate(records):
        validate_record(record, idx)
    return records


def validate_record(record: dict[str, Any], idx: int | None = None) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        prefix = f"record {idx}: " if idx is not None else ""
        raise ValueError(f"{prefix}missing required fields: {', '.join(missing)}")
    if not isinstance(record.get("paraphrase_queries"), list):
        raise ValueError(f"record {idx}: paraphrase_queries must be a list")


def case_to_request(record: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": record["edit_prompt"],
        "subject": record["subject"],
        "target_new": record["edited_answer"],
        "ground_truth": record["original_answer"],
    }


def iter_case_queries(record: dict[str, Any]) -> list[str]:
    return [record["query"]] + list(record.get("paraphrase_queries", []))


def condition_context(record: dict[str, Any], condition: str) -> str | None:
    if condition == "no_context":
        return None
    if condition not in CONDITIONS:
        raise ValueError(f"unknown RAG condition: {condition}")
    return record[condition]


def retrieved_answer_for_condition(record: dict[str, Any], condition: str) -> tuple[str | None, list[str]]:
    if condition == "no_context":
        return None, []
    if condition == "consistent_context":
        return record["edited_answer"], list(record.get("edited_aliases", []))
    if condition == "conflicting_context":
        return record["original_answer"], list(record.get("original_aliases", []))
    raise ValueError(f"unknown RAG condition: {condition}")


def build_rag_prompt(query: str, context: str | None) -> str:
    if not context:
        return f"Answer the question with a concise factual answer.\n\nQuestion: {query}\nAnswer:"
    return (
        "Answer the question with a concise factual answer. "
        "Use the retrieved context when it is relevant.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )


def normalized_answer_set(answer: Any, aliases: list[str] | None = None) -> set[str]:
    return {normalize_text(variant) for variant in answer_variants(answer, aliases)}


def answers_overlap(
    left_answer: Any,
    left_aliases: list[str] | None,
    right_answer: Any,
    right_aliases: list[str] | None,
) -> bool:
    return bool(normalized_answer_set(left_answer, left_aliases) & normalized_answer_set(right_answer, right_aliases))


def classify_generation(
    generation: str,
    record: dict[str, Any],
    retrieved_answer: str | None = None,
    retrieved_aliases: list[str] | None = None,
) -> dict[str, Any]:
    edited_hit = contains_answer(generation, record["edited_answer"], record.get("edited_aliases", []))
    original_hit = contains_answer(generation, record["original_answer"], record.get("original_aliases", []))
    retrieved_hit = False
    if retrieved_answer:
        retrieved_hit = contains_answer(generation, retrieved_answer, retrieved_aliases or [])

    edited_original_overlap = answers_overlap(
        record["edited_answer"],
        record.get("edited_aliases", []),
        record["original_answer"],
        record.get("original_aliases", []),
    )
    inconsistent = edited_hit and original_hit and not edited_original_overlap
    answer_class = answer_class_for_hits(
        edited_hit=edited_hit,
        original_hit=original_hit,
        retrieved_hit=retrieved_hit,
        inconsistent=inconsistent,
        record=record,
        retrieved_answer=retrieved_answer,
        retrieved_aliases=retrieved_aliases or [],
    )
    return {
        "edited_answer": edited_hit,
        "retrieved_answer": retrieved_hit,
        "original_answer": original_hit,
        "inconsistent_answer": inconsistent,
        "answer_class": answer_class,
    }


def answer_class_for_hits(
    *,
    edited_hit: bool,
    original_hit: bool,
    retrieved_hit: bool,
    inconsistent: bool,
    record: dict[str, Any],
    retrieved_answer: str | None,
    retrieved_aliases: list[str],
) -> str:
    if inconsistent:
        return "inconsistent"
    if retrieved_hit and retrieved_answer:
        if answers_overlap(retrieved_answer, retrieved_aliases, record["original_answer"], record.get("original_aliases", [])):
            return "retrieved"
        if answers_overlap(retrieved_answer, retrieved_aliases, record["edited_answer"], record.get("edited_aliases", [])):
            return "edited"
        return "retrieved"
    if edited_hit:
        return "edited"
    if original_hit:
        return "original"
    return "other"


def classify_row(
    record: dict[str, Any],
    state: str,
    condition: str,
    prompt_index: int,
    query: str,
    generation: str,
) -> dict[str, Any]:
    context = condition_context(record, condition)
    retrieved_answer, retrieved_aliases = retrieved_answer_for_condition(record, condition)
    classification = classify_generation(
        generation,
        record,
        retrieved_answer=retrieved_answer,
        retrieved_aliases=retrieved_aliases,
    )
    return {
        "case_id": record["case_id"],
        "subject": record["subject"],
        "relation": record["relation"],
        "state": state,
        "condition": condition,
        "prompt_index": prompt_index,
        "query": query,
        "context": context,
        "retrieved_answer": retrieved_answer,
        "generation": generation,
        "classification": classification,
    }


def mean_bool(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _rate(rows: list[dict[str, Any]], key: str, *, require_retrieved: bool = False) -> float | None:
    eligible = [
        row for row in rows
        if not require_retrieved or row.get("retrieved_answer") is not None
    ]
    return mean_bool([bool(row["classification"][key]) for row in eligible])


def _answer_class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["classification"]["answer_class"] for row in rows)
    return dict(sorted(counts.items()))


def consistency_rate(rows: list[dict[str, Any]]) -> float | None:
    grouped: dict[tuple[Any, str, str], list[str]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["state"], row["condition"])].append(
            row["classification"]["answer_class"]
        )
    if not grouped:
        return None
    consistent = [len(set(classes)) == 1 for classes in grouped.values()]
    return mean_bool(consistent)


def condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_generations": len(rows),
        "edited_answer_rate": _rate(rows, "edited_answer"),
        "retrieved_answer_rate": _rate(rows, "retrieved_answer", require_retrieved=True),
        "original_answer_rate": _rate(rows, "original_answer"),
        "inconsistent_answer_rate": _rate(rows, "inconsistent_answer"),
        "consistency_rate": consistency_rate(rows),
        "answer_class_counts": _answer_class_counts(rows),
    }


def conflict_sensitivity(rows: list[dict[str, Any]]) -> float | None:
    conflicting = [row for row in rows if row["condition"] == "conflicting_context"]
    overrides = [
        bool(row["classification"]["retrieved_answer"]) and not bool(row["classification"]["edited_answer"])
        for row in conflicting
    ]
    return mean_bool(overrides)


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_state = {state: [row for row in rows if row["state"] == state] for state in STATES}
    metrics = {
        "edited_answer_rate": _rate(by_state["post"], "edited_answer"),
        "retrieved_answer_rate": _rate(by_state["post"], "retrieved_answer", require_retrieved=True),
        "original_answer_rate": _rate(by_state["post"], "original_answer"),
        "conflict_sensitivity": conflict_sensitivity(by_state["post"]),
        "consistency_rate": consistency_rate(by_state["post"]),
        "pre_edited_answer_rate": _rate(by_state["pre"], "edited_answer"),
        "pre_retrieved_answer_rate": _rate(by_state["pre"], "retrieved_answer", require_retrieved=True),
        "pre_original_answer_rate": _rate(by_state["pre"], "original_answer"),
        "pre_conflict_sensitivity": conflict_sensitivity(by_state["pre"]),
        "pre_consistency_rate": consistency_rate(by_state["pre"]),
        "by_state_condition": {},
    }
    for state in STATES:
        metrics["by_state_condition"][state] = {}
        for condition in CONDITIONS:
            condition_rows = [row for row in by_state[state] if row["condition"] == condition]
            metrics["by_state_condition"][state][condition] = condition_metrics(condition_rows)
    return metrics


def flatten_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for detail in details:
        for state in STATES:
            rows.extend(detail.get(state, {}).get("generations", []))
    return rows
