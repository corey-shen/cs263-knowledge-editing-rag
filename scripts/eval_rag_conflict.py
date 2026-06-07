"""
RAG-vs-ROME conflict evaluation.

This experiment asks whether a ROME-edited model follows edited internal
knowledge or retrieved documents when the two conflict. It evaluates the same
queries before and after each ROME edit under three prompt conditions:
no context, context consistent with the edit, and context conflicting with it.

Usage:
    python scripts/eval_rag_conflict.py --method ROME --n_cases 5
    python scripts/eval_rag_conflict.py --method ROME --case_ids us_capital,france_capital
    python scripts/eval_rag_conflict.py --method PROMPT --case_ids us_capital --model_name distilgpt2
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmarks.rag_conflict import (
    CONDITIONS,
    aggregate_metrics,
    build_rag_prompt,
    case_to_request,
    classify_row,
    condition_context,
    flatten_details,
    iter_case_queries,
    load_records,
)


DEFAULT_DATA_PATH = "data/rag_conflict/handwritten.json"
DEFAULT_HPARAMS_PATH = "configs/ROME/gpt2-xl"
DEFAULT_PROMPT_MODEL = "distilgpt2"


def safe_key(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def partial_path_for(args: argparse.Namespace) -> Path:
    os.makedirs("results/benchmark_partials", exist_ok=True)
    data_part = safe_key(Path(args.data_path).stem)
    sample_part = safe_key(f"ids_{args.case_ids}") if args.case_ids else f"n{args.n_cases or 'all'}"
    return Path("results/benchmark_partials") / (
        f"rag_conflict_{args.method.lower()}_{data_part}_{sample_part}_seed{args.seed}_tok{args.max_new_tokens}.jsonl"
    )


def load_partial_results(path: Path) -> dict[int, dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[int(row["sample_index"])] = row["result"]
    return completed


def append_partial_result(path: Path, sample_index: int, result: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"sample_index": sample_index, "result": result}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return model, tok


def generate(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(generated, skip_special_tokens=True).strip()


def patch_easyedit_for_rome_only() -> None:
    """Avoid importing EasyEdit's optional IKE stack when only ROME is needed.

    Recent Colab runtimes can fail while importing sentence-transformers /
    torchcodec through EasyEdit's broad package imports. This keeps ROME runs
    focused on the modules this script actually uses.
    """
    easyedit_root = Path(__file__).resolve().parent.parent / "external" / "EasyEdit" / "easyeditor"
    package_init = easyedit_root / "__init__.py"
    models_init = easyedit_root / "models" / "__init__.py"
    if package_init.exists():
        package_init.write_text("from .models.rome.rome_hparams import ROMEHyperParams\n")
    if models_init.exists():
        models_init.write_text("")


@lru_cache(maxsize=1)
def load_rome_tools():
    patch_easyedit_for_rome_only()
    from easyeditor import ROMEHyperParams
    from easyeditor.models.rome.rome_main import apply_rome_to_model
    from easyeditor.util import nethook

    return ROMEHyperParams, apply_rome_to_model, nethook


def capture_weights(model, hparams) -> dict[str, torch.Tensor]:
    _, _, nethook = load_rome_tools()
    return {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        ).detach().clone()
        for layer in hparams.layers
    }


def restore_weights(model, weights_copy: dict[str, torch.Tensor]) -> None:
    _, _, nethook = load_rome_tools()
    with torch.no_grad():
        for name, original in weights_copy.items():
            weight = nethook.get_parameter(model, name)
            weight[...] = original.to(weight.device)


def apply_rome_edit(model, tok, hparams, request: dict[str, str]) -> None:
    _, apply_rome_to_model, _ = load_rome_tools()
    apply_rome_to_model(
        model=model,
        tok=tok,
        request=[request],
        hparams=hparams,
        return_orig_weights=False,
    )


def build_prompt_edited_prompt(base_prompt: str, request: dict[str, str] | None) -> str:
    if request is None:
        return base_prompt
    fact = f"{request['prompt']} {request['target_new']}."
    return (
        "Assume the following updated fact is true, and answer using it when relevant.\n\n"
        f"Updated fact: {fact}\n\n"
        f"{base_prompt}"
    )


def evaluate_case(
    model,
    tok,
    device: str,
    record: dict[str, Any],
    state: str,
    max_new_tokens: int,
    prompt_edit_request: dict[str, str] | None = None,
) -> dict[str, Any]:
    generations = []
    for condition in CONDITIONS:
        context = condition_context(record, condition)
        for prompt_index, query in enumerate(iter_case_queries(record)):
            model_prompt = build_rag_prompt(query, context)
            model_prompt = build_prompt_edited_prompt(model_prompt, prompt_edit_request)
            generation = generate(model, tok, model_prompt, device, max_new_tokens)
            row = classify_row(
                record=record,
                state=state,
                condition=condition,
                prompt_index=prompt_index,
                query=query,
                generation=generation,
            )
            row["model_prompt"] = model_prompt
            generations.append(row)
    return {
        "case_id": record["case_id"],
        "state": state,
        "generations": generations,
    }


def select_sample(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.case_ids:
        wanted = {case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()}
        sample = [record for record in records if str(record.get("case_id")) in wanted]
        missing = wanted - {str(record.get("case_id")) for record in sample}
        if missing:
            raise ValueError(f"case_ids not found: {', '.join(sorted(missing))}")
        return sample
    n_cases = len(records) if args.n_cases is None else min(args.n_cases, len(records))
    return random.Random(args.seed).sample(records, n_cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ROME", "PROMPT"], default="ROME",
                        help="ROME applies EasyEdit edits; PROMPT uses an in-context updated fact for a lightweight Colab result")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--hparams_path", default=DEFAULT_HPARAMS_PATH)
    parser.add_argument("--model_name", default=None,
                        help="Override model name. Defaults to ROME hparams model for ROME and distilgpt2 for PROMPT")
    parser.add_argument("--n_cases", type=int, default=None)
    parser.add_argument("--case_ids", default=None,
                        help="Optional comma-separated case_id values to run instead of random sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--runs_path", default="results/runs.jsonl")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore any matching partial-result file and start this run from scratch")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = load_records(args.data_path)
    sample = select_sample(records, args)
    if not sample:
        raise ValueError(f"No RAG conflict records selected from {args.data_path}")

    hparams = None
    if args.method == "ROME":
        assert torch.cuda.is_available(), (
            "CUDA required for ROME editing. Use --method PROMPT for a lightweight Colab/basic-result run."
        )
        ROMEHyperParams, _, _ = load_rome_tools()
        hparams = ROMEHyperParams.from_hparams(args.hparams_path)
        model_name = args.model_name or hparams.model_name
        device = f"cuda:{hparams.device}"
        print(f"Loading {model_name} for RAG-vs-ROME on {device} ...")
    else:
        model_name = args.model_name or DEFAULT_PROMPT_MODEL
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} for lightweight RAG conflict prompt baseline on {device} ...")
    model, tok = load_model(model_name, device)

    partial_path = partial_path_for(args)
    if args.no_resume and partial_path.exists():
        partial_path.unlink()
    completed = {} if args.no_resume else load_partial_results(partial_path)
    if completed:
        print(f"Loaded {len(completed)} completed cases from {partial_path}")

    for idx, record in enumerate(sample, start=1):
        sample_index = idx - 1
        if sample_index in completed:
            print(f"\nCase {idx}/{len(sample)} id={record['case_id']} already complete; skipping")
            continue

        request = case_to_request(record)
        print(f"\nCase {idx}/{len(sample)} id={record['case_id']} subject={record['subject']!r}")
        pre = evaluate_case(model, tok, device, record, "pre", args.max_new_tokens)

        if args.method == "ROME":
            original = capture_weights(model, hparams)
            try:
                apply_rome_edit(model, tok, hparams, request)
                post = evaluate_case(model, tok, device, record, "post", args.max_new_tokens)
            finally:
                restore_weights(model, original)
        else:
            post = evaluate_case(
                model,
                tok,
                device,
                record,
                "post",
                args.max_new_tokens,
                prompt_edit_request=request,
            )

        result = {
            "case_id": record["case_id"],
            "record": record,
            "request": request,
            "pre": pre,
            "post": post,
        }
        completed[sample_index] = result
        append_partial_result(partial_path, sample_index, result)

    details = [completed[i] for i in range(len(sample)) if i in completed]
    if len(details) != len(sample):
        raise RuntimeError(f"Expected {len(sample)} completed cases, found {len(details)}")

    rows = flatten_details(details)
    metrics = aggregate_metrics(rows)
    print("\nRAG-vs-ROME summary")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    os.makedirs("results/benchmark_details", exist_ok=True)
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    detail_path = Path("results/benchmark_details") / f"rag_conflict_{args.method.lower()}_{stamp}.json"
    detail_path.write_text(json.dumps(details, indent=2))

    os.makedirs(os.path.dirname(args.runs_path) or ".", exist_ok=True)
    run_record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "method": args.method,
        "model": model_name,
        "dataset": "RAGConflict-handwritten",
        "data_path": args.data_path,
        "n_samples": len(sample),
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "metrics": metrics,
        "details_path": str(detail_path),
        "partial_path": str(partial_path),
    }
    with open(args.runs_path, "a") as f:
        f.write(json.dumps(run_record) + "\n")
    print(f"Details written to {detail_path}")
    print(f"Result appended to {args.runs_path}")


if __name__ == "__main__":
    main()
