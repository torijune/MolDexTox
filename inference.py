from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover

    def load_dotenv(*_a: Any, **_k: Any) -> bool:
        return False

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(it: Any, **_k: Any) -> Any:
        return it

_MOLDETOX_ROOT = Path(__file__).resolve().parent
_TOX_AGENT_ROOT = _MOLDETOX_ROOT.parent
_DEFAULT_ENV = _TOX_AGENT_ROOT / ".env"
_DEFAULT_MERGED_TEST = (
    _MOLDETOX_ROOT
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
    / "test.csv"
)

# Default QA pack: Build_MolDeTox_QA writes under ace_safe_ver/QA/qa_sets/MolDeTox_QA/
# (hierarchical .../task*/both_repre/<step>/*.jsonl). Legacy flat test_task1_*.jsonl is also supported.
_DEFAULT_QA_PACK_ROOT = _TOX_AGENT_ROOT / "ace_safe_ver" / "QA" / "qa_sets" / "MolDeTox_QA"

# OpenAI json_schema (same shapes as inference_gpt.py)
JSON_SCHEMA = {
    "name": "molde_tox_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}

JSON_SCHEMA_STEPWISE_COT = {
    "name": "task3_stepwise_cot_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "step1_only_toxic_safe_fragments": {"type": "string"},
            "step1_reasoning": {"type": "string"},
            "step2_only_nontoxic_safe_fragments": {"type": "string"},
            "step2_reasoning": {"type": "string"},
            "step3_reasoning": {"type": "string"},
        },
        "required": [
            "answer",
            "step1_only_toxic_safe_fragments",
            "step1_reasoning",
            "step2_only_nontoxic_safe_fragments",
            "step2_reasoning",
            "step3_reasoning",
        ],
        "additionalProperties": False,
    },
}

_TASK_SHORT_TO_QA_FOLDER = {
    "task1": "task1_toxic_fragment_identification",
    "task2": "task2_nontoxic_fragment_generation",
    "task3": "task3_nontoxic_smiles_generation",
    "task3_nontoxic_safe_generation": "task3_nontoxic_safe_generation",
    "task3_stepwise_cot_safe_generation": "task3_stepwise_cot_nontoxic_safe_generation",
}

_ALLOWED_TASKS = frozenset(_TASK_SHORT_TO_QA_FOLDER.keys())


def _normalize_step(step: str) -> str:
    if step in ("single", "single_step"):
        return "single_step"
    if step in ("multi", "multi_step"):
        return "multi_step"
    return step


def _split_subdir(split: str, *, unseen_test: bool) -> str:
    if unseen_test:
        return "unseen_test"
    if split not in ("train", "test"):
        raise ValueError(f"split must be train|test (or use --unseen-test), got {split!r}")
    return split


def _is_moldeox_q_flat_bundle(qa_root: Path) -> bool:
    """True when QA matches MolDeTox_QA-style flat filenames (train_*/test_* prefix)."""
    return (
        (qa_root / "test" / "test_task1_single.jsonl").is_file()
        or (qa_root / "train" / "train_task1_single.jsonl").is_file()
    )


def _moldeox_q_flat_relative_jsonl(task_short: str, split: str, step_norm: str, variant: str) -> Optional[str]:
    """Relative path under qa_root/<split>/ for flat bundles (variant ``base`` only)."""
    if variant != "base" or split not in ("train", "test"):
        return None
    slab = "single" if step_norm == "single_step" else "multi"
    mapping = {
        "task1": f"{split}_task1_{slab}.jsonl",
        "task2": f"{split}_task2_{slab}.jsonl",
        "task3": f"{split}_task3_smiles_gen_{slab}.jsonl",
        "task3_nontoxic_safe_generation": f"{split}_task3_safe_gen_{slab}.jsonl",
    }
    return mapping.get(task_short)


def _qa_filename(task_short: str, variant: str) -> str:
    folder = _TASK_SHORT_TO_QA_FOLDER[task_short]
    if task_short == "task1":
        base = "task1_toxic_fragment_identification_qa"
    elif task_short == "task2":
        base = "task2_nontoxic_fragment_generation_qa"
    elif task_short == "task3":
        base = "task3_nontoxic_smiles_generation_qa"
    elif task_short == "task3_nontoxic_safe_generation":
        base = "task3_nontoxic_safe_generation_qa"
    else:
        base = "task3_stepwise_cot_nontoxic_safe_generation_qa"
    if variant == "base":
        return f"{base}.jsonl"
    return f"{base}_{variant}.jsonl"


def resolve_qa_jsonl(
    *,
    qa_root: Path,
    split: str,
    task_short: str,
    variant: str,
    step: str,
    unseen_test: bool,
) -> Path:
    qa_root_res = qa_root.expanduser().resolve()
    step_norm = _normalize_step(step)
    if (
        not unseen_test
        and _is_moldeox_q_flat_bundle(qa_root_res)
    ):
        rel = _moldeox_q_flat_relative_jsonl(task_short, split, step_norm, variant)
        if rel is not None:
            return qa_root_res / split / rel

    split_dir = _split_subdir(split, unseen_test=unseen_test)
    sub = _TASK_SHORT_TO_QA_FOLDER[task_short]
    fname = _qa_filename(task_short, variant)
    return qa_root_res / split_dir / sub / "both_repre" / step_norm / fname


def _strip_jsonl_line(line: str) -> str:
    s = line.strip()
    if s.startswith("$"):
        s = s[1:].lstrip()
    return s


def read_jsonl(
    path: Path,
    *,
    skip_bad_lines: bool = False,
    allow_missing: bool = False,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        if allow_missing:
            print(f"[WARN] missing JSONL: {path.resolve()}", file=sys.stderr)
            return []
        raise FileNotFoundError(f"JSONL not found: {path.resolve()}")
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = _strip_jsonl_line(line)
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                if skip_bad_lines:
                    print(f"[WARN] skip line {lineno} in {path}: {e}", file=sys.stderr)
                    continue
                raise
    return rows


def extract_gold(row: Dict[str, Any]) -> Any:
    return row.get("answer", "")


def extract_question(row: Dict[str, Any]) -> str:
    return str(row.get("question", ""))


def normalize_answer(ans: Any) -> str:
    if isinstance(ans, dict):
        return str(ans.get("answer", "") or "").strip()
    return str(ans or "").strip()


def parse_model_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(text[l : r + 1])
            except json.JSONDecodeError:
                return None
        return None


def _common_system_instruction() -> str:
    return (
        "You are a molecular toxicity reasoning assistant specialized in SAFE and SMILES representations.\n"
        "Follow the task instruction exactly and return ONLY the requested JSON object.\n"
        "Do not add explanations, markdown, code fences, or prose outside the JSON.\n"
        "Do not add extra keys unless explicitly required.\n"
    )


def _system_instruction_for_task(task: str) -> str:
    base = _common_system_instruction()
    if task == "task1":
        return (
            base
            + "Your task is to identify the fragment(s) in the toxic molecule that are most likely associated with toxicity.\n"
            + "Return the toxic-only SAFE fragment string exactly.\n"
            + "If there are multiple fragments, return them as a dot-separated SAFE string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )
    if task == "task2":
        return (
            base
            + "Your task is to generate the non-toxic replacement fragment(s) corresponding to the toxic fragment(s).\n"
            + "Return the non-toxic-only SAFE fragment string exactly.\n"
            + "If there are multiple fragments, return them as a dot-separated SAFE string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )
    if task == "task3_nontoxic_safe_generation":
        return (
            base
            + "Your task is to generate the resulting full non-toxic molecule in SAFE representation.\n"
            + "Return the complete non-toxic SAFE string for the whole molecule.\n"
            + "If there are multiple fragments, return them as a dot-separated SAFE string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )
    if task == "task3":
        return (
            base
            + "Your task is to generate the final non-toxic molecule as a single SMILES string.\n"
            + "Preserve the original molecular characteristics as much as possible while reducing toxicity.\n"
            + "Return only the final non-toxic molecule SMILES string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )
    if task == "task3_stepwise_cot_safe_generation":
        return (
            "You are a molecular toxicity reasoning assistant specialized in SAFE and SMILES representations.\n"
            "Solve the task through explicit intermediate reasoning steps.\n"
            "Return ONLY a single JSON object.\n"
            "Do not add markdown, code fences, or any text outside the JSON.\n"
            'The JSON must include "answer" as the final full non-toxic SAFE string for the whole molecule.\n'
            "Also include the required step1/step2 fragment fields and reasoning fields exactly as instructed in the prompt.\n"
        )
    return base


def call_openai_json(
    client: Any,
    model: str,
    question: str,
    system_instruction: str,
    *,
    json_schema: Optional[dict] = None,
    max_retries: int = 3,
    sleep_s: float = 0.5,
    temperature: float = 0.7,
) -> Tuple[Optional[Any], str]:
    last_err = None
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": question},
    ]
    schema_arg = json_schema or JSON_SCHEMA
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_schema", "json_schema": schema_arg},
                )
            except TypeError as te:
                if "response_format" in str(te):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                    )
                else:
                    raise
            raw = (resp.choices[0].message.content if resp.choices else "") or ""
            obj = parse_model_json(raw)
            if obj and "answer" in obj:
                return obj, raw
            return (raw.strip() if raw else None), raw
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (attempt + 1))
    return None, f"ERROR: {last_err}"


def _load_done_ids(predictions_path: Path) -> set:
    done: set = set()
    if not predictions_path.exists():
        return done
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = _strip_jsonl_line(line)
            if not line:
                continue
            try:
                obj = json.loads(line)
                i = obj.get("id")
                raw = str(obj.get("raw", "") or "")
                pred = obj.get("pred")
                is_error = raw.startswith("ERROR:")
                is_empty = pred is None or pred == "" or pred == {}
                if i is not None and (not is_error) and (not is_empty):
                    done.add(i)
            except (json.JSONDecodeError, TypeError):
                continue
    return done


def _ensure_eval_import():
    if str(_MOLDETOX_ROOT) not in sys.path:
        sys.path.insert(0, str(_MOLDETOX_ROOT))
    import evaluation as ev  # noqa: PLC0415

    return ev


def metrics_for_row(
    ev: Any,
    task: str,
    step: str,
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int],
) -> Dict[str, Any]:
    step_n = _normalize_step(step)
    if task == "task1":
        return ev.task1_toxic_fragment_identification_eval(
            gold_answer, llm_answer, step=step_n  # type: ignore[arg-type]
        )
    if task == "task2":
        return ev.task2_nontoxic_fragment_generation_eval(
            gold_answer,
            llm_answer,
            step=step_n,  # type: ignore[arg-type]
            row_id=row_id,
            context_row=None,
        )
    if task == "task3":
        return ev.task3_nontoxic_smiles_generation_eval(gold_answer, llm_answer, row_id=row_id)
    if task == "task3_nontoxic_safe_generation":
        return ev.task3_nontoxic_safe_generation_eval(gold_answer, llm_answer, row_id=row_id)
    if task == "task3_stepwise_cot_safe_generation":
        return ev.task3_stepwise_cot_nontoxic_safe_generation_eval(
            gold_answer, llm_answer, row_id=row_id
        )
    raise ValueError(f"unknown task for metrics: {task!r}")


def task_metric_keys(ev: Any, task: str, step: str) -> List[str]:
    return ev.metric_keys_for(task, _normalize_step(step))


def write_evaluation_summary(
    ev: Any,
    *,
    predictions_path: Path,
    eval_dir: Path,
    task: str,
    step: str,
    model: str,
    variant_suffix: str,
    inference_time_seconds: float,
    inference_samples: int,
    split_label: str,
) -> Path:
    step_norm = _normalize_step(step)
    safe_model = model.replace("/", "_")
    rows = read_jsonl(predictions_path, skip_bad_lines=True, allow_missing=True)
    keys = task_metric_keys(ev, task, step_norm)
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    correct = 0
    for line in rows:
        correct += int(line.get("correct", 0))
        m = {k: line.get(k) for k in keys if k in line and line.get(k) is not None}
        if len(m) < len(keys):
            rid = ev.row_id_for_merged_lookup(line)
            ga = line.get("gold")
            pr = line.get("pred")
            if task == "task3_stepwise_cot_safe_generation":
                qa_a = line.get("qa_answer")
                if isinstance(qa_a, dict) and qa_a:
                    gold_answer = qa_a
                else:
                    gold_answer = ga if isinstance(ga, dict) else {"answer": ga}
                llm_answer = pr if isinstance(pr, dict) else {"answer": pr}
            else:
                gold_answer = ga if isinstance(ga, dict) else {"answer": ga}
                llm_answer = pr if isinstance(pr, dict) else {"answer": pr}
            m = metrics_for_row(ev, task, step_norm, gold_answer, llm_answer, rid)
        ev.merge_finite_metrics_into_sums(metric_sums, metric_counts, m)
    total = len(rows)
    metric_means = ev.mean_metrics_from_sums(metric_sums, metric_counts, keys)
    metric_means = ev.augment_metrics_mean_with_em_accuracy(metric_means)
    summary: Dict[str, Any] = {
        "task": task,
        "step": step_norm,
        "split": split_label,
        "model": model,
        "variant": variant_suffix or "base",
        "inference_time_seconds": round(inference_time_seconds, 3),
        "inference_samples": inference_samples,
        "inference_seconds_per_sample": (
            round(inference_time_seconds / inference_samples, 6) if inference_samples > 0 else None
        ),
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "metrics_mean": metric_means,
        "predictions_path": str(predictions_path.resolve()),
    }
    name = f"evaluation_summary_{safe_model}"
    if variant_suffix:
        name = f"{name}_{variant_suffix}"
    out = eval_dir / f"{name}.json"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)
    return out


def _call_one_row(
    client: Any,
    model: str,
    row: Dict[str, Any],
    task: str,
    system_instruction: str,
    *,
    max_retries: int,
    sleep_s: float,
    temperature: float,
) -> Tuple[Dict[str, Any], Any, str]:
    q = extract_question(row)
    schema = (
        JSON_SCHEMA_STEPWISE_COT if task == "task3_stepwise_cot_safe_generation" else None
    )
    pred, raw = call_openai_json(
        client,
        model,
        q,
        system_instruction,
        json_schema=schema,
        max_retries=max_retries,
        sleep_s=sleep_s,
        temperature=temperature,
    )
    return row, pred, raw


def run_inference_job(
    *,
    qa_path: Path,
    out_dir: Path,
    task: str,
    step: str,
    split_label: str,
    model: str,
    variant: str,
    merged_split_csv: Optional[Path],
    num_samples: int,
    batch_size: int,
    max_retries: int,
    sleep_s: float,
    temperature: float,
    reset: bool,
) -> Tuple[Path, Path]:
    ev = _ensure_eval_import()
    if merged_split_csv is not None:
        ev.configure_eval_data_paths(merged_split_csv=str(merged_split_csv))

    if OpenAI is None:
        raise RuntimeError("Missing OpenAI SDK: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    step_norm = _normalize_step(step)
    rows = read_jsonl(qa_path)
    if num_samples > 0:
        rows = rows[:num_samples]

    variant_suffix = "" if variant == "base" else variant
    name_parts = [variant_suffix] if variant_suffix else []
    model_safe = model.replace("/", "_")
    pred_name = (
        f"predictions_{model_safe}.jsonl"
        if not name_parts
        else f"predictions_{model_safe}_{'_'.join(name_parts)}.jsonl"
    )

    task_out = out_dir / split_label / task / "both_repre" / step_norm
    results_dir = task_out / "results"
    eval_dir = task_out / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / pred_name

    if reset and pred_path.is_file():
        pred_path.unlink()

    done = set() if reset else _load_done_ids(pred_path)
    rows_todo = [r for r in rows if r.get("id") not in done]
    if done:
        print(f"[resume] {len(done)} done, {len(rows_todo)} remaining")

    system_instruction = _system_instruction_for_task(task)
    t0 = time.perf_counter()
    n_infer = len(rows_todo)
    mode = "w" if reset or not pred_path.exists() or not done else "a"

    def write_result(wf: Any, row: Dict[str, Any], pred: Any, raw: str) -> None:
        gold = extract_gold(row)
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(gold)
        is_correct = int(pred_norm == gold_norm)
        gold_answer = row.get("answer", gold)
        llm_answer = pred if isinstance(pred, dict) else {"answer": pred or ""}
        rid = ev.row_id_for_merged_lookup(row)
        metrics = metrics_for_row(ev, task, step_norm, gold_answer, llm_answer, rid)
        out_row: Dict[str, Any] = {
            "model": model,
            "task": task,
            "step": step_norm,
            "id": row.get("id"),
            "dataset_name": row.get("dataset_name", ""),
            "endpoint": row.get("endpoint") or row.get("dataset_name", "") or "",
            "source_index": row.get("source_index"),
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "raw": raw,
        }
        if task == "task3_stepwise_cot_safe_generation" and isinstance(row.get("answer"), dict):
            out_row["qa_answer"] = row.get("answer")
        out_row.update(metrics)
        wf.write(json.dumps(out_row, ensure_ascii=False, default=str) + "\n")
        wf.flush()

    with pred_path.open(mode, encoding="utf-8") as wf:
        bs = max(batch_size, 1)
        for batch_start in tqdm(
            range(0, len(rows_todo), bs),
            desc=f"{task}/{step_norm}",
            total=(len(rows_todo) + bs - 1) // bs,
        ):
            batch = rows_todo[batch_start : batch_start + bs]
            slots: List[Optional[Tuple[Dict[str, Any], Any, str]]] = [None] * len(batch)
            nxt = 0
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                fmap = {
                    ex.submit(
                        _call_one_row,
                        client,
                        model,
                        row,
                        task,
                        system_instruction,
                        max_retries=max_retries,
                        sleep_s=sleep_s,
                        temperature=temperature,
                    ): i
                    for i, row in enumerate(batch)
                }
                for fut in as_completed(fmap):
                    i = fmap[fut]
                    slots[i] = fut.result()
                while nxt < len(batch) and slots[nxt] is not None:
                    row, pred, raw = slots[nxt]  # type: ignore[misc]
                    write_result(wf, row, pred, raw)
                    nxt += 1
            if sleep_s > 0:
                time.sleep(sleep_s)

    elapsed = time.perf_counter() - t0
    summary_path = write_evaluation_summary(
        ev,
        predictions_path=pred_path,
        eval_dir=eval_dir,
        task=task,
        step=step_norm,
        model=model,
        variant_suffix=variant_suffix,
        inference_time_seconds=elapsed,
        inference_samples=n_infer,
        split_label=split_label,
    )
    print(f"predictions -> {pred_path}")
    print(f"summary     -> {summary_path}")
    return pred_path, summary_path


def _parse_tasks(s: str) -> List[str]:
    s = (s or "").strip()
    if s == "all":
        return list(_TASK_SHORT_TO_QA_FOLDER.keys())
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[str] = []
    seen: set = set()
    for p in parts:
        if p not in _ALLOWED_TASKS:
            raise ValueError(f"unknown task {p!r}; allowed: {sorted(_ALLOWED_TASKS)} or all")
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--env",
        type=Path,
        default=_DEFAULT_ENV,
        help="Path to .env (default: ToxAgent/.env)",
    )
    ap.add_argument(
        "--qa-root",
        type=Path,
        default=_DEFAULT_QA_PACK_ROOT,
        help=(
            "QA pack root (default: ToxAgent/ace_safe_ver/QA/qa_sets/MolDeTox_QA). "
            "If QA was built with Build_MolDeTox_QA --no_desc, use .../MolDeTox_QA_no_desc."
        ),
    )
    ap.add_argument(
        "--merged-split-csv",
        type=Path,
        default=_DEFAULT_MERGED_TEST,
        help=(
            "Split CSV with toxic_smiles for PRS and related metrics (passed to "
            "evaluation.configure_eval_data_paths; auxiliary scores may be 0 if absent)."
        ),
    )
    ap.add_argument(
        "--no-merged-split-csv",
        action="store_true",
        help="Do not load merged CSV (disables PRS-style auxiliary metrics).",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MolDeTox GPT inference and evaluation summaries (evaluation.py).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pin = sub.add_parser("infer", help="Run GPT on QA jsonl and write predictions + aggregate summary.")
    _add_common_args(pin)
    pin.add_argument("--model", type=str, default=os.environ.get("MOLDETOX_GPT_MODEL", "gpt-4.1-mini"))
    pin.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="QA subdirectory: train or test",
    )
    pin.add_argument(
        "--unseen-test",
        action="store_true",
        help="Read QA from qa-root/unseen_test/... (split label unseen_test)",
    )
    pin.add_argument(
        "--task",
        type=str,
        default="all",
        help="Comma-separated task names or all",
    )
    pin.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "icl1", "icl2", "icl4"],
        help="QA filename suffix: base -> *_qa.jsonl, icl4 -> *_qa_icl4.jsonl",
    )
    pin.add_argument(
        "--step",
        type=str,
        default="single_step",
        choices=["single_step", "multi_step", "single", "multi"],
    )
    pin.add_argument(
        "--out-dir",
        type=Path,
        default=_MOLDETOX_ROOT / "inference_outputs",
        help="Inference output root (default: MolDeTox/inference_outputs)",
    )
    pin.add_argument("--num-samples", type=int, default=0, help="0 means all samples")
    pin.add_argument("--batch-size", type=int, default=8)
    pin.add_argument("--max-retries", type=int, default=3)
    pin.add_argument("--sleep", type=float, default=0.25, help="Sleep between batches (seconds)")
    pin.add_argument("--temperature", type=float, default=0.7)
    pin.add_argument("--reset", action="store_true", help="Delete existing predictions and rerun from scratch")
    pin.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Explicit QA jsonl path (requires a single task in --task, not all)",
    )

    pev = sub.add_parser(
        "evaluate",
        help="Rebuild evaluation_summary from an existing predictions_*.jsonl only",
    )
    _add_common_args(pev)
    pev.add_argument("--predictions", type=Path, required=True)
    pev.add_argument("--task", type=str, required=True, choices=sorted(_ALLOWED_TASKS))
    pev.add_argument(
        "--step",
        type=str,
        default="single_step",
        choices=["single_step", "multi_step", "single", "multi"],
    )
    pev.add_argument("--model", type=str, default="model", help="Model name stored in summary JSON")
    pev.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Default: sibling of predictions dir: ../evaluation/",
    )

    args = parser.parse_args()
    load_dotenv(args.env.expanduser())

    merged = None if args.no_merged_split_csv else args.merged_split_csv

    if args.cmd == "evaluate":
        ev = _ensure_eval_import()
        if getattr(args, "no_merged_split_csv", False) or merged is None:
            ev.configure_eval_data_paths(merged_split_csv=None)
        else:
            ev.configure_eval_data_paths(merged_split_csv=str(merged))
        pred_path = args.predictions.expanduser().resolve()
        eval_dir = (
            args.eval_dir.expanduser().resolve()
            if args.eval_dir
            else pred_path.parent.parent / "evaluation"
        )
        step_norm = _normalize_step(args.step)
        summary = write_evaluation_summary(
            ev,
            predictions_path=pred_path,
            eval_dir=eval_dir,
            task=args.task,
            step=step_norm,
            model=args.model,
            variant_suffix="",
            inference_time_seconds=0.0,
            inference_samples=0,
            split_label="",
        )
        print(summary)
        return 0

    if getattr(args, "no_merged_split_csv", False):
        merged_infer = None
    else:
        merged_infer = merged

    tasks = _parse_tasks(args.task)
    split_label = "unseen_test" if args.unseen_test else args.split

    if args.data is not None:
        raw_task = (args.task or "").strip()
        if raw_task == "all":
            print("Error: with --data, pass a single task in --task (e.g. task1), not all.", file=sys.stderr)
            return 2
        task_list = _parse_tasks(raw_task)
        if len(task_list) != 1:
            print("Error: with --data, specify exactly one task in --task.", file=sys.stderr)
            return 2
        t_one = task_list[0]
        qa_path = args.data.expanduser().resolve()
        print(f"[QA] {qa_path} (task={t_one})")
        run_inference_job(
            qa_path=qa_path,
            out_dir=args.out_dir.expanduser().resolve(),
            task=t_one,
            step=args.step,
            split_label=split_label,
            model=args.model,
            variant=args.variant,
            merged_split_csv=merged_infer,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            sleep_s=args.sleep,
            temperature=args.temperature,
            reset=args.reset,
        )
        return 0

    for t in tasks:
        qa_path = resolve_qa_jsonl(
            qa_root=args.qa_root.expanduser().resolve(),
            split=args.split,
            task_short=t,
            variant=args.variant,
            step=args.step,
            unseen_test=args.unseen_test,
        )
        print(f"[QA] {qa_path}")
        run_inference_job(
            qa_path=qa_path,
            out_dir=args.out_dir.expanduser().resolve(),
            task=t,
            step=args.step,
            split_label=split_label,
            model=args.model,
            variant=args.variant,
            merged_split_csv=merged_infer,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            sleep_s=args.sleep,
            temperature=args.temperature,
            reset=args.reset,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
