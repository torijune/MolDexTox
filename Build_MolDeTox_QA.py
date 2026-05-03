"""
MolDeTox QA builder (JSONL).

Minimal footprint (pair CSV → QA): output CSV from ``ToxicityCliff_pairing.py`` plus this script and
``MolDetox_QA_template.py`` are enough for base QA generation (endpoint text is bundled in the template).

Optional few-shot ICL (``icl-k`` / ``all``) needs ``ICL_template.py`` beside this module.

Fixed tasks supported:
  task1, task2, task3, task3_nontoxic_safe_generation, task3_stepwise_cot_safe_generation

Representations always use ``both_repre`` subdirectory names included.

Variants:
  - ``base``: no ICL
  - ``icl-k``: few-shot ICL (--icl-k chooses K ∈ {{1,2,4}}, mapped internally to icl1/icl2/icl4)
  - ``all``: run base followed by icl-k

MCQA bundles are unsupported.
Inputs may be scaffold splits or ToxicityCliff pairing exports; decoded columns tolerate ``toxic_smiles`` /
``nontoxic_smiles`` aliases.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # optional pandas for unseen merges

_QA_DIR = Path(__file__).resolve().parent
# Ensure MolDetox_QA_template / optional ICL_template imports resolve beside this folder
if str(_QA_DIR) not in sys.path:
    sys.path.insert(0, str(_QA_DIR))


def _count_dot_separated_fragments(dot_separated: str) -> int:
    """Dot-separated SAFE token count."""
    s = (dot_separated or "").strip()
    if not s:
        return 0
    return len([p.strip() for p in s.split(".") if p.strip()])


def classify_step_task1(only_toxic_safe_fragments: str) -> str:
    """Task 1 stepping: multi when more than one toxic fragment token."""
    return "single_step" if _count_dot_separated_fragments(only_toxic_safe_fragments) == 1 else "multi_step"


def classify_step_task2_or_task3(
    only_toxic_safe_fragments: str,
    only_nontoxic_safe_fragments: str,
) -> str:
    """Task 2/3 stepping based on toxic & nontoxic fragment token counts."""
    n_t = _count_dot_separated_fragments(only_toxic_safe_fragments)
    n_nt = _count_dot_separated_fragments(only_nontoxic_safe_fragments)
    return "single_step" if n_t == 1 and n_nt == 1 else "multi_step"


# Index JSON emitted by get_icl_index.py (ICL construction without pairwise .npz files)
_DEFAULT_ICL_INDEX_JSON = _QA_DIR / "icl_train_topk_indices.json"

from MolDetox_QA_template import (
    task1_toxic_fragment_identification,
    task2_nontoxic_fragment_generation,
    task3_nontoxic_smiles_generation,
    task3_nontoxic_safe_generation,
    task3_stepwise_cot_nontoxic_safe_generation,
)
_icl_import_error = None
try:
    from ICL_template import (
        build_task1_toxic_fragment_identification_icl,
        build_task2_nontoxic_fragment_generation_icl,
        build_task3_nontoxic_smiles_generation_icl,
        build_task3_nontoxic_safe_generation_icl,
        build_task3_stepwise_cot_nontoxic_safe_generation_icl,
        build_task1_toxic_fragment_identification_icl_from_index_json,
        build_task2_nontoxic_fragment_generation_icl_from_index_json,
        build_task3_nontoxic_smiles_generation_icl_from_index_json,
        build_task3_nontoxic_safe_generation_icl_from_index_json,
        build_task3_stepwise_cot_nontoxic_safe_generation_icl_from_index_json,
    )
except Exception as e:
    _icl_import_error = e
    build_task1_toxic_fragment_identification_icl = (
        build_task2_nontoxic_fragment_generation_icl
    ) = build_task3_nontoxic_smiles_generation_icl = (
        build_task3_nontoxic_safe_generation_icl
    ) = build_task3_stepwise_cot_nontoxic_safe_generation_icl = (
        build_task1_toxic_fragment_identification_icl_from_index_json
    ) = build_task2_nontoxic_fragment_generation_icl_from_index_json = (
        build_task3_nontoxic_smiles_generation_icl_from_index_json
    ) = build_task3_nontoxic_safe_generation_icl_from_index_json = (
        build_task3_stepwise_cot_nontoxic_safe_generation_icl_from_index_json
    ) = None

# Default data paths (scaffold split train / test / unseen single CSVs under MolDeTox/splits/)
_DEFAULT_SPLIT_DIR = _QA_DIR / "splits" / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
_DEFAULT_TRAIN_CSV = _DEFAULT_SPLIT_DIR / "train.csv"
_DEFAULT_TEST_CSV = _DEFAULT_SPLIT_DIR / "test.csv"
_DEFAULT_UNSEEN_CSV = _DEFAULT_SPLIT_DIR / "unseen_endpoint_test.csv"
# Unseen merges fall back to */*/test.csv when the consolidated CSV above is unavailable
_DEFAULT_UNSEEN_SPLIT_DIR = _QA_DIR / "splits" / "scaffold_by_endpoint_property_outlier_dropped"

# Data paths (configured at runtime in main())
DATA_TASK1 = _DEFAULT_TEST_CSV  # task1_toxic_fragment_identification paths refresh at runtime via _configure_paths
DATA_TASK2 = _DEFAULT_TEST_CSV         # task2_nontoxic_fragment_generation
DATA_TASK3 = _DEFAULT_TEST_CSV         # task3_nontoxic_smiles_generation
CURRENT_SPLIT = "test"
# Locked representation subdirectory name
CURRENT_MOLECULE_REPR = "both_repre"
# Shuffle seed when non-None before writing JSONL (defaults to chronological order).
BUILD_QA_SHUFFLE_SEED: Optional[int] = None
# Append endpoint narratives to prompts (default True).
INCLUDE_ENDPOINT_DESCRIPTION: bool = True

# Output base directories (configured per split=train/test)
OUT_DIR_TASK1 = _QA_DIR / "test" / "task1_toxic_fragment_identification"
OUT_DIR_TASK2 = _QA_DIR / "test" / "task2_nontoxic_fragment_generation"
OUT_DIR_TASK3 = _QA_DIR / "test" / "task3_nontoxic_smiles_generation"
OUT_DIR_TASK3_NONToxic_SAFE_GENERATION = _QA_DIR / "test" / "task3_nontoxic_safe_generation"
OUT_DIR_TASK3_STEPWISE_COT_SAFE = _QA_DIR / "test" / "task3_stepwise_cot_nontoxic_safe_generation"

# QA output root (--qa_set rewrites under QA/qa_sets/<name>/)
QA_OUT_ROOT = _QA_DIR

# Toxicity cliff exports may alias decoded columns via toxic_smiles / nontoxic_smiles entries
REQUIRED_COLUMNS_TASK_MIN = [
    "dataset_name",
    "endpoint",
    "toxic_safe",
    "nontoxic_safe",
    "only_toxic_safe_fragments",
    "only_nontoxic_safe_fragments",
]


def _str_or_empty(val) -> str:
    if val is None:
        return ""
    # Robust handling for pandas-style NaNs
    try:
        if isinstance(val, float) and val != val:  # NaN
            return ""
    except Exception:
        pass
    if pd is not None:
        try:
            if pd.isna(val):
                return ""
        except Exception:
            pass
    return str(val).strip()


def _has_dataset_or_endpoint(row: dict) -> bool:
    """Require at least dataset_name or endpoint text."""
    dataset_name = _str_or_empty(row.get("dataset_name", ""))
    endpoint = _str_or_empty(row.get("endpoint", ""))
    return bool(dataset_name or endpoint)


def _merge_toxicity_cliff_row_aliases(row: dict) -> None:
    """Normalize decoded SMILES aliases on cliff-style CSV rows."""
    if not _str_or_empty(row.get("toxic_safe_decoded_smiles", "")):
        row["toxic_safe_decoded_smiles"] = _str_or_empty(row.get("toxic_smiles", ""))
    if not _str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")):
        row["nontoxic_safe_decoded_smiles"] = _str_or_empty(row.get("nontoxic_smiles", ""))
    row.setdefault("common_safe_fragments", "")


def _validate_pair_csv_headers(fieldnames: list[str], path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS_TASK_MIN if c not in fieldnames]
    if missing:
        raise ValueError(f"Missing column(s) in {path}: {missing}")
    if "toxic_safe_decoded_smiles" not in fieldnames and "toxic_smiles" not in fieldnames:
        raise ValueError(
            f"{path}: need column toxic_safe_decoded_smiles or toxic_smiles (ToxicityCliff_pairing output)"
        )
    if "nontoxic_safe_decoded_smiles" not in fieldnames and "nontoxic_smiles" not in fieldnames:
        raise ValueError(
            f"{path}: need column nontoxic_safe_decoded_smiles or nontoxic_smiles"
        )


def _iter_csv_rows(path: Path, _required_cols_unused: list[str] | None = None) -> tuple[list[str], list[tuple[int, dict]]]:
    """Read cliff/merged-compatible CSV pairs with enforced headers."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        _validate_pair_csv_headers(fieldnames, path)
        rows: list[tuple[int, dict]] = []
        for idx, row in enumerate(reader):
            _merge_toxicity_cliff_row_aliases(row)
            rows.append((idx, row))
        return fieldnames, rows

def _shuffle_and_reid(records: list[dict], seed: Optional[int]) -> list[dict]:
    """Shuffle records and reassign id to 0..n-1. Preserves dataset_name, endpoint, source_index."""
    if not records:
        return records
    if seed is not None:
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        for i, r in enumerate(shuffled):
            r = dict(r)
            r["id"] = i
            shuffled[i] = r
        return shuffled
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_task2():
    """Task 2: nontoxic_fragment_generation -> task2_nontoxic_fragment_generation/{single_step|multi_step}/task2_nontoxic_fragment_generation_qa.jsonl"""
    if not DATA_TASK2.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK2}")
    _, rows = _iter_csv_rows(DATA_TASK2)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = classify_step_task2_or_task3(only_toxic, only_nontoxic)

        question, answer = task2_nontoxic_fragment_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            only_nontoxic_safe_fragments=only_nontoxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
            "common_safe_fragments": _str_or_empty(row.get("common_safe_fragments", "")),
            "nontoxic_safe_decoded_smiles": _str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK2 / "single_step" / "task2_nontoxic_fragment_generation_qa.jsonl"
    out_multi = OUT_DIR_TASK2 / "multi_step" / "task2_nontoxic_fragment_generation_qa.jsonl"
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 2: single_step={len(records_single)} -> {out_single}")
    print(f"Task 2: multi_step ={len(records_multi)} -> {out_multi}")
    print(f"Task 2: skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}")
    return out_single, out_multi


def build_task1():
    """Task 1: toxic_fragment_identification -> task1_toxic_fragment_identification/{single_step|multi_step}/task1_toxic_fragment_identification_qa.jsonl"""
    if not DATA_TASK1.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK1}")
    _, rows = _iter_csv_rows(DATA_TASK1)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        step = classify_step_task1(only_toxic)
        question, answer = task1_toxic_fragment_identification(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK1 / "single_step" / "task1_toxic_fragment_identification_qa.jsonl"
    out_multi = OUT_DIR_TASK1 / "multi_step" / "task1_toxic_fragment_identification_qa.jsonl"
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 1: single_step={len(records_single)} -> {out_single}")
    print(f"Task 1: multi_step ={len(records_multi)} -> {out_multi}")
    print(f"Task 1: skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}")
    return out_single, out_multi


def build_task3():
    """Task 3: nontoxic_smiles_generation -> task3_nontoxic_smiles_generation/{single_step|multi_step}/task3_nontoxic_smiles_generation_qa.jsonl"""
    if not DATA_TASK3.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK3}")
    _, rows = _iter_csv_rows(DATA_TASK3)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = classify_step_task2_or_task3(only_toxic, only_nontoxic)

        question, answer = task3_nontoxic_smiles_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK3 / "single_step" / "task3_nontoxic_smiles_generation_qa.jsonl"
    out_multi = OUT_DIR_TASK3 / "multi_step" / "task3_nontoxic_smiles_generation_qa.jsonl"
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 3: single_step={len(records_single)} -> {out_single}")
    print(f"Task 3: multi_step ={len(records_multi)} -> {out_multi}")
    print(f"Task 3: skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}")
    return out_single, out_multi


def build_task3_nontoxic_safe_generation():
    """Task 3: nontoxic_safe_generation -> task3_nontoxic_safe_generation/{single_step|multi_step}/task3_nontoxic_safe_generation_qa.jsonl"""
    if not DATA_TASK3.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK3}")
    _, rows = _iter_csv_rows(DATA_TASK3)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = classify_step_task2_or_task3(only_toxic, only_nontoxic)

        question, answer = task3_nontoxic_safe_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            nontoxic_safe=_str_or_empty(row["nontoxic_safe"]),
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = (
        OUT_DIR_TASK3_NONToxic_SAFE_GENERATION
        / "single_step"
        / "task3_nontoxic_safe_generation_qa.jsonl"
    )
    out_multi = (
        OUT_DIR_TASK3_NONToxic_SAFE_GENERATION
        / "multi_step"
        / "task3_nontoxic_safe_generation_qa.jsonl"
    )
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 3 nontoxic safe: single_step={len(records_single)} -> {out_single}")
    print(f"Task 3 nontoxic safe: multi_step ={len(records_multi)} -> {out_multi}")
    print(
        "Task 3 nontoxic safe: "
        f"skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}"
    )
    return out_single, out_multi


def build_task3_stepwise_cot_safe_generation():
    """
    Task 3 stepwise CoT emitting full SAFE strings:
      one model message; Step 1/2 mirror SMILES-variant reasoning with SAFE fragments; final JSON answer is SAFE.
    """
    if not DATA_TASK3.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK3}")
    _, rows = _iter_csv_rows(DATA_TASK3)

    records_single: list[dict] = []
    records_multi: list[dict] = []
    skipped_missing_dataset_endpoint = 0
    for idx, row in rows:
        if not _has_dataset_or_endpoint(row):
            skipped_missing_dataset_endpoint += 1
            continue
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = classify_step_task2_or_task3(only_toxic, only_nontoxic)

        question, answer = task3_stepwise_cot_nontoxic_safe_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
            only_toxic_safe_fragments=only_toxic,
            only_nontoxic_safe_fragments=only_nontoxic,
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
            include_endpoint_description=INCLUDE_ENDPOINT_DESCRIPTION,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = (
        OUT_DIR_TASK3_STEPWISE_COT_SAFE
        / "single_step"
        / "task3_stepwise_cot_nontoxic_safe_generation_qa.jsonl"
    )
    out_multi = (
        OUT_DIR_TASK3_STEPWISE_COT_SAFE
        / "multi_step"
        / "task3_stepwise_cot_nontoxic_safe_generation_qa.jsonl"
    )
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 3 stepwise CoT (SAFE): single_step={len(records_single)} -> {out_single}")
    print(f"Task 3 stepwise CoT (SAFE): multi_step ={len(records_multi)} -> {out_multi}")
    print(
        "Task 3 stepwise CoT (SAFE): "
        f"skipped dataset/endpoint missing rows = {skipped_missing_dataset_endpoint}"
    )
    return out_single, out_multi

def _icl_variant_tag_for_k(icl_k: int) -> str:
    """Normalize K-shot settings to filenames expected by ``ICL_template`` (supports 1/2/4 only)."""
    if icl_k not in (1, 2, 4):
        raise ValueError("ICL-K: --icl-k must be 1, 2, or 4.")
    return f"icl{icl_k}"


def _configure_paths(
    split: str,
    input_csv: Path | None,
    molecule_repr: str = "both_repre",
    unseen: bool = False,
) -> None:
    """
    Configure DATA_* and OUT_DIR_* globals for the desired split/input CSV/unseen mode.

    ``molecule_repr`` is accepted only for backwards compatibility—the writer always persists ``both_repre``.
    """
    global DATA_TASK1, DATA_TASK2, DATA_TASK3
    global OUT_DIR_TASK1, OUT_DIR_TASK2, OUT_DIR_TASK3, OUT_DIR_TASK3_NONToxic_SAFE_GENERATION, OUT_DIR_TASK3_STEPWISE_COT_SAFE
    global CURRENT_SPLIT, CURRENT_MOLECULE_REPR

    CURRENT_SPLIT = split
    repr_dir = "both_repre"
    CURRENT_MOLECULE_REPR = repr_dir

    if unseen and split != "test":
        raise ValueError("--unseen is only compatible with --split test.")

    root = QA_OUT_ROOT
    if split == "train":
        data_path = input_csv or _DEFAULT_TRAIN_CSV
        split_dir = root / "train"
    else:
        data_path = input_csv or _DEFAULT_TEST_CSV
        split_dir = root / "test"

    if unseen:
        split_dir = root / "unseen_test"
        if input_csv is not None:
            data_path = input_csv
            print(f"[unseen] input CSV: {data_path}")
        elif _DEFAULT_UNSEEN_CSV.is_file():
            data_path = _DEFAULT_UNSEEN_CSV
            print(f"[unseen] default bundle CSV: {data_path}")
        else:
            # Fallback: concatenate legacy nested test.csv blobs
            unseen_root = _DEFAULT_UNSEEN_SPLIT_DIR / "unseen_endpoint_test"
            if not unseen_root.is_dir():
                raise FileNotFoundError(
                    f"Neither {_DEFAULT_UNSEEN_CSV} nor nested unseen directory exists: {unseen_root}"
                )
            unseen_csvs = sorted(unseen_root.glob("*/*/test.csv"))
            if not unseen_csvs:
                raise FileNotFoundError(f"No unseen test.csv files beneath {unseen_root}")
            if pd is None:
                raise RuntimeError(
                    "pandas is required to merge unseen endpoint splits automatically. "
                    "Install pandas or pass a consolidated unseen CSV via --input_csv."
                )
            dfs = []
            for p in unseen_csvs:
                df = pd.read_csv(p)
                if not df.empty:
                    dfs.append(df)
            if not dfs:
                raise ValueError(f"Every unseen_endpoint_test CSV was empty beneath {unseen_root}")
            merged_unseen = pd.concat(dfs, ignore_index=True)
            merged_unseen_path = _DEFAULT_UNSEEN_SPLIT_DIR / "merged_unseen_test.csv"
            merged_unseen.to_csv(merged_unseen_path, index=False)
            data_path = merged_unseen_path
            print(
                f"[unseen] merged {len(unseen_csvs)} unseen test shards "
                f"({len(merged_unseen)} rows) -> {merged_unseen_path}"
            )

    DATA_TASK1 = data_path   # toxic_fragment_identification
    DATA_TASK2 = data_path   # nontoxic_fragment_generation
    DATA_TASK3 = data_path   # nontoxic_smiles_generation

    OUT_DIR_TASK1 = split_dir / "task1_toxic_fragment_identification" / repr_dir
    OUT_DIR_TASK2 = split_dir / "task2_nontoxic_fragment_generation" / repr_dir
    OUT_DIR_TASK3 = split_dir / "task3_nontoxic_smiles_generation" / repr_dir
    OUT_DIR_TASK3_NONToxic_SAFE_GENERATION = split_dir / "task3_nontoxic_safe_generation" / repr_dir
    OUT_DIR_TASK3_STEPWISE_COT_SAFE = split_dir / "task3_stepwise_cot_nontoxic_safe_generation" / repr_dir


def main():
    ap = argparse.ArgumentParser(description="MolDeTox QA jsonl (base or ICL-K).")
    ap.add_argument(
        "--task",
        choices=[
            "task1",
            "task2",
            "task3",
            "task3_nontoxic_safe_generation",
            "task3_stepwise_cot_safe_generation",
            "all",
        ],
        default="all",
        help=(
            "task1 · task2 · task3 · task3_nontoxic_safe_generation · "
            "task3_stepwise_cot_safe_generation · all (runs every task above)"
        ),
    )
    ap.add_argument(
        "--variant",
        choices=["base", "icl-k", "all"],
        default="base",
        help=(
            "base: QA without ICL. icl-k: few-shot (--icl-k). "
            "all: sequential base followed by icl-k. Default base."
        ),
    )
    ap.add_argument(
        "--icl-k",
        type=int,
        default=4,
        choices=[1, 2, 4],
        metavar="K",
        help="ICL-K slot selector (supports 1,2,4 only). Applies when variant is icl-k or all (default K=4).",
    )
    ap.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help=(
            "Split to materialize QA for ('train','test'); use 'all' to loop train then test "
            "(default 'test')."
        ),
    )
    ap.add_argument(
        "--unseen",
        action="store_true",
        help=(
            "Emit QA under unseen_test/. Prefers single unseen_endpoint_test.csv when present; "
            "otherwise merges legacy nested test.csv shards (requires pandas)."
        ),
    )
    ap.add_argument(
        "--input_csv",
        type=Path,
        default=None,
        help=(
            "Override SAFE pair CSV. Defaults derive from scaffold split paths or unseen bundles "
            "(see unseen flag)."
        ),
    )
    ap.add_argument(
        "--train_csv",
        type=Path,
        default=None,
        help="Train CSV override when --split train or all.",
    )
    ap.add_argument(
        "--test_csv",
        type=Path,
        default=None,
        help="Test CSV override when --split test or all.",
    )
    ap.add_argument(
        "--qa_set",
        type=str,
        default=None,
        help="Optional QA bundle name persisted under qa_sets/<name>/ relative to MolDeTox.",
    )
    ap.add_argument(
        "--no_desc",
        action="store_true",
        help="Strip bundled endpoint narratives from prompts.",
    )
    ap.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Shuffle RNG seed consulted when --shuffle is provided (default 42).",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle records before emitting JSONL (uses --shuffle_seed).",
    )
    ap.add_argument(
        "--no_shuffle",
        action="store_true",
        default=True,
        help="(Default) Preserve CSV order unless --shuffle is passed.",
    )
    ap.add_argument(
        "--sim-dir",
        type=Path,
        default=None,
        help=(
            "Directory with toxic-vs-toxic similarity artifacts "
            "(toxic_safe_decoded_smiles_matrix.npy, toxic_safe_decoded_smiles_list.json)."
        ),
    )
    ap.add_argument(
        "--prebuild-toxic-sim-matrix",
        action="store_true",
        help=(
            "Automatically build toxic-toxic similarity caches from DATA_TASK1 before ICL emits."
        ),
    )
    ap.add_argument(
        "--icl-from-index-json",
        action="store_true",
        help=(
            "Build ICL from icl_train_topk_indices.json (top indices + merged train) instead "
            "of dense toxic similarity matrices."
        ),
    )
    ap.add_argument(
        "--icl-json",
        type=Path,
        default=None,
        help=f"Explicit path for icl index JSON (defaults to {_DEFAULT_ICL_INDEX_JSON}).",
    )
    ap.add_argument(
        "--icl-job-name",
        type=str,
        default=None,
        help=(
            'job "name" field inside icl_train_topk_indices.json '
            '(optional when unique paths disambiguate the job payload).'
        ),
    )
    args = ap.parse_args()

    global INCLUDE_ENDPOINT_DESCRIPTION
    INCLUDE_ENDPOINT_DESCRIPTION = not bool(args.no_desc)

    # Disambiguate qa_set roots whenever descriptions are suppressed
    if args.no_desc:
        if getattr(args, "qa_set_root", None):
            _p = Path(str(args.qa_set_root)).expanduser().resolve()
            if not _p.name.endswith("_no_desc"):
                args.qa_set_root = str(_p.with_name(_p.name + "_no_desc"))
        elif getattr(args, "qa_set", None):
            _name = str(args.qa_set).strip()
            if _name and not _name.endswith("_no_desc"):
                args.qa_set = _name + "_no_desc"

    global BUILD_QA_SHUFFLE_SEED
    BUILD_QA_SHUFFLE_SEED = args.shuffle_seed if args.shuffle else None

    global QA_OUT_ROOT
    if args.qa_set:
        safe = (args.qa_set or "").strip().strip("/").replace("..", "").replace("\\", "_")
        safe = safe.replace("/", "_")
        if not safe:
            raise ValueError("--qa_set cannot be empty.")
        QA_OUT_ROOT = (_QA_DIR / "qa_sets" / safe).resolve()
        QA_OUT_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"[qa_set] output_root={QA_OUT_ROOT}")

    icl_tag = _icl_variant_tag_for_k(args.icl_k)
    if args.variant == "base":
        variants_to_run = ["base"]
    elif args.variant == "icl-k":
        variants_to_run = [icl_tag]
    else:
        variants_to_run = ["base", icl_tag]

    def _input_csv_for_split(split: str) -> Path | None:
        if args.unseen:
            return args.input_csv
        if split == "train":
            return args.train_csv or args.input_csv
        if split == "test":
            return args.test_csv or args.input_csv
        return args.input_csv

    splits_to_run = ["train", "test"] if args.split == "all" else [args.split]

    # Configure globals before similarity prebuild consumes DATA_TASK1
    _configure_paths(
        split=splits_to_run[0],
        input_csv=_input_csv_for_split(splits_to_run[0]),
        molecule_repr="both_repre",
        unseen=args.unseen,
    )
    sim_dir_for_icl = str(args.sim_dir) if args.sim_dir else None
    if (
        args.prebuild_toxic_sim_matrix
        and any(v != "base" for v in variants_to_run)
        and not args.icl_from_index_json
    ):
        from ICL_template import DEFAULT_SIM_OUT_DIR, build_toxic_toxic_sim_matrix

        out_sim = args.sim_dir if args.sim_dir is not None else DEFAULT_SIM_OUT_DIR
        print(f"[prebuild-toxic-sim-matrix] pairs_csv={DATA_TASK1} -> {out_sim}")
        build_toxic_toxic_sim_matrix(pairs_csv=DATA_TASK1, out_dir=out_sim)

    for split_name in splits_to_run:
        if args.unseen and split_name != "test":
            continue
        if args.split == "all":
            print(f"[split={split_name}]")

        _configure_paths(
            split=split_name,
            input_csv=_input_csv_for_split(split_name),
            molecule_repr="both_repre",
            unseen=args.unseen,
        )

        for v in variants_to_run:
            # Cannot mine ICL neighbors without paired train shards—fall back to base QA.
            if (
                v != "base"
                and split_name == "test"
                and (args.input_csv is not None)
                and (args.train_csv is None)
                and (not args.unseen)
            ):
                print(
                    f"[WARN] variant={v} requested but --train_csv not provided while --input_csv is set. "
                    "Falling back to base QA (no ICL) for this split."
                )
                v_eff = "base"
            else:
                v_eff = v

            if v_eff == "base":
                if args.task in ("task1", "all"):
                    build_task1()
                if args.task in ("task2", "all"):
                    build_task2()
                if args.task in ("task3", "all"):
                    build_task3()
                if args.task in ("task3_nontoxic_safe_generation", "all"):
                    build_task3_nontoxic_safe_generation()
                if args.task in ("task3_stepwise_cot_safe_generation", "all"):
                    build_task3_stepwise_cot_safe_generation()
            else:
                icl_json_path = (
                    args.icl_json if args.icl_json is not None else _DEFAULT_ICL_INDEX_JSON
                )
                if args.icl_from_index_json:
                    if (
                        build_task1_toxic_fragment_identification_icl_from_index_json is None
                        or build_task2_nontoxic_fragment_generation_icl_from_index_json is None
                        or build_task3_nontoxic_smiles_generation_icl_from_index_json is None
                        or build_task3_nontoxic_safe_generation_icl_from_index_json is None
                        or build_task3_stepwise_cot_nontoxic_safe_generation_icl_from_index_json
                        is None
                    ):
                        msg = "ICL_template import failed; cannot build ICL QA (index JSON)."
                        if _icl_import_error is not None:
                            raise RuntimeError(msg) from _icl_import_error
                        raise RuntimeError(msg)
                    idx_kw: dict = {
                        "test_csv": DATA_TASK1,
                        "icl_json": icl_json_path,
                        "job_name": args.icl_job_name,
                        "variants": [v_eff],
                        "molecule_repr": CURRENT_MOLECULE_REPR,
                    }
                    if args.task in ("task1", "all"):
                        build_task1_toxic_fragment_identification_icl_from_index_json(
                            **idx_kw,
                            out_dir=OUT_DIR_TASK1,
                        )
                    if args.task in ("task2", "all"):
                        build_task2_nontoxic_fragment_generation_icl_from_index_json(
                            **idx_kw,
                            out_dir=OUT_DIR_TASK2,
                        )
                    if args.task in ("task3", "all"):
                        build_task3_nontoxic_smiles_generation_icl_from_index_json(
                            **idx_kw,
                            out_dir=OUT_DIR_TASK3,
                        )
                    if args.task in ("task3_nontoxic_safe_generation", "all"):
                        build_task3_nontoxic_safe_generation_icl_from_index_json(
                            **idx_kw,
                            out_dir=OUT_DIR_TASK3_NONToxic_SAFE_GENERATION,
                        )
                    if args.task in ("task3_stepwise_cot_safe_generation", "all"):
                        build_task3_stepwise_cot_nontoxic_safe_generation_icl_from_index_json(
                            **idx_kw,
                            out_dir=OUT_DIR_TASK3_STEPWISE_COT_SAFE,
                        )
                else:
                    if (
                        build_task1_toxic_fragment_identification_icl is None
                        or build_task2_nontoxic_fragment_generation_icl is None
                        or build_task3_nontoxic_smiles_generation_icl is None
                        or build_task3_nontoxic_safe_generation_icl is None
                        or build_task3_stepwise_cot_nontoxic_safe_generation_icl is None
                    ):
                        msg = "ICL_template import failed; cannot build ICL QA."
                        if _icl_import_error is not None:
                            raise RuntimeError(msg) from _icl_import_error
                        raise RuntimeError(msg)
                    if args.task in ("task1", "all"):
                        build_task1_toxic_fragment_identification_icl(
                            variants=[v_eff],
                            pairs_csv=DATA_TASK1,
                            sim_dir=sim_dir_for_icl,
                            out_dir=OUT_DIR_TASK1,
                            molecule_repr=CURRENT_MOLECULE_REPR,
                        )
                    if args.task in ("task2", "all"):
                        build_task2_nontoxic_fragment_generation_icl(
                            variants=[v_eff],
                            pairs_csv=DATA_TASK2,
                            sim_dir=sim_dir_for_icl,
                            out_dir=OUT_DIR_TASK2,
                            molecule_repr=CURRENT_MOLECULE_REPR,
                        )
                    if args.task in ("task3", "all"):
                        build_task3_nontoxic_smiles_generation_icl(
                            variants=[v_eff],
                            pairs_csv=DATA_TASK3,
                            sim_dir=sim_dir_for_icl,
                            out_dir=OUT_DIR_TASK3,
                            molecule_repr=CURRENT_MOLECULE_REPR,
                        )
                    if args.task in ("task3_nontoxic_safe_generation", "all"):
                        build_task3_nontoxic_safe_generation_icl(
                            variants=[v_eff],
                            pairs_csv=DATA_TASK3,
                            sim_dir=sim_dir_for_icl,
                            out_dir=OUT_DIR_TASK3_NONToxic_SAFE_GENERATION,
                            molecule_repr=CURRENT_MOLECULE_REPR,
                        )
                    if args.task in ("task3_stepwise_cot_safe_generation", "all"):
                        build_task3_stepwise_cot_nontoxic_safe_generation_icl(
                            variants=[v_eff],
                            pairs_csv=DATA_TASK3,
                            sim_dir=sim_dir_for_icl,
                            out_dir=OUT_DIR_TASK3_STEPWISE_COT_SAFE,
                            molecule_repr=CURRENT_MOLECULE_REPR,
                        )


if __name__ == "__main__":
    main()
