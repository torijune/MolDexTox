"""
Evaluation metrics for SAFE-centric MolDeTox QA splits.

Answers may be dict payloads such as ``{"answer": "..."}`` or literal strings paired with predicted strings.

Task3-style metrics canonize SMILES through RDKit (``Chem.MolToSmiles(..., canonical=True)``) prior to similarity.
Morgan fingerprints always use ``MORGAN_FP_NBITS`` bits (defaults to 1024).

Property-related scores reuse the inlined ``eval_prs`` helper (six QED-aligned descriptors). Non-finite
QED-derived values force zero-valued contributions while still occupying the aggregates.

Merged split CSV bookkeeping:
  • ``configure_eval_data_paths`` registers ``merged_split_csv`` paths for toxicity-aware lookups.
  • When unset, PRS-aligned signals fall back to zero.

Supported workflows: ``task1``, ``task2``, ``task3``, ``task3_nontoxic_safe_generation``, and
``task3_stepwise_cot_safe_generation`` for paper-aligned reproduction.

Emitted metric dictionaries align with ``TASK_METRIC_KEYS`` / ``metric_keys_for(task, step)``.

Summaries per task/step:
• **task1**: ``fragment_EM`` (+ ``fragment_F1`` in multi fragments). ``*_Acc`` adds mean×100 in summaries.
• **task2**: ``fragment_EM``, ``fragment_Levenshtein`` (+ ``fragment_F1`` for ``multi_step`` fragments).
• **task3 cohort** (SAFE or SMILES, including chain-of-thought variants): canonical ``exact_match``,
  ``bleu``, ``levenshtein``, fingerprint similarities, ``validity``, ``prs_property_score``.
"""
from __future__ import annotations

import csv
from collections import Counter
import math
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED

try:
    from rdkit.Chem.Fingerprints import FingerprintMols
except ImportError:
    FingerprintMols = None
try:
    from rdkit.Chem import MACCSkeys
except ImportError:
    MACCSkeys = None
# ---------------------------------------------------------------------------
# PRS: QED 6 descriptors (MW, ALOGP, HBA, HBD, PSA, ROTB), no extra scaling
# (inlined from eval_property.py)
# ---------------------------------------------------------------------------
PROPS_6 = ("MW", "ALOGP", "HBA", "HBD", "PSA", "ROTB")
ScoreMode = Literal["exponential", "linear", "reciprocal"]


def qed_six_from_mol(mol: Any) -> float:
    if mol is None:
        return float("nan")
    p = QED.properties(mol)
    w_mean = QED.WEIGHT_MEAN
    t = 0.0
    sw = 0.0
    for name in PROPS_6:
        pi = getattr(p, name)
        di = QED.ads(pi, QED.adsParameters[name])
        wi = getattr(w_mean, name)
        t += wi * math.log(di)
        sw += wi
    return math.exp(t / sw)


def qed_six_from_smiles(smi: str) -> float:
    return qed_six_from_mol(Chem.MolFromSmiles(smi))


def _score_from_x_abs_diff(x: float, mode: ScoreMode) -> float:
    if mode == "exponential":
        return math.exp(-x)
    if mode == "linear":
        return max(0.0, 1.0 - x)
    if mode == "reciprocal":
        return 1.0 / (1.0 + x)
    raise ValueError(f"unknown mode: {mode!r}")


def eval_prs(
    toxic_smiles: str,
    nontoxic_smiles: str,
    *,
    mode: ScoreMode = "exponential",
) -> dict[str, Any]:
    qt = qed_six_from_smiles(str(toxic_smiles))
    qn = qed_six_from_smiles(str(nontoxic_smiles))
    if not (math.isfinite(qt) and math.isfinite(qn)):
        return {
            "score": 0.0,
            "x_abs_diff": 0.0,
            "qed6_toxic": 0.0,
            "qed6_nontoxic": 0.0,
            "mode": mode,
        }
    x = abs(qt - qn)
    score = _score_from_x_abs_diff(x, mode)
    return {
        "score": float(score),
        "x_abs_diff": float(x),
        "qed6_toxic": float(qt),
        "qed6_nontoxic": float(qn),
        "mode": mode,
    }


def compute_qed(smiles: str) -> float:
    """RDKit drug-likeness QED (0–1). Invalid SMILES raises ValueError."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return float(QED.qed(mol))


# SAFE -> SMILES: optional ``safe`` package
SAFE_DECODE_IMPORT_ERROR: Optional[BaseException] = None
try:
    from safe.converter import decode as safe_decode
except Exception:
    try:
        from safe.safe.converter import decode as safe_decode  # type: ignore
    except Exception as _safe_dec_err:  # pragma: no cover - optional dependency
        safe_decode = None
        SAFE_DECODE_IMPORT_ERROR = _safe_dec_err

_MERGED_RAW_CSV: Optional[Path] = None

_MERGED_RAW_ROWS: Optional[list[dict[str, Any]]] = None

_MERGED_TRIED: bool = False


def configure_eval_data_paths(
    *,
    merged_split_csv: Optional[str | Path] = None,
    pairs_csv: Optional[str | Path] = None,
) -> None:
    """Point lookups at ``merged_split_csv`` and purge cached merges; legacy ``pairs_csv`` is ignored."""

    def _p(x: Optional[str | Path]) -> Optional[Path]:
        if x is None or x == "":
            return None
        return Path(x).expanduser().resolve()

    global _MERGED_RAW_CSV
    global _MERGED_RAW_ROWS
    global _MERGED_TRIED

    del pairs_csv  # legacy kwarg discarded for backwards compatibility
    _MERGED_RAW_CSV = _p(merged_split_csv)
    _MERGED_RAW_ROWS = None
    _MERGED_TRIED = False


def configure_eval_merged_split(csv_path: str | Path) -> None:
    """Back-compat alias naming merged split CSV bundles (argument is CSV path string)."""
    configure_eval_data_paths(merged_split_csv=csv_path)


def _read_csv_rows(path: Path) -> Optional[list[dict[str, Any]]]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None

# Morgan fingerprint bit width (task3 ``morgan_fts`` telemetry)
MORGAN_FP_NBITS = 1024

# Enumerate sanctioned metric dict keys per (task × step bucket).
TASK3_METRIC_KEYS: list[str] = [
    "exact_match",
    "bleu",
    "levenshtein",
    "rdk_fts",
    "maccs_fts",
    "morgan_fts",
    "validity",
    "prs_property_score",
]

TASK_METRIC_KEYS: dict[str, dict[str, list[str]]] = {
    "task1": {
        "single_step": ["fragment_EM"],
        "multi_step": ["fragment_EM", "fragment_F1"],
    },
    "task2": {
        "single_step": ["fragment_EM", "fragment_Levenshtein"],
        "multi_step": ["fragment_EM", "fragment_Levenshtein", "fragment_F1"],
    },
    "task3": {
        "single_step": list(TASK3_METRIC_KEYS),
        "multi_step": list(TASK3_METRIC_KEYS),
    },
    "task3_nontoxic_safe_generation": {
        "single_step": list(TASK3_METRIC_KEYS),
        "multi_step": list(TASK3_METRIC_KEYS),
    },
    "task3_stepwise_cot_safe_generation": {
        "single_step": list(TASK3_METRIC_KEYS),
        "multi_step": list(TASK3_METRIC_KEYS),
    },
}


def augment_metrics_mean_with_em_accuracy(
    metric_means: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    """
    Append ``{base}_Acc = mean * 100`` companions for exact-match style metrics when ``metrics_mean`` stores 0-1 means.

    Targets keys named ``EM``, ending with ``_EM``, or ``exact_match`` for SMILES equality.
    """
    out: Dict[str, Optional[float]] = dict(metric_means)
    for k, v in metric_means.items():
        if not (k == "EM" or k.endswith("_EM") or k == "exact_match"):
            continue
        acc_key = f"{k}_Acc"
        if v is None:
            out[acc_key] = None
        else:
            out[acc_key] = float(v) * 100.0
    return out


StepNorm = Literal["single_step", "multi_step"]

_TASK3_LIKE: frozenset[str] = frozenset(
    {
        "task3",
        "task3_nontoxic_safe_generation",
        "task3_stepwise_cot_safe_generation",
    }
)


def metric_keys_for(task: str, step: StepNorm | str) -> list[str]:
    """Ordered metric keys expected for ``task`` + ``step`` aggregation."""
    s = str(step).strip()
    if s not in ("single_step", "multi_step"):
        raise ValueError(f"step must be single_step or multi_step, got {step!r}")
    tmap = TASK_METRIC_KEYS.get(task)
    if tmap is None:
        raise KeyError(f"unknown task: {task!r}")
    if task in _TASK3_LIKE:
        return list(tmap["single_step"])
    return list(tmap[s])


def merge_finite_metrics_into_sums(
    metric_sums: Dict[str, float],
    metric_counts: Dict[str, int],
    metrics: Dict[str, Any],
) -> None:
    """Accumulate finite numeric metrics; skip NaN/inf entries (e.g., unstable PRS rows)."""
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            fv = float(v)
            if math.isfinite(fv):
                metric_sums[k] = metric_sums.get(k, 0.0) + fv
                metric_counts[k] = metric_counts.get(k, 0) + 1


def mean_metrics_from_sums(
    metric_sums: Dict[str, float],
    metric_counts: Dict[str, int],
    keys: list[str],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        c = metric_counts.get(k, 0)
        if c > 0 and k in metric_sums:
            out[k] = metric_sums[k] / c
        else:
            out[k] = None
    return out


def _property_prs_tuple_for_task3_smiles(
    row_id: Optional[int],
    pred_smiles: str,
) -> Tuple[float, float, float, float]:
    """
    Run ``eval_prs`` between merged toxic SMILES (``source_index`` row) and predicted nontoxic SMILES.

    Returns zero quadruple when rows are missing, SMILES empty, or exceptions fire so averages stay well-defined.
    """
    zero4 = (0.0, 0.0, 0.0, 0.0)
    row = _get_merged_test_row_by_id(row_id)
    toxic = str(row.get("toxic_smiles", "") or "").strip() if row is not None else ""
    pred_s = (pred_smiles or "").strip()
    if not toxic or not pred_s:
        return zero4
    try:
        r = eval_prs(toxic, pred_s, mode="exponential")
    except Exception:
        return zero4

    def _prs_float(v: Any) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        return x if math.isfinite(x) else 0.0

    return (
        _prs_float(r.get("score", 0.0)),
        _prs_float(r.get("x_abs_diff", 0.0)),
        _prs_float(r.get("qed6_toxic", 0.0)),
        _prs_float(r.get("qed6_nontoxic", 0.0)),
    )


def _extract_answer(ans: Any) -> str:
    """Pull string answers from dict-or-string gold/pred payloads."""
    if ans is None:
        return ""
    if isinstance(ans, dict):
        return str(ans.get("answer", "")).strip()
    return str(ans).strip()


def _tokenize_safe_fragments(s: str) -> list[str]:
    """
    Split SAFE strings on ``.`` after stripping whitespace; empty tokens drop out.

    Single-step labels usually carry one token; multi-step examples carry two or more.
    """
    s = (s or "").strip()
    if not s:
        return []
    # Remove inline spaces then split on separator dots
    return [tok for tok in s.replace(" ", "").split(".") if tok]


def _fragments_multiset_equal(gold: str, pred: str) -> bool:
    """
    Compare dot-separated SAFE fragments as multisets (order ignored, duplicates respected).

    Example: ``Frag1.Frag2.Frag3`` matches ``Frag2.Frag3.Frag1``.
    """
    g = _tokenize_safe_fragments(gold)
    p = _tokenize_safe_fragments(pred)
    return sorted(g) == sorted(p)


def _fragment_set_precision_recall_f1(gold: str, pred: str) -> Tuple[float, float, float]:
    """
    Set-based precision/recall/F1 on dot-separated SAFE fragments (shared by single/multi steps).

    Steps:
    - Tokenize via ``_tokenize_safe_fragments`` then convert to sets.
    - ``TP = |gold ∩ pred|``
    - ``Precision = TP / |pred|`` (0 if pred empty)
    - ``Recall = TP / |gold|`` (gold empty handled as edge case)
    - ``F1 = harmonic mean`` with zero guard
    - Both empty → perfect scores (1,1,1)
    """
    gold_toks = _tokenize_safe_fragments(gold)
    pred_toks = _tokenize_safe_fragments(pred)
    gold_set = set(gold_toks)
    pred_set = set(pred_toks)

    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gold_set:
        return 0.0 if pred_set else 1.0, 1.0, (0.0 if pred_set else 1.0)
    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _tokenize_char_ngrams(s: str, n: int = 4) -> list[str]:
    """Character n-gram iterator with sliding windows (BLEU-style overlap)."""
    s = (s or "").strip().replace(" ", "")
    if not s or n < 1:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _bleu1_safe_fragments(gold: str, pred: str, use_char_ngrams: bool = True, ngram_n: int = 1) -> float:
    """
    BLEU-1 style precision for SAFE strings.

    - ``use_char_ngrams=True`` (default): operate on character n-grams (``ngram_n=1`` means characters).
    - ``False``: treat dot-separated fragments as tokens.
    """
    if use_char_ngrams:
        gold_tokens = _tokenize_char_ngrams(gold, ngram_n)
        pred_tokens = _tokenize_char_ngrams(pred, ngram_n)
    else:
        gold_tokens = _tokenize_safe_fragments(gold)
        pred_tokens = _tokenize_safe_fragments(pred)

    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)

    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, gold_counts.get(t, 0))

    precision = overlap / max(len(pred_tokens), 1)
    return float(precision)


def _safe_to_smiles_validity(safe_str: str) -> float:
    """
    Decode SAFE to SMILES (when ``safe_decode`` exists) and score RDKit parseability.

    Returns ``0.0`` if helpers are missing or Mol construction fails; ``1.0`` on success.
    """
    safe_str = (safe_str or "").strip()
    if not safe_str or safe_decode is None:
        return 0.0
    try:
        # SAFE decode -> SMILES
        decoded_smiles = safe_decode(safe_str)
    except Exception:
        return 0.0

    if not decoded_smiles:
        return 0.0

    try:
        mol = Chem.MolFromSmiles(str(decoded_smiles))
    except Exception:
        return 0.0

    return 1.0 if mol is not None else 0.0


def _decode_safe_to_smiles(safe_str: str) -> Optional[str]:
    """Decode SAFE to SMILES or return ``None``."""
    safe_str = (safe_str or "").strip()
    if not safe_str or safe_decode is None:
        return None
    try:
        decoded_smiles = safe_decode(safe_str)
    except Exception:
        return None
    decoded_smiles = (decoded_smiles or "").strip()
    return decoded_smiles or None


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    smiles = (smiles or "").strip()
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _morgan_tanimoto(
    smiles1: str,
    smiles2: str,
    radius: int = 2,
    nbits: int = MORGAN_FP_NBITS,
) -> Optional[float]:
    """
    Morgan fingerprint Tanimoto between SMILES strings (``None`` on failure).

    Callers should canonicalize SMILES first because downstream comparisons assume normalized forms.
    """
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    try:
        # Prefer MorganGenerator to avoid legacy GetMorganFingerprintAsBitVect warnings.
        try:
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

            gen = GetMorganGenerator(radius=radius, fpSize=nbits)
            fp1 = gen.GetFingerprint(mol1)
            fp2 = gen.GetFingerprint(mol2)
        except Exception:
            # Older RDKit fallback
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nbits)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nbits)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return None


def _rdkit_tanimoto(smiles1: str, smiles2: str) -> float:
    """RDKit topological fingerprint Tanimoto (0.0 if unavailable)."""
    if FingerprintMols is None:
        return 0.0
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _maccs_tanimoto(smiles1: str, smiles2: str) -> float:
    """MACCS key Tanimoto similarity (0.0 if unavailable)."""
    if MACCSkeys is None:
        return 0.0
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _load_merged_raw_rows() -> Optional[list[dict[str, Any]]]:
    """Lazy-load merged CSV rows for ``source_index`` joins."""
    global _MERGED_RAW_ROWS, _MERGED_TRIED
    if not _MERGED_TRIED:
        _MERGED_TRIED = True
        if _MERGED_RAW_CSV is not None and _MERGED_RAW_CSV.is_file():
            _MERGED_RAW_ROWS = _read_csv_rows(_MERGED_RAW_CSV)
        else:
            _MERGED_RAW_ROWS = None
    return _MERGED_RAW_ROWS


def _get_merged_test_row_by_id(row_id: Optional[int]) -> Optional[dict[str, Any]]:
    """Fetch merged CSV row by QA ``source_index``."""
    if row_id is None:
        return None
    rows = _load_merged_raw_rows()
    if rows is None:
        return None
    if row_id < 0 or row_id >= len(rows):
        return None
    return rows[int(row_id)]


def row_id_for_merged_lookup(row: Optional[Dict[str, Any]]) -> Optional[int]:
    """
    Resolve merged-row index from prediction/QA dicts.

    Prefer ``source_index``; fall back to ``id`` when JSON nulls break naive ``dict.get`` chains.
    """
    if not row:
        return None
    si = row.get("source_index")
    if si is not None:
        try:
            return int(si)
        except (TypeError, ValueError):
            return None
    i = row.get("id")
    if i is None:
        return None
    try:
        return int(i)
    except (TypeError, ValueError):
        return None


def _levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance."""
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = cur
    return dp[lb]


def _fragment_levenshtein_mean_over_pred(gold: str, pred: str) -> float:
    """
    Fragment-aware Levenshtein for Task2 (lower is better, mirroring Task3 string distance).

    For each predicted fragment token pick the minimum distance to any gold fragment and average those minima.
    """
    gold_toks = _tokenize_safe_fragments(gold)
    pred_toks = _tokenize_safe_fragments(pred)
    if not gold_toks or not pred_toks:
        return float(_levenshtein((gold or "").replace(" ", ""), (pred or "").replace(" ", "")))
    per_pred: list[float] = []
    for p in pred_toks:
        best: float | None = None
        for g in gold_toks:
            d = float(_levenshtein(p, g))
            if best is None or d < best:
                best = d
                if best <= 0.0:
                    break
        per_pred.append(float(best or 0.0))
    if not per_pred:
        return float(_levenshtein((gold or "").replace(" ", ""), (pred or "").replace(" ", "")))
    return float(sum(per_pred) / len(per_pred))


def task1_toxic_fragment_identification_eval(
    gold_answer: Any,
    llm_answer: Any,
    *,
    step: StepNorm,
) -> Dict[str, float]:
    """
    Task 1 toxic fragment tagging.

    - ``single_step``: ``fragment_EM`` only.
    - ``multi_step``: ``fragment_EM`` plus multiset ``fragment_F1``.
    """
    gold = _extract_answer(gold_answer)
    pred = _extract_answer(llm_answer)

    fragment_EM = 1.0 if gold and _fragments_multiset_equal(gold, pred) else 0.0
    out: Dict[str, float] = {"fragment_EM": fragment_EM}
    if step == "multi_step":
        _, _, _, _, fragment_F1 = _fragment_set_precision_recall_f1(gold, pred)
        out["fragment_F1"] = float(fragment_F1)
    return out

def task2_nontoxic_fragment_generation_eval(
    gold_answer: Any,
    llm_answer: Any,
    *,
    step: StepNorm,
    row_id: Optional[int] = None,
    context_row: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Task 2 nontoxic SAFE fragment prediction.

    - ``single_step``: ``fragment_EM`` and ``fragment_Levenshtein``.
    - ``multi_step``: same keys plus ``fragment_F1``.

    ``row_id`` / ``context_row`` stay for signature compatibility only.
    """
    del row_id, context_row
    gold = _extract_answer(gold_answer)
    pred = _extract_answer(llm_answer)

    fragment_EM = 1.0 if gold and _fragments_multiset_equal(gold, pred) else 0.0
    fragment_Levenshtein = _fragment_levenshtein_mean_over_pred(gold, pred)
    out: Dict[str, float] = {
        "fragment_EM": fragment_EM,
        "fragment_Levenshtein": float(fragment_Levenshtein),
    }
    if step == "multi_step":
        _, _, _, _, fragment_F1 = _fragment_set_precision_recall_f1(gold, pred)
        out["fragment_F1"] = float(fragment_F1)
    return out

def task3_nontoxic_smiles_generation_eval(
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    Task 3 SMILES generation (single-string answers).

    Keys mirror ``TASK3_METRIC_KEYS``. Molecule-level scores (``exact_match``, ``bleu``,
    ``levenshtein``, fingerprints, ``prs_property_score``) use RDKit-canonical SMILES when
    both molecules parse; ``bleu``/``levenshtein`` are 0 if either side fails. ``validity``
    still reflects whether the raw prediction string parses.
    """
    gold_s = (_extract_answer(gold_answer) or "").strip()
    pred_s = (_extract_answer(llm_answer) or "").strip()

    validity = 1.0 if pred_s and _mol_from_smiles(pred_s) is not None else 0.0

    can_gold: Optional[str] = None
    can_pred: Optional[str] = None
    if gold_s:
        mol_g = _mol_from_smiles(gold_s)
        if mol_g is not None:
            can_gold = Chem.MolToSmiles(mol_g, canonical=True)
    if pred_s:
        mol_p = _mol_from_smiles(pred_s)
        if mol_p is not None:
            can_pred = Chem.MolToSmiles(mol_p, canonical=True)

    exact_match = 1.0 if (can_gold and can_pred and can_gold == can_pred) else 0.0
    if can_gold is not None and can_pred is not None:
        bleu = _bleu1_safe_fragments(can_gold, can_pred, use_char_ngrams=True, ngram_n=1)
        levenshtein = float(_levenshtein(can_gold, can_pred))
    else:
        bleu = 0.0
        levenshtein = 0.0

    rdk_fts = 0.0
    maccs_fts = 0.0
    morgan_fts = 0.0
    if can_gold and can_pred:
        rdk_fts = _rdkit_tanimoto(can_gold, can_pred)
        maccs_fts = _maccs_tanimoto(can_gold, can_pred)
        m = _morgan_tanimoto(can_gold, can_pred)
        morgan_fts = m if m is not None else 0.0

    pred_for_prs = ((can_pred or pred_s) or "").strip()
    prs_s, _, _, _ = _property_prs_tuple_for_task3_smiles(row_id, pred_for_prs)

    return {
        "exact_match": float(exact_match),
        "bleu": float(bleu),
        "levenshtein": float(levenshtein),
        "rdk_fts": float(rdk_fts),
        "maccs_fts": float(maccs_fts),
        "morgan_fts": float(morgan_fts),
        "validity": float(validity),
        "prs_property_score": float(prs_s),
    }


def task3_nontoxic_safe_generation_eval(
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    Task 3 SAFE generation (dot-concatenated SAFE strings).

    After decoding to canonical SMILES the eight Task3 keys match ``task3_nontoxic_smiles_generation_eval``.
    """
    gold_safe = (_extract_answer(gold_answer) or "").strip()
    pred_safe = (_extract_answer(llm_answer) or "").strip()

    gold_decoded = _decode_safe_to_smiles(gold_safe)
    pred_decoded = _decode_safe_to_smiles(pred_safe)

    can_gold: Optional[str] = None
    can_pred: Optional[str] = None

    mol_gold = _mol_from_smiles(gold_decoded or "") if gold_decoded else None
    mol_pred = _mol_from_smiles(pred_decoded or "") if pred_decoded else None
    if mol_gold is not None:
        can_gold = Chem.MolToSmiles(mol_gold, canonical=True)
    if mol_pred is not None:
        can_pred = Chem.MolToSmiles(mol_pred, canonical=True)

    decode_ok = pred_decoded is not None
    mol_ok = mol_pred is not None
    validity = 1.0 if (decode_ok and mol_ok) else 0.0

    exact_match = (
        1.0
        if (can_gold is not None and can_pred is not None and can_gold == can_pred)
        else 0.0
    )

    if can_gold is not None and can_pred is not None:
        bleu = _bleu1_safe_fragments(can_gold, can_pred, use_char_ngrams=True, ngram_n=1)
        levenshtein = float(_levenshtein(can_gold, can_pred))
    else:
        bleu = 0.0
        levenshtein = 0.0

    rdk_fts = 0.0
    maccs_fts = 0.0
    morgan_fts = 0.0
    if can_gold is not None and can_pred is not None:
        rdk_fts = _rdkit_tanimoto(can_gold, can_pred)
        maccs_fts = _maccs_tanimoto(can_gold, can_pred)
        morgan = _morgan_tanimoto(can_gold, can_pred)
        morgan_fts = morgan if morgan is not None else 0.0

    pred_for_prs = (can_pred or pred_decoded or "").strip()
    prs_s, _, _, _ = _property_prs_tuple_for_task3_smiles(row_id, pred_for_prs)

    return {
        "exact_match": float(exact_match),
        "bleu": float(bleu),
        "levenshtein": float(levenshtein),
        "rdk_fts": float(rdk_fts),
        "maccs_fts": float(maccs_fts),
        "morgan_fts": float(morgan_fts),
        "validity": float(validity),
        "prs_property_score": float(prs_s),
    }


def task3_stepwise_cot_nontoxic_safe_generation_eval(
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    Stepwise SAFE CoT evaluated solely on final JSON ``answer`` using ``task3_nontoxic_safe_generation_eval``.
    """
    return task3_nontoxic_safe_generation_eval(gold_answer, llm_answer, row_id=row_id)
