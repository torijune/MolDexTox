"""
Toxicity-cliff pairing pipeline mirroring the paper appendix.

1. **Step 1 screening** keeps a pair whenever **any** criterion fires:
   - Bemis–Murcko scaffold Morgan (ECFP-like) Tanimoto ≥ 0.9
   - Full-molecule fingerprint Tanimoto ≥ 0.9 (radius 2, 1024 bits; chirality omitted)
   - Length-normalized SMILES Levenshtein similarity ≥ 0.9
2. Attach SAFE strings, fragment diffs, and exclusive/common fragment inventories.
3. **Step 4** enforces conserved-core constraints then drops outliers in exclusive fragment lengths/counts via Tukey IQR caps (fallback 28 / 4).
4. **Step 5** removes property-delta outliers (|Δ| on MW, logP, TPSA, HBD, HBA, rotatable bonds).

The ``MolDeTox/`` directory containing this file is treated as project root.

Core deps: NumPy/Pandas/TQDM/RDKit plus colocated ``safe_functions`` for SMILES→SAFE (datamol/loguru backed).
Install ``rapidfuzz`` to accelerate Levenshtein batching.

I/O guards restrict reads/writes to the MolDeTox subtree (attach-safe / safe-pipeline families).

CLI recipes (execute inside ``MolDeTox``)::

    python ToxicityCliff_pairing.py pair --workers 8
    python ToxicityCliff_pairing.py attach-safe
    python ToxicityCliff_pairing.py compare-safe
    python ToxicityCliff_pairing.py filter-safe
    python ToxicityCliff_pairing.py property-delta --input pairs_safe_filtered.csv
    python ToxicityCliff_pairing.py drop-outliers --input pairs_xxx_property_delta.csv
    python ToxicityCliff_pairing.py safe-pipeline --pairs data/.../pairs.csv

``safe-pipeline`` performs attach→compare→filter→property deltas→Δ-outlier pruning **purely in memory** and emits
only ``MolDeTox/toxicitycliff.csv`` with the nine ``FINAL_TOXICITYCLIFF_COLUMNS`` (no intermediate CSV dumps).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, MolSurf
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from safe_functions import SAFEEncodeError, smiles_to_safe as _encode_smiles_to_safe

_SCRIPT_DIR = Path(__file__).resolve().parent
MOLDETOX_ROOT = _SCRIPT_DIR

# Default inputs for staged CLIs when --pairs is omitted
DEFAULT_PAIRS_CSV = MOLDETOX_ROOT / "data" / "molecular_ace_pairing_batch" / "pairs.csv"
DEFAULT_PAIRS_SAFE_CSV = MOLDETOX_ROOT / "pairs_safe.csv"
DEFAULT_PAIRS_SAFE_COMPARED_CSV = MOLDETOX_ROOT / "pairs_safe_compared.csv"
DEFAULT_PAIRS_SAFE_FILTERED_CSV = MOLDETOX_ROOT / "pairs_safe_filtered.csv"

FINAL_TOXICITYCLIFF_COLUMNS: List[str] = [
    "dataset_name",
    "endpoint",
    "toxic_smiles",
    "nontoxic_smiles",
    "toxic_safe",
    "nontoxic_safe",
    "only_toxic_safe_fragments",
    "only_nontoxic_safe_fragments",
    "common_safe_fragments",
]

# safe-pipeline always writes this single nine-column artifact
TOXICITYCLIFF_CSV = MOLDETOX_ROOT / "toxicitycliff.csv"

EXCLUDE_PAIRING_DATASETS = {"toxcast_df"}


def _path_must_be_under_moldetox(p: Path, *, label: str) -> Path:
    """Ensure paths stay inside the MolDeTox root."""
    rp = p.resolve()
    root = MOLDETOX_ROOT.resolve()
    try:
        rp.relative_to(root)
    except ValueError as e:
        raise ValueError(f"{label} must be under MolDeTox root {root}: {p}") from e
    return rp


def _canonical_smiles_for_export(val: Any) -> str:
    """Canonical RDKit SMILES (isomeric) for exported toxic/nontoxic reference columns."""
    if val is None:
        return ""
    try:
        if isinstance(val, float) and np.isnan(val):
            return ""
    except Exception:
        pass
    if pd.isna(val):
        return ""
    t = str(val).strip()
    if not t or t.lower() == "nan":
        return ""
    mol = Chem.MolFromSmiles(t)
    if mol is None:
        return t
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def finalize_toxicitycliff_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep nine export columns and canonicalize SMILES strings."""
    out = df.copy()
    if "toxic_smiles" in out.columns:
        out["toxic_smiles"] = out["toxic_smiles"].map(_canonical_smiles_for_export)
    if "nontoxic_smiles" in out.columns:
        out["nontoxic_smiles"] = out["nontoxic_smiles"].map(_canonical_smiles_for_export)
    missing = [c for c in FINAL_TOXICITYCLIFF_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Final export missing columns: {missing}")
    return out[FINAL_TOXICITYCLIFF_COLUMNS]


ECFP_RADIUS = 2
ECFP_SIZE = 1024
SIM_THRESHOLD = 0.9
CHUNK_TOXIC_SMILES = 200

PROPERTY_DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB"]
SAFE_SEP = "."
COL_ONLY_TOXIC_FRAG = "only_toxic_safe_fragments"
COL_ONLY_NONTOXIC_FRAG = "only_nontoxic_safe_fragments"

def _levenshtein_distance_py(a: str, b: str) -> int:
    """Pure-Python Wagner–Fischer distance (no optional deps)."""
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
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def _smiles_similarity_batch_pure_python(
    toxic_chunk: list[str], nontoxic_list: list[str]
) -> np.ndarray:
    """Fallback normalized Levenshtein similarity without rapidfuzz."""
    n_t, n_n = len(toxic_chunk), len(nontoxic_list)
    out = np.zeros((n_t, n_n), dtype=np.float32)
    for i in range(n_t):
        for j in range(n_n):
            d = float(_levenshtein_distance_py(toxic_chunk[i], nontoxic_list[j]))
            mx = float(max(len(toxic_chunk[i]), len(nontoxic_list[j]), 1))
            out[i, j] = 1.0 - (d / mx)
    return out


# =============================================================================
# SMILES / similarity helpers (MolecularACE-aligned)
# =============================================================================


def canonicalize_smiles_list(
    smiles_list: list[str],
    *,
    isomeric: bool = True,
    keep_invalid: bool = True,
) -> list[str]:
    """Return RDKit-canonical SMILES variants."""
    out: list[str] = []
    for smi in smiles_list:
        if not smi or not isinstance(smi, str):
            out.append("" if not keep_invalid else (smi if isinstance(smi, str) else ""))
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                out.append(smi if keep_invalid else "")
                continue
            out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric))
        except Exception:
            out.append(smi if keep_invalid else "")
    return out


def build_ecfp4_list(smiles_list: list[str], include_chirality: bool = False) -> list:
    """ECFP fingerprints (radius 2 / 1024 bits); ``None`` placeholders on failure."""
    fpgen = AllChem.GetMorganGenerator(
        radius=ECFP_RADIUS, fpSize=ECFP_SIZE, includeChirality=include_chirality
    )
    out = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                out.append(fpgen.GetFingerprint(mol))
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return out


def build_scaffold_fp_list(smiles_list: list[str]) -> list:
    """Scaffold fingerprints; ``None`` when Murcko decomposition fails."""
    fpgen = AllChem.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_SIZE, includeChirality=False)
    out = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                out.append(None)
                continue
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
                out.append(None)
                continue
            out.append(fpgen.GetFingerprint(scaffold))
        except Exception:
            out.append(None)
    return out


def smiles_similarity_batch(toxic_chunk: list[str], nontoxic_list: list[str]) -> np.ndarray:
    """Normalized Levenshtein similarity matrix toxic_chunk × nontoxic_list."""
    if not toxic_chunk or not nontoxic_list:
        return np.zeros((len(toxic_chunk), len(nontoxic_list)), dtype=np.float32)
    try:
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        return _smiles_similarity_batch_pure_python(toxic_chunk, nontoxic_list)
    dist = process.cdist(
        toxic_chunk,
        nontoxic_list,
        scorer=Levenshtein.distance,
        dtype=np.int32,
        workers=1,
    )
    len_t = np.array([len(s) for s in toxic_chunk], dtype=np.float32)
    len_n = np.array([len(s) for s in nontoxic_list], dtype=np.float32)
    max_len = np.maximum(len_t[:, None], len_n[None, :])
    np.maximum(max_len, 1.0, out=max_len)
    return 1.0 - (dist.astype(np.float32) / max_len)


def process_endpoint(
    dataset: str,
    endpoint: str,
    toxic_smiles: list[str],
    nontoxic_smiles: list[str],
    save_sim_path: Path | None = None,
    canonicalize_smiles: bool = True,
) -> tuple[str, str, list[tuple[str, str]], int]:
    """
    Paper Step 1: retain candidate pairs fulfilling **any** similarity rule.

    - Scaffold fingerprint Tanimoto ≥ 0.9
    - Full-molecule Morgan Tanimoto ≥ 0.9 (chirality off)
    - Normalized SMILES Levenshtein ≥ 0.9

    Returns ``(dataset, endpoint, pairs, count)``.
    """
    n_t, n_n = len(toxic_smiles), len(nontoxic_smiles)
    empty: list[tuple[str, str]] = []
    if n_t == 0 or n_n == 0:
        return dataset, endpoint, empty, 0

    if canonicalize_smiles:
        toxic_smiles = canonicalize_smiles_list(toxic_smiles, isomeric=True, keep_invalid=True)
        nontoxic_smiles = canonicalize_smiles_list(nontoxic_smiles, isomeric=True, keep_invalid=True)

    fp_toxic_full = build_ecfp4_list(toxic_smiles, include_chirality=False)
    fp_nontoxic_full = build_ecfp4_list(nontoxic_smiles, include_chirality=False)
    valid_n_full = [(j, fp) for j, fp in enumerate(fp_nontoxic_full) if fp is not None]
    if not valid_n_full:
        return dataset, endpoint, empty, 0
    fp_nontoxic_full_list = [f for _, f in valid_n_full]

    fp_toxic_scaffold = build_scaffold_fp_list(toxic_smiles)
    fp_nontoxic_scaffold = build_scaffold_fp_list(nontoxic_smiles)
    valid_n_scaffold = [(j, fp) for j, fp in enumerate(fp_nontoxic_scaffold) if fp is not None]
    fp_nontoxic_scaffold_list = [f for _, f in valid_n_scaffold]

    ecfp4_full_sim = np.full((n_t, n_n), np.nan, dtype=np.float32)
    scaffold_sim = np.full((n_t, n_n), np.nan, dtype=np.float32)

    for i in range(n_t):
        if fp_toxic_full[i] is not None:
            sims = DataStructs.BulkTanimotoSimilarity(fp_toxic_full[i], fp_nontoxic_full_list)
            for k, sim in enumerate(sims):
                j = valid_n_full[k][0]
                ecfp4_full_sim[i, j] = sim
        if fp_toxic_scaffold[i] is not None:
            sims = DataStructs.BulkTanimotoSimilarity(fp_toxic_scaffold[i], fp_nontoxic_scaffold_list)
            for k, sim in enumerate(sims):
                j = valid_n_scaffold[k][0]
                scaffold_sim[i, j] = sim

    smiles_sim = np.full((n_t, n_n), np.nan, dtype=np.float32)
    if nontoxic_smiles:
        for start in range(0, n_t, CHUNK_TOXIC_SMILES):
            end = min(start + CHUNK_TOXIC_SMILES, n_t)
            smiles_sim[start:end, :] = smiles_similarity_batch(toxic_smiles[start:end], nontoxic_smiles)

    full_ok = ecfp4_full_sim >= SIM_THRESHOLD
    scaffold_ok = scaffold_sim >= SIM_THRESHOLD
    smiles_ok = np.isfinite(smiles_sim) & (smiles_sim >= SIM_THRESHOLD)
    pass_pair = full_ok | scaffold_ok | smiles_ok

    pairs_all = set(zip(*np.where(pass_pair)))
    rows_all = [(toxic_smiles[i], nontoxic_smiles[j]) for i, j in pairs_all]

    if save_sim_path is not None:
        save_sim_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_sim_path,
            ecfp4_full_molecule_sim=ecfp4_full_sim,
            substructure_sim=ecfp4_full_sim,
            scaffold_sim=scaffold_sim,
            smiles_sim=smiles_sim,
            n_toxic=n_t,
            n_nontoxic=n_n,
        )

    return dataset, endpoint, rows_all, len(pairs_all)


def _load_and_process_one(args: tuple) -> tuple:
    dataset, endpoint, path_toxic, path_nontoxic, path_sim_npz, canonicalize_smiles = args
    with open(path_toxic, "rb") as f:
        toxic_smiles = pickle.load(f)
    with open(path_nontoxic, "rb") as f:
        nontoxic_smiles = pickle.load(f)
    return process_endpoint(
        dataset,
        endpoint,
        toxic_smiles,
        nontoxic_smiles,
        save_sim_path=path_sim_npz,
        canonicalize_smiles=canonicalize_smiles,
    )


def run_molecular_ace_pairing(
    *,
    summary_csv: Path | None = None,
    out_dir: Path | None = None,
    sim_matrices_dir: Path | None = None,
    n_workers: int | None = None,
    canonicalize_smiles: bool = True,
    project_root: Path | None = None,
    ace_root: Path | None = None,
) -> None:
    """Batch Step 1 pairing from ``scaffold_sim/summary.csv`` + per-endpoint pickle lists (writes ``pairs.csv`` only)."""
    root = project_root or ace_root or MOLDETOX_ROOT
    summary_path = summary_csv or (root / "scaffold_sim" / "summary.csv")
    if not summary_path.exists():
        raise FileNotFoundError(f"Not found: {summary_path}")

    out = out_dir or (root / "data" / "molecular_ace_pairing_batch")
    pairs_csv = out / "pairs.csv"

    out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    summary["_safe_ep"] = summary["path_npz"].apply(lambda p: Path(p).stem)
    summary = summary[~summary["dataset"].isin(EXCLUDE_PAIRING_DATASETS)]

    task_args: list[tuple[str, str, Path, Path, Optional[Path], bool]] = []
    for _, row in summary.iterrows():
        dataset, endpoint = row["dataset"], row["endpoint"]
        safe_ep = row["_safe_ep"]
        npz_dir = (root / Path(row["path_npz"])).parent
        path_toxic = npz_dir / f"{safe_ep}_toxic_smiles.pkl"
        path_nontoxic = npz_dir / f"{safe_ep}_nontoxic_smiles.pkl"
        if path_toxic.exists() and path_nontoxic.exists():
            task_args.append((dataset, endpoint, path_toxic, path_nontoxic, None, canonicalize_smiles))

    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    all_rows: list[dict] = []
    count_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_load_and_process_one, a): a for a in task_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Endpoints"):
            try:
                dataset, endpoint, pairs_list, n_pairs = fut.result()
                count_rows.append({"dataset": dataset, "endpoint": endpoint, "n_pairs": n_pairs})

                def add_rows(rows: list[dict], lst: list[tuple[str, str]]) -> None:
                    for s_t, s_n in lst:
                        rows.append(
                            {
                                "dataset_name": dataset,
                                "endpoint": endpoint,
                                "toxic_smiles": s_t,
                                "nontoxic_smiles": s_n,
                            }
                        )

                add_rows(all_rows, pairs_list)
            except Exception as e:
                args = futures[fut]
                tqdm.write(f"Error {args[0]}/{args[1]}: {e}")

    def save_pairs(path: Path, rows: list[dict], name: str) -> None:
        if rows:
            df = pd.DataFrame(rows).drop_duplicates()
        else:
            df = pd.DataFrame(columns=["dataset_name", "endpoint", "toxic_smiles", "nontoxic_smiles"])
        df.to_csv(path, index=False)
        print(f"Saved: {path} ({len(df):,} rows) [{name}]")

    save_pairs(pairs_csv, all_rows, "all")

    print(f"Endpoints processed: {len(count_rows):,}")
    print(f"Total pairs (all): {pd.DataFrame(count_rows)['n_pairs'].sum():,}")


# =============================================================================
# SAFE mappings
# =============================================================================


def load_safe_mapping(mapping_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    map_df = pd.read_csv(mapping_path)
    if "smiles" not in map_df.columns or "safe" not in map_df.columns:
        raise ValueError("Mapping CSV must have columns 'smiles' and 'safe'.")
    smiles_to_safe = dict(
        zip(map_df["smiles"].astype(str).str.strip(), map_df["safe"].fillna("").astype(str))
    )
    canon_to_safe: dict[str, str] = {}
    if "canonical_smiles" in map_df.columns:
        canon_to_safe = dict(
            zip(
                map_df["canonical_smiles"].astype(str).str.strip(),
                map_df["safe"].fillna("").astype(str),
            )
        )
        canon_to_safe = {k: v for k, v in canon_to_safe.items() if k and str(k) != "nan"}
    return smiles_to_safe, canon_to_safe


def build_safe_mapping_from_pairs_with_encode(
    df: pd.DataFrame,
    *,
    preload_smiles_to_safe: dict[str, str] | None = None,
    preload_canon_to_safe: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Encode ``toxic_smiles`` / ``nontoxic_smiles`` via ``safe_functions.smiles_to_safe``.

    Preloaded CSV entries keep their non-empty SAFE strings without re-encoding.
    """
    smiles_to_safe: dict[str, str] = dict(preload_smiles_to_safe or {})
    canon_to_safe: dict[str, str] = dict(preload_canon_to_safe or {})
    unique: set[str] = set()
    for col in ("toxic_smiles", "nontoxic_smiles"):
        if col not in df.columns:
            continue
        for v in df[col].astype(str):
            s = str(v).strip()
            if s and s.lower() != "nan":
                unique.add(s)
    for s in tqdm(sorted(unique), desc="SMILES→SAFE (safe_functions)"):
        if (smiles_to_safe.get(s) or "").strip():
            continue
        out = ""
        try:
            out = _encode_smiles_to_safe(s, canonical=True) or ""
        except SAFEEncodeError:
            out = ""
        except Exception:
            out = ""
        smiles_to_safe[s] = out
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            can = Chem.MolToSmiles(mol, isomericSmiles=True)
            canon_to_safe[can] = out
    return smiles_to_safe, canon_to_safe


def lookup_safe_column(
    smiles_series: pd.Series,
    smiles_to_safe: dict[str, str],
    canon_to_safe: dict[str, str],
) -> list[str]:
    out: list[str] = []
    for s in smiles_series:
        s = str(s).strip() if pd.notna(s) else ""
        safe_str = smiles_to_safe.get(s)
        if safe_str is None and s:
            safe_str = canon_to_safe.get(s, "")
        out.append(safe_str if safe_str is not None else "")
    return out


def attach_safe_to_pairs(
    df: pd.DataFrame,
    smiles_to_safe: dict[str, str],
    canon_to_safe: dict[str, str],
) -> pd.DataFrame:
    """Augment dataframe with ``toxic_safe`` / ``nontoxic_safe`` lookups."""
    return df.assign(
        toxic_safe=lookup_safe_column(df["toxic_smiles"], smiles_to_safe, canon_to_safe),
        nontoxic_safe=lookup_safe_column(df["nontoxic_smiles"], smiles_to_safe, canon_to_safe),
    )


def run_attach_safe(
    pairs_csv: Path | None = None,
    mapping_csv: Path | None = None,
    output_csv: Path | None = None,
    *,
    write_disk: bool = True,
) -> pd.DataFrame:
    pairs_csv = pairs_csv or DEFAULT_PAIRS_CSV
    output_csv = output_csv or DEFAULT_PAIRS_SAFE_CSV
    _path_must_be_under_moldetox(pairs_csv, label="pairs_csv")
    if write_disk:
        _path_must_be_under_moldetox(output_csv, label="output_csv")

    if not pairs_csv.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")

    df = pd.read_csv(pairs_csv)
    for col in ["toxic_smiles", "nontoxic_smiles"]:
        if col not in df.columns:
            raise ValueError(f"Pairs CSV must have '{col}'.")

    preload_s: dict[str, str] = {}
    preload_c: dict[str, str] = {}
    if mapping_csv is not None:
        _path_must_be_under_moldetox(mapping_csv, label="mapping_csv")
        if mapping_csv.is_file():
            preload_s, preload_c = load_safe_mapping(mapping_csv)

    smiles_to_safe, canon_to_safe = build_safe_mapping_from_pairs_with_encode(
        df,
        preload_smiles_to_safe=preload_s,
        preload_canon_to_safe=preload_c,
    )
    df = attach_safe_to_pairs(df, smiles_to_safe, canon_to_safe)
    n = len(df)
    print(f"Rows missing toxic_safe: {int((df['toxic_safe'] == '').sum())} / {n}")
    print(f"Rows missing nontoxic_safe: {int((df['nontoxic_safe'] == '').sum())} / {n}")

    if write_disk:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")
    return df


# =============================================================================
# SAFE fragment comparisons (compare_safe)
# =============================================================================


def safe_to_fragments(safe_str: Any) -> set[str]:
    if pd.isna(safe_str) or not str(safe_str).strip():
        return set()
    return {s.strip() for s in str(safe_str).split(SAFE_SEP) if s.strip()}


def compare_fragments(toxic_safe: Any, nontoxic_safe: Any) -> tuple[set[str], set[str], set[str]]:
    t_set = safe_to_fragments(toxic_safe)
    n_set = safe_to_fragments(nontoxic_safe)
    common = t_set & n_set
    only_toxic = t_set - n_set
    only_nontoxic = n_set - t_set
    return common, only_toxic, only_nontoxic


def build_unique_safe_json(only_toxic: set[str], only_nontoxic: set[str]) -> str:
    out = []
    for frag in sorted(only_toxic):
        out.append({"fragment": frag, "reason": "only_in_toxic"})
    for frag in sorted(only_nontoxic):
        out.append({"fragment": frag, "reason": "only_in_nontoxic"})
    return json.dumps(out, ensure_ascii=False) if out else "[]"


def enrich_pairs_with_safe_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Add fragment analytics columns compatible with attach-safe CSVs."""
    if "toxic_safe" not in df.columns or "nontoxic_safe" not in df.columns:
        raise ValueError("Need toxic_safe, nontoxic_safe columns.")

    common_list: list[str] = []
    only_toxic_list: list[str] = []
    only_nontoxic_list: list[str] = []
    has_safe_diff_list: list[bool] = []
    unique_safe_list: list[str] = []
    n_common_list: list[int] = []
    n_only_toxic_list: list[int] = []
    n_only_nontoxic_list: list[int] = []
    toxic_fragments_str_list: list[str] = []
    nontoxic_fragments_str_list: list[str] = []

    for _, row in df.iterrows():
        toxic_safe = row.get("toxic_safe", "")
        nontoxic_safe = row.get("nontoxic_safe", "")
        common, only_toxic, only_nontoxic = compare_fragments(toxic_safe, nontoxic_safe)

        toxic_fragments_str_list.append(SAFE_SEP.join(sorted(safe_to_fragments(toxic_safe))))
        nontoxic_fragments_str_list.append(SAFE_SEP.join(sorted(safe_to_fragments(nontoxic_safe))))
        common_list.append(SAFE_SEP.join(sorted(common)))
        only_toxic_list.append(SAFE_SEP.join(sorted(only_toxic)))
        only_nontoxic_list.append(SAFE_SEP.join(sorted(only_nontoxic)))
        has_safe_diff_list.append(len(only_toxic) > 0 or len(only_nontoxic) > 0)
        unique_safe_list.append(build_unique_safe_json(only_toxic, only_nontoxic))
        n_common_list.append(len(common))
        n_only_toxic_list.append(len(only_toxic))
        n_only_nontoxic_list.append(len(only_nontoxic))

    return df.assign(
        toxic_safe_fragments=toxic_fragments_str_list,
        nontoxic_safe_fragments=nontoxic_fragments_str_list,
        common_safe_fragments=common_list,
        only_toxic_safe_fragments=only_toxic_list,
        only_nontoxic_safe_fragments=only_nontoxic_list,
        n_common_safe=n_common_list,
        n_only_toxic_safe=n_only_toxic_list,
        n_only_nontoxic_safe=n_only_nontoxic_list,
        has_safe_diff=has_safe_diff_list,
        unique_safe=unique_safe_list,
    )


def run_compare_safe(
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    *,
    df: pd.DataFrame | None = None,
    write_disk: bool = True,
) -> pd.DataFrame:
    input_csv = input_csv or DEFAULT_PAIRS_SAFE_CSV
    output_csv = output_csv or DEFAULT_PAIRS_SAFE_COMPARED_CSV
    if df is None:
        if not input_csv.exists():
            raise FileNotFoundError(f"Not found: {input_csv}")
        df = pd.read_csv(input_csv)
    df = enrich_pairs_with_safe_comparison(df)

    out_cols = [
        "dataset_name",
        "endpoint",
        "toxic_smiles",
        "nontoxic_smiles",
        "toxic_safe",
        "nontoxic_safe",
        "toxic_safe_fragments",
        "nontoxic_safe_fragments",
        "common_safe_fragments",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
        "n_common_safe",
        "n_only_toxic_safe",
        "n_only_nontoxic_safe",
        "has_safe_diff",
        "unique_safe",
    ]
    df_out = df[[c for c in out_cols if c in df.columns]]
    if write_disk:
        _path_must_be_under_moldetox(output_csv, label="output_csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv} ({len(df_out.columns)} columns)")
    return df_out


# =============================================================================
# SAFE filters (paper Step 4 lengths/counts caps after conserved-core enforcement)
# =============================================================================


def collect_only_fragment_token_lengths(df: pd.DataFrame) -> list[int]:
    """Collect per-token lengths from exclusive-toxic / exclusive-nontoxic fragment columns."""
    lengths: list[int] = []
    for col in (COL_ONLY_TOXIC_FRAG, COL_ONLY_NONTOXIC_FRAG):
        if col not in df.columns:
            continue
        for s in df[col].fillna(""):
            for p in str(s).split(SAFE_SEP):
                p = p.strip()
                if p:
                    lengths.append(len(p))
    return lengths


def tukey_upper_fence_from_values(values: list[int], *, min_points: int = 4) -> float | None:
    if len(values) < min_points:
        return None
    x = pd.Series(values, dtype=float)
    q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return float(q3 + 1.5 * iqr)


def row_has_only_fragment_length_outlier(row: pd.Series, upper_fence: float) -> bool:
    """True when any exclusive fragment token exceeds the Tukey upper fence."""
    for col in (COL_ONLY_TOXIC_FRAG, COL_ONLY_NONTOXIC_FRAG):
        s = row.get(col)
        if s is None:
            continue
        t = str(s).strip()
        if not t or t.lower() == "nan":
            continue
        for p in t.split(SAFE_SEP):
            p = p.strip()
            if p and len(p) > upper_fence:
                return True
    return False


def series_tukey_upper_fence(series: pd.Series, *, fallback: float) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 4:
        return fallback
    q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return fallback
    return float(q3 + 1.5 * iqr)


def _has_any_fragment_ge(s: Any, min_length: int) -> bool:
    """Fixed-threshold mode helper: any exclusive fragment length ≥ ``min_length``."""
    if s is None:
        return False
    t = str(s).strip()
    if not t or t.lower() == "nan":
        return False
    parts = [p.strip() for p in t.split(SAFE_SEP) if p.strip()]
    return any(len(p) >= min_length for p in parts)


def apply_safe_pair_filters(
    df: pd.DataFrame,
    *,
    use_iqr: bool = True,
    fallback_frag_len_ge: int = 28,
    fallback_max_only_count: int = 4,
) -> pd.DataFrame:
    """
    Paper Step 4 filtering.

    - ``use_iqr=True`` (default): apply Tukey fences to exclusive fragment token lengths and ``n_only_*`` counts.
      Sparse / degenerate samples fall back to ``fallback_frag_len_ge`` and ``fallback_max_only_count``.
    - ``use_iqr=False``: reproduce legacy fixed-threshold behavior.
    """
    for c in [
        "n_common_safe",
        "n_only_toxic_safe",
        "n_only_nontoxic_safe",
        COL_ONLY_TOXIC_FRAG,
        COL_ONLY_NONTOXIC_FRAG,
    ]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Run compare-safe step first.")

    n_start = len(df)
    df = df[df["n_common_safe"].ne(0)].copy()
    mask_both_zero = (df["n_only_nontoxic_safe"] == 0) & (df["n_only_toxic_safe"] == 0)
    df = df[~mask_both_zero].copy()
    n_after_core = len(df)

    if not use_iqr:
        mask_long = df.apply(
            lambda r: _has_any_fragment_ge(r.get(COL_ONLY_TOXIC_FRAG), fallback_frag_len_ge)
            or _has_any_fragment_ge(r.get(COL_ONLY_NONTOXIC_FRAG), fallback_frag_len_ge),
            axis=1,
        )
        df = df[~mask_long].copy()
        df = df[
            (df["n_only_toxic_safe"] <= fallback_max_only_count)
            & (df["n_only_nontoxic_safe"] <= fallback_max_only_count)
        ].copy()
        print(
            f"Filtered (fixed thresholds): {n_start} -> {len(df)} "
            f"(len>={fallback_frag_len_ge}, n_only<={fallback_max_only_count}); "
            f"after core rules: {n_after_core}"
        )
        return df

    lengths = collect_only_fragment_token_lengths(df)
    len_upper = tukey_upper_fence_from_values(lengths)
    if len_upper is None:
        len_upper = float(fallback_frag_len_ge - 1)

    mask_long = df.apply(lambda r: row_has_only_fragment_length_outlier(r, len_upper), axis=1)
    df = df[~mask_long].copy()
    n_after_len = len(df)

    upper_nt_only = series_tukey_upper_fence(
        df["n_only_toxic_safe"], fallback=float(fallback_max_only_count)
    )
    upper_nnt_only = series_tukey_upper_fence(
        df["n_only_nontoxic_safe"], fallback=float(fallback_max_only_count)
    )
    df = df[(df["n_only_toxic_safe"] <= upper_nt_only) & (df["n_only_nontoxic_safe"] <= upper_nnt_only)].copy()

    print(
        f"Filtered (IQR): {n_start} -> {len(df)} "
        f"| after shared/non-trivial: {n_after_core} "
        f"| fragment length upper (Tukey): {len_upper:.4g} "
        f"| n_only upper (toxic, nontoxic): {upper_nt_only:.4g}, {upper_nnt_only:.4g} "
        f"| after length step: {n_after_len}"
    )
    return df


def run_filter_safe(
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    *,
    df: pd.DataFrame | None = None,
    use_iqr: bool = True,
    fallback_frag_len_ge: int = 28,
    fallback_max_only_count: int = 4,
    write_disk: bool = True,
) -> pd.DataFrame:
    input_csv = input_csv or DEFAULT_PAIRS_SAFE_COMPARED_CSV
    output_csv = output_csv or DEFAULT_PAIRS_SAFE_FILTERED_CSV
    if df is None:
        if not input_csv.exists():
            raise FileNotFoundError(f"Not found: {input_csv}")
        df = pd.read_csv(input_csv)
    df = apply_safe_pair_filters(
        df,
        use_iqr=use_iqr,
        fallback_frag_len_ge=fallback_frag_len_ge,
        fallback_max_only_count=fallback_max_only_count,
    )
    if write_disk:
        _path_must_be_under_moldetox(output_csv, label="output_csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv}")
    return df


# =============================================================================
# Physicochemical deltas
# =============================================================================


def get_descriptors(smiles: str) -> dict | None:
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "MW": Descriptors.ExactMolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "TPSA": MolSurf.TPSA(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotB": Lipinski.NumRotatableBonds(mol),
        }
    except Exception:
        return None


def add_property_deltas(
    df: pd.DataFrame,
    toxic_col: str = "toxic_smiles",
    nontoxic_col: str = "nontoxic_smiles",
    cache: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Append descriptor columns plus signed/absolute deltas between toxic and nontoxic SMILES."""
    if toxic_col not in df.columns or nontoxic_col not in df.columns:
        raise ValueError(f"CSV must have '{toxic_col}' and '{nontoxic_col}'.")

    n_pairs = len(df)
    cache = cache if cache is not None else {}
    names = PROPERTY_DESCRIPTOR_NAMES

    toxic_prop = {name: [float("nan")] * n_pairs for name in names}
    nontoxic_prop = {name: [float("nan")] * n_pairs for name in names}
    delta_signed = {name: [float("nan")] * n_pairs for name in names}
    delta_abs = {name: [float("nan")] * n_pairs for name in names}

    it = df.iterrows()
    if verbose:
        it = tqdm(it, total=n_pairs, desc="Property delta")

    for pos, (_, row) in enumerate(it):
        smi_t = row[toxic_col]
        smi_n = row[nontoxic_col]
        if smi_t not in cache:
            cache[smi_t] = get_descriptors(smi_t)
        if smi_n not in cache:
            cache[smi_n] = get_descriptors(smi_n)
        desc_t = cache.get(smi_t)
        desc_n = cache.get(smi_n)
        if desc_t is None or desc_n is None:
            continue
        for name in names:
            vt, vn = desc_t.get(name), desc_n.get(name)
            if vt is None or vn is None or not (np.isfinite(vt) and np.isfinite(vn)):
                continue
            vt_f, vn_f = float(vt), float(vn)
            toxic_prop[name][pos] = vt_f
            nontoxic_prop[name][pos] = vn_f
            d = vt_f - vn_f
            delta_signed[name][pos] = d
            delta_abs[name][pos] = abs(d)

    out = df.copy()
    for name in names:
        out[f"toxic_{name}"] = toxic_prop[name]
        out[f"nontoxic_{name}"] = nontoxic_prop[name]
        out[f"delta_{name}"] = delta_signed[name]
        out[f"delta_abs_{name}"] = delta_abs[name]
    return out


def run_property_delta(
    input_csv: Path | None = None,
    output_csv: Path | None = None,
    *,
    df: pd.DataFrame | None = None,
    verbose: bool = True,
    write_disk: bool = True,
) -> pd.DataFrame:
    if df is None:
        if input_csv is None or not input_csv.exists():
            raise FileNotFoundError(f"Not found: {input_csv}")
        df = pd.read_csv(input_csv)

    df = add_property_deltas(df, cache={}, verbose=verbose)

    if write_disk:
        out = output_csv
        if out is None:
            if input_csv is None:
                raise ValueError("output_csv or input_csv required when write_disk=True")
            suffix = input_csv.suffix if input_csv.suffix else ".csv"
            out = input_csv.with_name(input_csv.stem + "_property_delta" + suffix)
        _path_must_be_under_moldetox(out, label="output_csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        n_valid = df["delta_abs_MW"].notna().sum() if "delta_abs_MW" in df.columns else 0
        print(f"Saved: {out} (pairs={len(df):,}, valid MW={n_valid:,})")
    return df


# =============================================================================
# Property outliers (Δ IQR pruning)
# =============================================================================

DEFAULT_OUTLIER_DESCRIPTORS = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB"]


def _iqr_fences(x: pd.Series) -> Tuple[float, float, float, float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return (float("nan"),) * 5
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def compute_delta_abs_thresholds(df: pd.DataFrame, delta_abs_cols: List[str]) -> Dict[str, Dict[str, float]]:
    thr: Dict[str, Dict[str, float]] = {}
    for c in delta_abs_cols:
        q1, q3, iqr, lower, upper = _iqr_fences(df[c])
        thr[c] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
        }
    return thr


def outlier_any_mask_for_thresholds(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]) -> pd.Series:
    masks = []
    for c, t in thresholds.items():
        x = pd.to_numeric(df[c], errors="coerce")
        lower, upper = t["lower_fence"], t["upper_fence"]
        if np.isnan(lower) or np.isnan(upper):
            masks.append(pd.Series(False, index=df.index))
        else:
            masks.append((x < lower) | (x > upper))
    if not masks:
        return pd.Series(False, index=df.index)
    m = masks[0].copy()
    for mm in masks[1:]:
        m |= mm
    return m


def drop_property_outliers_dataframe(
    df: pd.DataFrame,
    descriptors: List[str] | None = None,
) -> pd.DataFrame:
    """Filter ``delta_abs_*`` outliers in-memory via Tukey fences (disk writes handled elsewhere)."""
    descriptors = descriptors or list(DEFAULT_OUTLIER_DESCRIPTORS)
    delta_abs_cols: List[str] = []
    for d in descriptors:
        c = f"delta_abs_{d}"
        if c in df.columns:
            delta_abs_cols.append(c)
        else:
            raise ValueError(f"Missing column: {c}")
    thresholds = compute_delta_abs_thresholds(df, delta_abs_cols)
    out_any = outlier_any_mask_for_thresholds(df, thresholds)
    return df.loc[~out_any].copy()


def run_drop_property_outliers(
    input_csv: Path,
    out_dir: Path | None = None,
    descriptors: List[str] | None = None,
    kept_csv: Path | None = None,
    dropped_csv: Path | None = None,
    thresholds_csv: Path | None = None,
    *,
    df: pd.DataFrame | None = None,
    write_disk: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into kept/dropped subsets using Δ Tukey thresholds (optional CSV auditing)."""
    if df is None:
        if not input_csv.exists():
            raise FileNotFoundError(f"Not found: {input_csv}")
        df = pd.read_csv(input_csv)

    descriptors = descriptors or list(DEFAULT_OUTLIER_DESCRIPTORS)
    delta_abs_cols: List[str] = []
    for d in descriptors:
        c = f"delta_abs_{d}"
        if c in df.columns:
            delta_abs_cols.append(c)
        else:
            raise ValueError(f"Missing column: {c}")

    thresholds = compute_delta_abs_thresholds(df, delta_abs_cols)
    out_any = outlier_any_mask_for_thresholds(df, thresholds)
    dropped = df.loc[out_any].copy()
    kept = df.loc[~out_any].copy()

    if write_disk:
        out_dir = out_dir or (
            (input_csv.parent if input_csv is not None else MOLDETOX_ROOT) / "dropped_property_outliers"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        kept_csv = kept_csv or (out_dir / "pairs_property_outlier_dropped.csv")
        dropped_csv = dropped_csv or (out_dir / "pairs_property_outliers_only.csv")
        thresholds_csv = thresholds_csv or (out_dir / "outlier_thresholds_iqr_1p5.csv")
        thr_rows = []
        for c, t in thresholds.items():
            thr_rows.append({"descriptor": c.replace("delta_abs_", ""), "column": c, **t})
        thr_df = pd.DataFrame(thr_rows)[["descriptor", "column", "q1", "q3", "iqr", "lower_fence", "upper_fence"]]
        thr_df.to_csv(thresholds_csv, index=False)
        kept.to_csv(kept_csv, index=False)
        dropped.to_csv(dropped_csv, index=False)
        total, out_n = len(df), int(out_any.sum())
        print(f"Input rows: {total:,}; dropped (outlier any): {out_n:,}; kept: {len(kept):,}")
        print(f"Saved kept: {kept_csv}")
        print(f"Saved dropped: {dropped_csv}")
    else:
        total, out_n = len(df), int(out_any.sum())
        print(f"Property outlier drop: {total:,} -> kept {len(kept):,} (dropped {out_n:,})")
    return kept, dropped


# =============================================================================
# orchestration
# =============================================================================


def run_safe_pipeline_after_pairing(
    *,
    pairs_csv: Path | None = None,
    mapping_csv: Path | None = None,
    use_iqr: bool = True,
    fallback_frag_len_ge: int = 28,
    fallback_max_only_count: int = 4,
    descriptors: List[str] | None = None,
    skip_property_outlier_drop: bool = False,
) -> pd.DataFrame:
    """
    Run Steps 2–5 purely in RAM and overwrite ``TOXICITYCLIFF_CSV`` with the final nine-column schema.

    Toxic/nontoxic SMILES columns store canonical decoded references downstream.
    """
    pairs_csv = pairs_csv or DEFAULT_PAIRS_CSV
    _path_must_be_under_moldetox(pairs_csv, label="pairs_csv")
    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")

    df = pd.read_csv(pairs_csv)
    for col in ["toxic_smiles", "nontoxic_smiles", "dataset_name", "endpoint"]:
        if col not in df.columns:
            raise ValueError(f"Pairs CSV must have '{col}'.")

    preload_s: dict[str, str] = {}
    preload_c: dict[str, str] = {}
    if mapping_csv is not None:
        _path_must_be_under_moldetox(mapping_csv, label="mapping_csv")
        if mapping_csv.is_file():
            preload_s, preload_c = load_safe_mapping(mapping_csv)

    smiles_to_safe, canon_to_safe = build_safe_mapping_from_pairs_with_encode(
        df,
        preload_smiles_to_safe=preload_s,
        preload_canon_to_safe=preload_c,
    )
    df = attach_safe_to_pairs(df, smiles_to_safe, canon_to_safe)
    df = enrich_pairs_with_safe_comparison(df)
    df = apply_safe_pair_filters(
        df,
        use_iqr=use_iqr,
        fallback_frag_len_ge=fallback_frag_len_ge,
        fallback_max_only_count=fallback_max_only_count,
    )
    df = add_property_deltas(df, cache={}, verbose=False)
    if not skip_property_outlier_drop:
        df = drop_property_outliers_dataframe(df, descriptors=descriptors)

    df = finalize_toxicitycliff_pair_table(df)
    TOXICITYCLIFF_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TOXICITYCLIFF_CSV, index=False)
    print(f"Saved: {TOXICITYCLIFF_CSV} ({len(df):,} rows, {len(FINAL_TOXICITYCLIFF_COLUMNS)} cols)")
    return df


def _add_commands(sub: argparse._SubParsersAction) -> None:
    p_pair = sub.add_parser("pair", help="Batch MolecularACE-style pairing via summary CSV + pickles")
    p_pair.add_argument("--workers", type=int, default=None)
    p_pair.add_argument("--summary", type=Path, default=None)
    p_pair.add_argument("--out-dir", type=Path, default=None)
    p_pair.add_argument("--no-canonicalize-smiles", action="store_true")

    p_safe = sub.add_parser("attach-safe", help="Attach SAFE columns to pairs.csv (safe_functions encoder)")
    p_safe.add_argument("--pairs", type=Path, default=None)
    p_safe.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Optional MolDeTox smiles/safe CSV preload (only blank SMILES re-encoded afterward)",
    )
    p_safe.add_argument("--output", type=Path, default=None)

    p_cmp = sub.add_parser("compare-safe", help="Annotate SAFE fragment intersections/differences")
    p_cmp.add_argument("--input", type=Path, default=None)
    p_cmp.add_argument("--output", type=Path, default=None)

    p_filt = sub.add_parser("filter-safe", help="Apply Step 4 SAFE novelty filters (IQR defaults)")
    p_filt.add_argument("--input", type=Path, default=None)
    p_filt.add_argument("--output", type=Path, default=None)
    p_filt.add_argument(
        "--fixed-thresholds",
        action="store_true",
        help="Swap Tukey fences for deterministic fragment length/count thresholds",
    )
    p_filt.add_argument(
        "--fallback-frag-len-ge",
        type=int,
        default=28,
        help="Treat exclusive fragments with length ≥ this value as outliers (fixed mode / IQR fallback)",
    )
    p_filt.add_argument(
        "--fallback-max-only-count",
        type=int,
        default=4,
        help="Caps on counts of exclusive fragments when IQR collapses / fixed thresholds",
    )

    p_pd = sub.add_parser("property-delta", help="Append RDKit property deltas versus toxic baseline")
    p_pd.add_argument("--input", type=Path, required=True)
    p_pd.add_argument("--output", type=Path, default=None)
    p_pd.add_argument("--quiet", action="store_true")

    p_do = sub.add_parser("drop-outliers", help="Strip rows flagged by Tukey outliers on delta_abs_*")
    p_do.add_argument("--input", type=Path, required=True)
    p_do.add_argument("--out-dir", type=Path, default=None)
    p_do.add_argument("--descriptors", type=str, default=",".join(DEFAULT_OUTLIER_DESCRIPTORS))

    p_all = sub.add_parser(
        "safe-pipeline",
        help="Execute Steps 2–5 entirely in-memory and overwrite MolDeTox/toxicitycliff.csv (9 cols)",
    )
    p_all.add_argument("--pairs", type=Path, default=None, help="Step1 pairs.csv (dataset_name, endpoint, toxic_smiles, nontoxic_smiles)")
    p_all.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="Optional SMILES→SAFE mapping CSV reused by attach-safe",
    )
    p_all.add_argument("--fixed-thresholds", action="store_true")
    p_all.add_argument("--fallback-frag-len-ge", type=int, default=28)
    p_all.add_argument("--fallback-max-only-count", type=int, default=4)
    p_all.add_argument(
        "--skip-property-outlier-drop",
        action="store_true",
        help="Skip Step 5 property-outlier pruning (persist post-SAFE-filter state only)",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified Toxicity-Cliff pairing utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_commands(sub)
    args = parser.parse_args(argv)

    if args.cmd == "pair":
        run_molecular_ace_pairing(
            summary_csv=args.summary,
            out_dir=args.out_dir,
            n_workers=args.workers,
            canonicalize_smiles=not args.no_canonicalize_smiles,
        )
    elif args.cmd == "attach-safe":
        run_attach_safe(
            pairs_csv=args.pairs,
            mapping_csv=args.mapping,
            output_csv=args.output,
        )
    elif args.cmd == "compare-safe":
        run_compare_safe(input_csv=args.input, output_csv=args.output)
    elif args.cmd == "filter-safe":
        run_filter_safe(
            input_csv=args.input,
            output_csv=args.output,
            use_iqr=not args.fixed_thresholds,
            fallback_frag_len_ge=args.fallback_frag_len_ge,
            fallback_max_only_count=args.fallback_max_only_count,
        )
    elif args.cmd == "property-delta":
        run_property_delta(args.input, args.output, verbose=not args.quiet)
    elif args.cmd == "drop-outliers":
        desc = [d.strip() for d in str(args.descriptors).split(",") if d.strip()]
        run_drop_property_outliers(args.input, out_dir=args.out_dir, descriptors=desc)
    elif args.cmd == "safe-pipeline":
        run_safe_pipeline_after_pairing(
            pairs_csv=args.pairs,
            mapping_csv=args.mapping,
            use_iqr=not args.fixed_thresholds,
            fallback_frag_len_ge=args.fallback_frag_len_ge,
            fallback_max_only_count=args.fallback_max_only_count,
            skip_property_outlier_drop=args.skip_property_outlier_drop,
        )


if __name__ == "__main__":
    main()
