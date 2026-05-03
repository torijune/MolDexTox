"""
Bemis–Murcko scaffold train/test split (no validation split).

Fixed ratio **train : test = 9 : 1** (``FRAC_TRAIN=0.9``, ``FRAC_TEST=0.1``).

- Uses ``ScaffoldSplitter`` only for indices.
- ``split_pairs_csv_to_train_test()`` reads a ToxicityCliff/merged-style pair CSV and writes
  ``train.csv`` / ``test.csv`` in the chosen directory → ready for ``Build_MolDeTox_QA.py``.

CLI::

    python spliter.py --input pairs_safe_filtered.csv --out-dir ./splits/my_run
    # -> ./splits/my_run/train.csv , test.csv

Requires ``rdkit`` (stdlib ``csv`` only besides that).
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, List, Optional, Protocol, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# Fixed split ratio train : test = 9 : 1
FRAC_TRAIN = 0.9
FRAC_TEST = 0.1


class _DatasetLike(Protocol):
    """Dataset exposes SMILES strings in ``ids``."""

    ids: Sequence[str]

    def __len__(self) -> int: ...


class Splitter:
    """Base class for splitter implementations."""

    def split(
        self,
        dataset: Any,
        seed: Optional[int] = None,
        log_every_n: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        raise NotImplementedError


def _generate_scaffold(
    smiles: str,
    include_chirality: bool = False,
) -> Union[str, None]:
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.") from None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.info(
            "Not generating scaffold for smiles %s - invalid smiles string",
            smiles,
        )
        return None

    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


class _SimpleSmilesDataset:
    """Minimal dataset adapter for ScaffoldSplitter (no DeepChem)."""

    def __init__(self, ids: List[str]) -> None:
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)


class ScaffoldSplitter(Splitter):
    """
    Group indices by Bemis–Murcko scaffold, then allocate ~90% train / ~10% test by scaffold set size ordering.
    No validation split.

    Rows with no scaffold (invalid/empty SMILES) are appended to **train** only.
    """

    def split(
        self,
        dataset: _DatasetLike,
        *,
        seed: Optional[int] = None,
        log_every_n: Optional[int] = 1000,
    ) -> Tuple[List[int], List[int]]:
        del seed  # deterministic split; kept for API compatibility

        scaffold_sets = self.generate_scaffolds(dataset, log_every_n=log_every_n or 1000)

        n = len(dataset)
        test_cutoff = FRAC_TEST * n
        train_inds: List[int] = []
        test_inds: List[int] = []

        logger.info(
            "Scaffold split train:test = %.1f:%.1f (test-first, no valid)",
            FRAC_TRAIN,
            FRAC_TEST,
        )
        for scaffold_set in reversed(scaffold_sets):
            if len(test_inds) + len(scaffold_set) <= test_cutoff:
                test_inds += scaffold_set
            else:
                train_inds += scaffold_set

        # Rows without a scaffold bucket are forced into train
        assigned = set(train_inds) | set(test_inds)
        orphans = [i for i in range(n) if i not in assigned]
        if orphans:
            logger.warning(
                "%d rows without scaffold assignment → appended to train (invalid/blank SMILES, etc.).",
                len(orphans),
            )
            train_inds.extend(orphans)

        train_inds.sort()
        test_inds.sort()
        return train_inds, test_inds

    def generate_scaffolds(
        self,
        dataset: _DatasetLike,
        log_every_n: int = 1000,
    ) -> List[List[int]]:
        scaffolds: dict[str, List[int]] = {}
        data_len = len(dataset)

        logger.info("About to generate scaffolds")
        for ind, smiles in enumerate(dataset.ids):
            if ind % log_every_n == 0:
                logger.info("Generating scaffold %d/%d", ind, data_len)
            scaffold = _generate_scaffold(str(smiles))
            if scaffold is not None:
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = [ind]
                else:
                    scaffolds[scaffold].append(ind)

        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (_scaffold, scaffold_set) in sorted(
                scaffolds.items(),
                key=lambda x: (len(x[1]), x[1][0]),
                reverse=True,
            )
        ]
        return scaffold_sets


def _str_or_empty(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def _merge_toxicity_cliff_row_aliases(row: dict[str, str]) -> None:
    """Same column aliases as Build_MolDeTox_QA for split-time SMILES fields."""
    if not _str_or_empty(row.get("toxic_safe_decoded_smiles", "")):
        row["toxic_safe_decoded_smiles"] = _str_or_empty(row.get("toxic_smiles", ""))
    if not _str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")):
        row["nontoxic_safe_decoded_smiles"] = _str_or_empty(row.get("nontoxic_smiles", ""))
    row.setdefault("common_safe_fragments", "")


def resolve_smiles_column(fieldnames: Sequence[str], explicit: Optional[str] = None) -> str:
    """Resolve toxic-side SMILES column for scaffold hashing."""
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"--smiles-col={explicit!r} is not among CSV headers.")
        return explicit
    for c in ("toxic_safe_decoded_smiles", "toxic_smiles"):
        if c in fieldnames:
            return c
    raise ValueError(
        "Cannot find SMILES column: expected toxic_safe_decoded_smiles or toxic_smiles headers, "
        "or pass --smiles-col."
    )


def split_pairs_csv_to_train_test(
    input_csv: Path | str,
    output_dir: Path | str,
    *,
    smiles_col: Optional[str] = None,
    train_filename: str = "train.csv",
    test_filename: str = "test.csv",
    encoding: str = "utf-8",
) -> Tuple[Path, Path, int, int]:
    """
    Read a pair CSV and write scaffold 9:1 ``train.csv`` / ``test.csv``.

    Parameters
    ----------
    input_csv
        Output of ToxicityCliff_pairing or any merged pair CSV compatible with Build_MolDeTox_QA.
    output_dir
        Target directory (created if missing).

    Returns
    -------
    train_path, test_path, n_train, n_test
    """
    input_path = Path(input_csv).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"CSV has no header row: {input_path}")

        col = resolve_smiles_column(fieldnames, smiles_col)
        rows: List[dict[str, str]] = []
        for raw in reader:
            row = {k: _str_or_empty(raw.get(k, "")) for k in fieldnames}
            _merge_toxicity_cliff_row_aliases(row)
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV has no data rows: {input_path}")

    smiles_list = [_str_or_empty(r.get(col, "")) for r in rows]
    ds = _SimpleSmilesDataset(smiles_list)
    train_idx, test_idx = ScaffoldSplitter().split(ds)

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    train_path = out_dir / train_filename
    test_path = out_dir / test_filename

    with open(train_path, "w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(train_rows)

    with open(test_path, "w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(test_rows)

    logger.info(
        "Saved train=%d rows -> %s, test=%d rows -> %s",
        len(train_rows),
        train_path,
        len(test_rows),
        test_path,
    )
    return train_path, test_path, len(train_rows), len(test_rows)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(
        description="Toxicity cliff pair CSV → scaffold 9:1 train.csv / test.csv",
    )
    ap.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input pair CSV (e.g. pairs_safe_filtered.csv)",
    )
    ap.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory for train.csv and test.csv",
    )
    ap.add_argument(
        "--smiles-col",
        type=str,
        default=None,
        help="Column for scaffold hashing (default toxic_safe_decoded_smiles else toxic_smiles)",
    )
    ap.add_argument(
        "--train-name",
        type=str,
        default="train.csv",
        help="Train output filename (default train.csv)",
    )
    ap.add_argument(
        "--test-name",
        type=str,
        default="test.csv",
        help="Test output filename (default test.csv)",
    )
    args = ap.parse_args(argv)

    split_pairs_csv_to_train_test(
        args.input,
        args.out_dir,
        smiles_col=args.smiles_col,
        train_filename=args.train_name,
        test_filename=args.test_name,
    )


__all__ = [
    "FRAC_TRAIN",
    "FRAC_TEST",
    "Splitter",
    "ScaffoldSplitter",
    "_SimpleSmilesDataset",
    "_generate_scaffold",
    "resolve_smiles_column",
    "split_pairs_csv_to_train_test",
    "main",
]

if __name__ == "__main__":
    main()
