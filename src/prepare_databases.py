"""Download the Hugging Face datasets and create three SQLite databases."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DATASETS = {
    "institutions": {
        "repo": "Mahadih534/Institutional-Information-of-Bangladesh",
        "db": DATA_DIR / "institutions.db",
        "table": "institutions",
    },
    "hospitals": {
        "repo": "Mahadih534/all-bangladeshi-hospitals",
        "db": DATA_DIR / "hospitals.db",
        "table": "hospitals",
    },
    "restaurants": {
        "repo": "Mahadih534/Bangladeshi-Restaurant-Data",
        "db": DATA_DIR / "restaurants.db",
        "table": "restaurants",
    },
}

def clean_column_name(value: Any) -> str:
    """Convert a source header to a stable SQLite-friendly name."""
    name = str(value).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    if not name:
        name = "unnamed_column"
    if name[0].isdigit():
        name = f"col_{name}"
    return name

def infer_sql_type(series: pd.Series) -> str:
    """Infer a practical SQLite type without forcing identifiers to numbers."""
    non_null = series.dropna()
    if non_null.empty:
        return "TEXT"
    if pd.api.types.is_bool_dtype(series):
        return "INTEGER"
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"

    # Convert numeric-looking object columns where nearly all values parse.
    values = non_null.astype(str).str.strip()
    numeric = pd.to_numeric(values.str.replace(",", "", regex=False), errors="coerce")
    if len(numeric) and numeric.notna().mean() >= 0.95:
        if (numeric.dropna() % 1 == 0).all():
            return "INTEGER"
        return "REAL"
    return "TEXT"

def load_dataset_frame(repo_id: str) -> pd.DataFrame:
    """Load a CSV/tabular Hugging Face dataset, combining splits if needed."""
    dataset = load_dataset(repo_id)
    frames = []
    for split_name in dataset:
        split = dataset[split_name]
        frame = split.to_pandas()
        if not frame.empty:
            frame["source_split"] = split_name
            frames.append(frame)
    if not frames:
        raise ValueError(f"No rows found in Hugging Face dataset: {repo_id}")
    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [clean_column_name(c) for c in combined.columns]

    # Ensure headers remain unique after normalization.
    seen = {}
    new_columns = []
    for col in combined.columns:
        seen[col] = seen.get(col, 0) + 1
        new_columns.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
    combined.columns = new_columns
    return combined

def create_database(frame: pd.DataFrame, db_path: Path, table_name: str) -> None:
    """Create a SQLite table using inferred column types."""
    if frame.empty:
        raise ValueError(f"Refusing to create empty table: {table_name}")
    column_types = {col: infer_sql_type(frame[col]) for col in frame.columns}
    with sqlite3.connect(db_path) as conn:
        frame.to_sql(table_name, conn, if_exists="replace", index=False)
        # SQLite's pandas-created schema is adequate for common types; print
        # inferred types so they can be reviewed against the source dataset.
        print(f"\n{db_path.name} -> {table_name}: {len(frame):,} rows")
        print("Columns:")
        for col, sql_type in column_types.items():
            print(f"  {col}: {sql_type}")
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_source_split" '
            f'ON "{table_name}" ("source_split")'
        ) if "source_split" in frame.columns else None

def main() -> None:
    for key, cfg in DATASETS.items():
        print(f"\nLoading {cfg['repo']} ...")
        frame = load_dataset_frame(cfg["repo"])
        create_database(frame, cfg["db"], cfg["table"])
    print("\nAll three databases are ready in the data/ directory.")

if __name__ == "__main__":
    main()
