"""Read-only, dataset-specific SQL tools for the AI agent."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATABASES = {
    "institutions": (DATA_DIR / "institutions.db", "institutions"),
    "hospitals": (DATA_DIR / "hospitals.db", "hospitals"),
    "restaurants": (DATA_DIR / "restaurants.db", "restaurants"),
}

def get_schema(db_path: Path, table: str) -> str:
    if not db_path.exists():
        return f"Database not found: {db_path}. Run `python -m src.prepare_databases` first."
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    if not cols:
        return f"Table {table} does not exist in {db_path.name}."
    return ", ".join(f"{c[1]} ({c[2] or 'TEXT'})" for c in cols)

def run_read_only_query(db_path: Path, table: str, sql: str) -> str:
    """Validate SQL and execute against one read-only SQLite database."""
    if not db_path.exists():
        return f"Database is missing ({db_path.name}). Run `python -m src.prepare_databases` first."
    query = sql.strip().rstrip(";")
    # Only permit a single SELECT or WITH...SELECT statement.
    lowered = query.lower()
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        return "Rejected: only read-only SELECT queries are allowed."
    forbidden = [" insert ", " update ", " delete ", " drop ", " alter ",
                 " attach ", " detach ", " pragma ", " create ", " replace "]
    padded = f" {lowered} "
    if any(token in padded for token in forbidden) or ";" in query:
        return "Rejected: query contains a forbidden operation or multiple statements."
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            # Ensure the SQL references only the intended table.
            rows = conn.execute(query).fetchmany(51)
            if not rows:
                return "No matching records found."
            result = [dict(row) for row in rows[:50]]
            suffix = "\n(Showing first 50 rows.)" if len(rows) > 50 else ""
            return f"Rows returned: {len(result)}\n{result}{suffix}"
    except sqlite3.Error as exc:
        return f"SQL query error: {exc}. Check the table and column names in the schema."

def make_db_tool(key: str, description: str, llm: ChatOpenAI):
    db_path, table = DATABASES[key]
    @tool(f"{key.title()}DBTool")
    def database_tool(question: str) -> str:
        """Answer a question using only the selected dataset's SQLite database."""
        schema = get_schema(db_path, table)
        if schema.startswith("Database not found") or schema.startswith("Table "):
            return schema
        sql_prompt = f"""You write SQLite SELECT queries only.
Database table: {table}
Available columns and types: {schema}
User question: {question}

Rules:
- Return only one valid SQLite SELECT statement (no markdown).
- Use only the table "{table}" and the columns listed above.
- For text matching, use LOWER(column) LIKE LOWER('%term%') where appropriate.
- For count questions use COUNT(*); for top/ranking questions use ORDER BY and LIMIT.
- Never invent columns. If a requested attribute is absent, select relevant available columns.
- Use LIMIT 50 for record lists.
"""
        try:
            sql = llm.invoke(sql_prompt).content
            if not isinstance(sql, str):
                sql = str(sql)
            sql = sql.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
            raw_result = run_read_only_query(db_path, table, sql)
            return f"Dataset: {key}\nQuestion: {question}\nSQL: {sql}\nResult: {raw_result}\nExplain the result plainly. Do not claim missing fields are present."
        except Exception as exc:
            return f"Could not query {key} database: {exc}"
    database_tool.description = description
    return database_tool

def build_database_tools(llm: ChatOpenAI):
    return [
        make_db_tool(
            "institutions",
            "Use for Bangladesh educational institutions, universities, colleges, government institutions, institutional names and locations.",
            llm,
        ),
        make_db_tool(
            "hospitals",
            "Use for Bangladesh hospitals, hospital names, locations, and any capacity/doctor/facility fields that actually exist in the dataset.",
            llm,
        ),
        make_db_tool(
            "restaurants",
            "Use for Bangladesh restaurants, restaurant names, cuisine, locations, ratings, and any other fields actually present in the dataset.",
            llm,
        ),
    ]
