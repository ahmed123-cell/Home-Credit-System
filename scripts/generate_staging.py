"""Generate dbt staging models by inferring column types from the raw tables."""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import sql

load_dotenv()

OUT = Path("dbt/models/staging")
INT_RE = r"^-?[0-9]+$"
NUM_RE = r"^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$"

# Known data-quality rules (applied to the application tables and previous_application)
SENTINEL_TABLES = {"application_train", "application_test", "previous_application"}
XNA_TABLES = {"application_train", "application_test"}
DAYS_SENTINEL = 365243


def infer_types(conn, table: str, cols: list[str]) -> dict[str, str]:
    exprs, params = [], []
    for c in cols:
        exprs.append(
            sql.SQL(
                "count(*) filter (where {c} is not null and {c} !~ %s), "
                "count(*) filter (where {c} is not null and {c} !~ %s)"
            ).format(c=sql.Identifier(c))
        )
        params += [INT_RE, NUM_RE]
    query = sql.SQL("select {} from {}").format(
        sql.SQL(", ").join(exprs), sql.Identifier("raw", table)
    )
    row = conn.execute(query, params).fetchone()
    types = {}
    for i, c in enumerate(cols):
        non_int, non_num = row[2 * i], row[2 * i + 1]
        types[c] = (
            "bigint" if non_int == 0 else "double precision" if non_num == 0 else "text"
        )
    return types


def column_expr(table: str, col: str, typ: str) -> str:
    if typ == "text":
        if table in XNA_TABLES:
            return f"nullif({col}, 'XNA') as {col}"
        return f"{col} as {col}"
    cast = f"{col}::{typ}"
    if table in SENTINEL_TABLES and col.startswith("days_"):
        return f"case when {cast} = {DAYS_SENTINEL} then null else {cast} end as {col}"
    return f"{cast} as {col}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "select table_name from information_schema.tables "
                "where table_schema = 'raw' order by 1"
            )
        ]
        for table in tables:
            cols = [
                r[0]
                for r in conn.execute(
                    "select column_name from information_schema.columns "
                    "where table_schema = 'raw' and table_name = %s "
                    "and column_name <> '_loaded_at' order by ordinal_position",
                    (table,),
                )
            ]
            types = infer_types(conn, table, cols)
            body = ",\n    ".join(column_expr(table, c, types[c]) for c in cols)
            model = f"select\n    {body}\nfrom {{{{ source('raw', '{table}') }}}}\n"
            (OUT / f"stg_{table}.sql").write_text(model)
            n_text = sum(t == "text" for t in types.values())
            print(f"stg_{table}.sql: {len(cols)} columns ({n_text} text)")


if __name__ == "__main__":
    main()
