"""Load raw Home Credit CSVs into PostgreSQL `raw` schema (no transformations)."""

import csv
import logging
import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import sql

load_dotenv()  # reads .env from the current working directory

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
SCHEMA = "raw"
CHUNK_SIZE = 1024 * 1024  # 1 MB

# Known row counts from the Kaggle dataset, used as a sanity check
EXPECTED_ROWS = {
    "application_train": 307_511,
    "application_test": 48_744,
    "bureau": 1_716_428,
    "bureau_balance": 27_299_925,
    "credit_card_balance": 3_840_312,
    "installments_payments": 13_605_401,
    "pos_cash_balance": 10_001_358,
    "previous_application": 1_670_214,
    "sample_submission": 48_744,
}

log = logging.getLogger("ingestion")


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def load_file(conn: psycopg.Connection, path: Path) -> int:
    table = path.stem.lower()
    columns = [c.lower() for c in read_header(path)]
    target = sql.Identifier(SCHEMA, table)
    col_idents = [sql.Identifier(c) for c in columns]

    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(target))
        cur.execute(
            sql.SQL(
                "CREATE TABLE {} ({}, _loaded_at timestamptz DEFAULT now())"
            ).format(
                target,
                sql.SQL(", ").join(sql.SQL("{} text").format(c) for c in col_idents),
            )
        )
        copy_sql = sql.SQL(
            "COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)"
        ).format(target, sql.SQL(", ").join(col_idents))

        with cur.copy(copy_sql) as copy, path.open("rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                copy.write(chunk)

        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(target))
        rows = cur.fetchone()[0]
    conn.commit()

    expected = EXPECTED_ROWS.get(table)
    if expected is not None and rows != expected:
        log.warning(
            "%s: loaded %s rows, expected %s", table, f"{rows:,}", f"{expected:,}"
        )
    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit("Set DATABASE_URL, e.g. postgresql://user:pass@localhost:5432/dbname")

    files = sorted(RAW_DIR.glob("*.csv"))
    with psycopg.connect(dsn) as conn:
        conn.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA))
        )
        conn.commit()
        for path in files:
            start = time.perf_counter()
            rows = load_file(conn, path)
            log.info(
                "%-28s %12s rows  %6.1fs",
                path.name,
                f"{rows:,}",
                time.perf_counter() - start,
            )


if __name__ == "__main__":
    main()
