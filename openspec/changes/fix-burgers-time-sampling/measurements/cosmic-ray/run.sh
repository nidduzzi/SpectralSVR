#!/usr/bin/env bash
set -eu
tree=$1
config=$2
session=$3
archived=$4
row_ranges=$5
cd "$tree"
uv run --extra test --extra mutation python "$archived/build_test_map.py" "$tree" "$tree.test_map.json"
uv run --extra mutation cosmic-ray baseline "$config"
uv run --extra mutation cosmic-ray init "$config" "$session"
python3 - "$session" "$row_ranges" <<'PY'
import sqlite3, sys
session, row_ranges = sys.argv[1], sys.argv[2]
kept = [tuple(map(int, span.split("-"))) for span in row_ranges.split(",")]
outside = " AND ".join("start_pos_row NOT BETWEEN ? AND ?" for _ in kept)
with sqlite3.connect(session) as db:
    db.execute(f"DELETE FROM mutation_specs WHERE {outside}", [row for span in kept for row in span])
    db.execute("DELETE FROM work_items WHERE job_id NOT IN (SELECT job_id FROM mutation_specs)")
    print("mutants kept:", db.execute("SELECT count(*) FROM work_items").fetchone()[0])
PY
uv run --extra mutation cosmic-ray exec "$config" "$session"
uv run --extra mutation cr-report "$session"
