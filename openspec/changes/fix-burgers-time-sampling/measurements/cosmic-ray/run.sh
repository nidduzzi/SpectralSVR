#!/usr/bin/env bash
set -eu
tree=$1
config=$2
session=$3
archived=$4
first_row=$5
last_row=$6
cd "$tree"
uv run --extra test --extra mutation python "$archived/build_test_map.py" "$tree" "$tree.test_map.json"
uv run --extra mutation cosmic-ray baseline "$config"
uv run --extra mutation cosmic-ray init "$config" "$session"
python3 - "$session" "$first_row" "$last_row" <<'PY'
import sqlite3, sys
session, first_row, last_row = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with sqlite3.connect(session) as db:
    db.execute("DELETE FROM mutation_specs WHERE start_pos_row NOT BETWEEN ? AND ?", (first_row, last_row))
    db.execute("DELETE FROM work_items WHERE job_id NOT IN (SELECT job_id FROM mutation_specs)")
    print("mutants kept:", db.execute("SELECT count(*) FROM work_items").fetchone()[0])
PY
uv run --extra mutation cosmic-ray exec "$config" "$session"
uv run --extra mutation cr-report "$session"
