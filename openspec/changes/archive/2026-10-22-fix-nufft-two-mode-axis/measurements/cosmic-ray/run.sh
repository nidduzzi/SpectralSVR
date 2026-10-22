#!/usr/bin/env bash
set -eu
tree=$1
config=$2
session=$3
archived=$4
cd "$tree"
uv run --extra test --extra mutation python "$archived/build_test_map.py" "$tree" "$tree.test_map.json"
uv run --extra mutation cosmic-ray baseline "$config"
uv run --extra mutation cosmic-ray init "$config" "$session"
uv run --extra mutation cosmic-ray exec "$config" "$session"
uv run --extra mutation cr-report "$session"
