#!/usr/bin/env bash
set -u
phase=$1
configs=$(cd "$(dirname "$0")" && pwd)
root=${MUTATION_ROOT:-$HOME/.cache/spectralsvr-mutation}
for tree in base after; do
  cd "$root/$tree" || exit 1
  for config in "$configs"/"$phase"-*.toml; do
    module=$(basename "$config" .toml)
    session="$root/$tree-$module.sqlite"
    echo "=== $tree $module $(date -Is)"
    uv run --extra mutation cosmic-ray baseline "$config" || { echo "baseline failed: $tree $module"; exit 1; }
    [ -f "$session" ] || uv run --extra mutation cosmic-ray init "$config" "$session"
    uv run --extra mutation cosmic-ray exec "$config" "$session"
    uv run --extra mutation cr-report "$session" > "$root/$tree-$module.report.txt"
    tail -3 "$root/$tree-$module.report.txt"
  done
done
echo "=== done $(date -Is)"
