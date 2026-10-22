#!/usr/bin/env bash
root=${MUTATION_ROOT:-$HOME/.cache/spectralsvr-mutation}
started=$(date +%s.%N)
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -p CPUQuota=${MUTATION_CPU_QUOTA:-400%} -p TasksMax=${MUTATION_TASKS_MAX:-512} --quiet \
  uv run --extra test --extra baselines pytest --durations=5 --durations-min=0.5 "$@"
status=$?
elapsed=$(echo "$(date +%s.%N) - $started" | bc)
mutated=$(git diff --unified=0 | grep -E '^(\+\+\+ |@@ |\+[^+])' | tr '\n' ' ')
printf '%s\t%s\t%.2f\t%s\t%s\n' "$(date -Is)" "$(basename "$PWD")" "$elapsed" "$status" "$mutated" >> "$root/timings.tsv"
exit $status
