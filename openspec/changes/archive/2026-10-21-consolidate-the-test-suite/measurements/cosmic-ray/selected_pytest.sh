#!/usr/bin/env bash
root=${MUTATION_ROOT:-$HOME/.cache/spectralsvr-mutation}
configs=$(cd "$(dirname "$0")" && pwd)
mapfile -t selection < <(uv run --extra mutation python "$configs/select_tests.py" "$root/$(basename "$PWD").test_map.json")
seconds=${selection[0]}
tests=("${selection[@]:1}")
parallel=()
if [ "$(echo "$seconds > ${MUTATION_PARALLEL_ABOVE:-20}" | bc)" = 1 ]; then
  parallel=(-n "${MUTATION_WORKERS:-4}")
  export MUTATION_CPU_QUOTA=${MUTATION_PARALLEL_CPU_QUOTA:-800%}
fi
"$configs/timed_pytest.sh" -x -q -p no:cacheprovider --hypothesis-seed=0 "${parallel[@]}" "${tests[@]}"
status=$?
if [ "$status" = 5 ]; then
  export MUTATION_CPU_QUOTA=${MUTATION_PARALLEL_CPU_QUOTA:-800%}
  exec "$configs/timed_pytest.sh" -x -q -p no:cacheprovider --hypothesis-seed=0 -n "${MUTATION_WORKERS:-4}" tests
fi
exit "$status"
