# SpectralSVR task runner. Requires `just` (https://github.com/casey/just) and
# `uv`. Recipes run inside uv-managed environments (Python 3.13).

# list recipes
default:
    @just --list

host_memory_cap := "systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet"
test_workers := "-n 6"
threads_per_worker := "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"

# create/refresh the full dev environment
sync:
    uv sync --extra test --extra lint --extra notebook --extra baselines

# run the test suite (pass extra pytest args, e.g. `just test -k fourier`)
test *args:
    {{threads_per_worker}} {{host_memory_cap}} uv run --extra test --extra baselines pytest {{test_workers}} {{args}}

# run tests with coverage and print a missing-lines report
cov *args:
    # COVERAGE_CORE=sysmon avoids a torch+coverage C-tracer clash
    COVERAGE_CORE=sysmon {{host_memory_cap}} uv run --extra test --extra baselines coverage run -m pytest {{args}}
    uv run --extra test --extra baselines coverage report -m

# regenerate the committed comparison record over the default seed set (slow; resume with --append)
bench *args:
    {{host_memory_cap}} uv run --extra baselines python scripts/run_benchmark.py --device cuda --out benchmarks/operator_comparison.csv {{args}}

# lint + type-check (ruff, mypy, ty)
lint:
    uv run --extra lint ruff check SpectralSVR tests scripts
    uv run --extra lint mypy SpectralSVR scripts
    uv run --extra lint ty check SpectralSVR scripts

# auto-format and apply ruff fixes
fmt:
    uv run --extra lint ruff format SpectralSVR tests scripts
    uv run --extra lint ruff check --fix SpectralSVR tests scripts

# individual linters
ruff:
    uv run --extra lint ruff check SpectralSVR tests scripts

mypy:
    uv run --extra lint mypy SpectralSVR scripts

ty:
    uv run --extra lint ty check SpectralSVR scripts

mutation_home := env("XDG_CACHE_HOME", home_directory() / ".cache") / "spectralsvr-mutation"

# mutation-test a target in mutation/, skipping declared equivalents; with base, only lines changed since base (slow)
mutate target base="":
    mkdir -p {{mutation_home}}
    rm -f {{mutation_home}}/{{target}}.sqlite
    {{threads_per_worker}} {{host_memory_cap}} uv run --extra test --extra baselines cosmic-ray baseline mutation/{{target}}.toml
    uv run --extra test cosmic-ray init mutation/{{target}}.toml {{mutation_home}}/{{target}}.sqlite
    {{ if base == "" { "" } else { "printf '[cosmic-ray.filters.git-filter]\\nbranch = \"" + base + "\"\\n' | uv run --extra test cr-filter-git --config - " + mutation_home / target + ".sqlite" } }}
    uv run --extra test python scripts/equivalent_mutants.py {{mutation_home}}/{{target}}.sqlite
    {{threads_per_worker}} {{host_memory_cap}} uv run --extra test --extra baselines cosmic-ray exec mutation/{{target}}.toml {{mutation_home}}/{{target}}.sqlite
    uv run --extra test cr-report {{mutation_home}}/{{target}}.sqlite > {{mutation_home}}/{{target}}.report.txt
    tail -n 4 {{mutation_home}}/{{target}}.report.txt

# everything CI should gate on
check: lint test
