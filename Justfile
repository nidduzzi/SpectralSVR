# SpectralSVR task runner. Requires `just` (https://github.com/casey/just) and
# `uv`. Recipes run inside uv-managed environments (Python 3.13).

# list recipes
default:
    @just --list

host_memory_cap := "systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 --quiet"

# create/refresh the full dev environment
sync:
    uv sync --extra test --extra lint --extra notebook --extra baselines

# run the test suite (pass extra pytest args, e.g. `just test -k fourier`)
test *args:
    {{host_memory_cap}} uv run --extra test --extra baselines pytest {{args}}

# run tests with coverage and print a missing-lines report
cov *args:
    # COVERAGE_CORE=sysmon avoids a torch+coverage C-tracer clash
    COVERAGE_CORE=sysmon {{host_memory_cap}} uv run --extra test --extra baselines coverage run -m pytest {{args}}
    uv run --extra test --extra baselines coverage report -m

# regenerate the committed comparison record over the default seed set (slow; resume with --append)
bench *args:
    {{host_memory_cap}} uv run --extra baselines python scripts/run_benchmark.py --out benchmarks/operator_comparison.csv {{args}}

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

# mutation-test the operator harness (slow; regenerates mutation.sqlite)
mutate:
    uv run --extra mutation cosmic-ray baseline cosmic-ray.toml
    rm -f mutation.sqlite
    uv run --extra mutation cosmic-ray init cosmic-ray.toml mutation.sqlite
    uv run --extra mutation cosmic-ray exec cosmic-ray.toml mutation.sqlite
    uv run --extra mutation cr-report mutation.sqlite

# mutation-test the lines of config's module changed since base (slow)
mutate-changed config base:
    uv run --extra mutation cosmic-ray baseline {{config}}
    rm -f mutation-{{file_stem(config)}}.sqlite
    uv run --extra mutation cosmic-ray init {{config}} mutation-{{file_stem(config)}}.sqlite
    printf '[cosmic-ray.filters.git-filter]\nbranch = "{{base}}"\n' | uv run --extra mutation cr-filter-git --config - mutation-{{file_stem(config)}}.sqlite
    uv run --extra mutation cosmic-ray exec {{config}} mutation-{{file_stem(config)}}.sqlite
    uv run --extra mutation cr-report mutation-{{file_stem(config)}}.sqlite

# everything CI should gate on
check: lint test
