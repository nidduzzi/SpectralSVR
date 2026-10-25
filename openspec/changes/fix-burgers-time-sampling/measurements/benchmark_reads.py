import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / "scripts"))
import run_benchmark as bench  # noqa: E402

DATASETS = ("burgers-numerical", "burgers-forced")
DEVICE = torch.device("cuda")


def clean_checkout_commit():
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout
    if changed:
        raise SystemExit(f"tracked files differ from the checked-out commit:\n{changed}")
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


torch.cuda.set_per_process_memory_fraction(bench.ACCELERATOR_MEMORY_FRACTION)
commit = clean_checkout_commit()
for key in DATASETS:
    spec = bench.DATASETS[key]
    for seed in bench.SEED_SET:
        case = spec.build(seed, False, DEVICE)
        (operator,) = bench.build_operators(["spectralsvr"], case.query, spec.deeponet, seed, DEVICE)
        fitted = operator.fit(case.split.train)
        prediction = fitted.predict(case.split.test.f, case.query)
        targets = case.split.test.targets(case.query)
        json.dump(
            {
                "commit": commit,
                "dataset": key,
                "seed": seed,
                "train_sensors": digest(case.split.train.sensors),
                "train_f": digest(case.split.train.f),
                "train_u": digest(case.split.train.u.coeff),
                "test_f": digest(case.split.test.f),
                "test_u": digest(case.split.test.u.coeff),
                "query": digest(case.query),
                "targets": digest(targets),
                "prediction": digest(prediction),
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        torch.save(
            {"targets": targets.detach().cpu(), "prediction": prediction.detach().cpu()},
            Path(sys.argv[1]) / f"{commit}_{key}_{seed}.pt",
        )
