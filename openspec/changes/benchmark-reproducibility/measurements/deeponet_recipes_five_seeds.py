import json
import sys
import time

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25)

sys.path.insert(0, "scripts")
import run_benchmark as bench
from SpectralSVR.operators import DeepONetOperator, Lu2021DeepxdeExample, Lu2022Burgers
from SpectralSVR.utils import get_metrics

recipe_key, dataset, seed, iterations = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
case = bench.DATASETS[dataset].build(seed, False, torch.device("cpu"))
target = case.split.test.targets(case.query).cpu().contiguous()
periods = (1.0,) if case.query.shape[1] == 1 else (None, 1.0)
recipe = Lu2021DeepxdeExample() if recipe_key == "lu2021" else Lu2022Burgers(periods=periods)
operator = DeepONetOperator(trunk=case.query, recipe=recipe, iterations=iterations, seed=seed)
start = time.perf_counter()
fitted = operator.fit(case.split.train)
seconds = time.perf_counter() - start
pred = fitted.predict(case.split.test.f, case.query).cpu().contiguous()
metrics = get_metrics(pred, target)
print(json.dumps({"recipe": recipe.name, "dataset": dataset, "seed": seed, "iterations": iterations,
                  "r2": metrics["r2"], "rel_l2": metrics["mean_per_sample_relative_l2"], "rrse": metrics["rrse"],
                  "seconds": round(seconds, 1)}))
