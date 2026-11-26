import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "scripts")
import run_benchmark as bench
from SpectralSVR.operators import deeponet as adapter
from SpectralSVR.utils import get_metrics

variant, dataset, shift = sys.argv[1], sys.argv[2], float(sys.argv[3])
iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 10_000
host = torch.device("cpu")
case = bench.DATASETS[dataset].build(0, False, host)

use_he = variant in ("he", "he_in", "he_in_out", "he_in_out_rescale")
use_in = variant in ("he_in", "he_in_out", "he_in_out_rescale")
use_out = variant in ("he_in_out", "he_in_out_rescale")
use_rescale = variant == "he_in_out_rescale"

def unit_where_constant(std):
    return np.where(std > 0, std, 1.0).astype(np.float32)

train, test, query = case.split.train, case.split.test, case.query
branch = (train.f.cpu().double() * (1 + shift)).numpy().astype(np.float32)
trunk = query.cpu().double().numpy().astype(np.float32)
y = train.targets(query).cpu().numpy().astype(np.float32)
target = test.targets(query).cpu().double().contiguous()
test_f = test.f.cpu().double()
dde, Net = adapter._import_deepxde()
in_mean, in_std = branch.mean(0), unit_where_constant(branch.std(0))
out_mean, out_std = y.mean(0), unit_where_constant(y.std(0))
if use_in:
    branch = (branch - in_mean) / in_std
if use_out:
    y = (y - out_mean) / out_std

data = dde.data.TripleCartesianProd(X_train=(branch, trunk), y_train=y, X_test=(branch, trunk), y_test=y)
dde.config.set_random_seed(0)
width = 40
net = Net([branch.shape[1], width, width], [trunk.shape[1], width, width], "relu", "He normal" if use_he else "Glorot normal")
if use_rescale:
    factor = math.sqrt(8 / width)
    net.apply_output_transform(lambda inputs, outputs: outputs * factor)
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
start = time.perf_counter()
model.train(iterations=iterations, display_every=iterations, verbose=0)
seconds = time.perf_counter() - start

test_branch = (test_f * (1 + shift)).numpy().astype(np.float32)
if use_in:
    test_branch = (test_branch - in_mean) / in_std
pred = model.predict((test_branch, trunk))
if use_out:
    pred = pred * out_std + out_mean
pred = torch.as_tensor(pred, dtype=torch.float64, device="cpu")
metrics = get_metrics(pred.contiguous(), target)
print(json.dumps({"variant": variant, "dataset": dataset, "shift": shift, "iterations": iterations,
                  "r2": metrics["r2"], "rel_l2": metrics["mean_per_sample_relative_l2"], "rrse": metrics["rrse"],
                  "train_device": str(next(model.net.parameters()).device), "seconds": round(seconds, 1),
                  "constant_out_points": int((y.std(0) == 0).sum()) if not use_out else int((out_std == 1.0).sum())}))
