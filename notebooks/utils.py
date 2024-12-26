import matplotlib.pyplot as plt
from typing import Literal


def save_to_pgf(
    fname: str, verbosity: Literal["removed", "kept"] | None = None, plt=plt
):
    start_idx = fname.find(".pgf")
    assert start_idx >= 0, "invalid file name"
    tmp_prefix = "tmp_"
    plt.savefig(tmp_prefix + fname, transparent=True)
    plt.savefig(fname, transparent=True)
    with open(fname, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if not line.startswith("%%"):
                f.write(line)
        f.truncate()
