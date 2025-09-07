import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Literal, Any


def save_to_pgf(
    fname: str,
    verbosity: Literal["removed", "kept"] | None = None,
    plt: Figure | Any = plt,
):
    start_idx = fname.find(".pgf")
    assert start_idx >= 0, "invalid file name"
    # tmp_prefix = "tmp_"
    # plt.savefig(tmp_prefix + fname, transparent=True)
    plt.savefig(fname, transparent=True, bbox_inches="tight")
    with open(fname, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if not line.startswith("%%"):
                f.write(line)
        f.truncate()
