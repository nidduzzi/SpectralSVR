import matplotlib.pyplot as plt
from typing import Literal


def save_to_pgf(fname: str, verbosity: Literal["removed", "kept"] | None = None):
    start_idx = fname.find(".pgf")
    assert start_idx >= 0, "invalid file name"
    tmp_prefix = "tmp_"
    # raw_fname = fname.split(".")[0]
    plt.savefig(tmp_prefix + fname)
    fr = open(tmp_prefix + fname, "r")
    fw = open(fname, "w")
    for line in fr.readlines():
        # reading all lines except comments
        if not (line.startswith("%%")):
            # printing those lines
            # storing only those lines that
            # do not begin with "TextGenerator"
            fw.write(line)
            if verbosity == "kept":
                print(line)
        elif verbosity == "removed":
            print(line)

    # close and save the files
    fw.close()
    fr.close()
