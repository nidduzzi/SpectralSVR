import collections
import pathlib
import statistics
import sys

ROOT = pathlib.Path.home() / ".cache" / "spectralsvr-mutation"
SLOWEST_SHOWN = 10


def read(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        stamp, tree, elapsed, status, mutated = line.split("\t", 4)
        module = (
            mutated.split("+++ b/")[1].split(" ")[0] if "+++ b/" in mutated else "?"
        )
        rows.append(
            {
                "stamp": stamp,
                "tree": tree,
                "elapsed": float(elapsed),
                "status": int(status),
                "module": module,
                "mutated": mutated,
            }
        )
    return rows


def summarize(rows: list[dict]) -> None:
    by_group = collections.defaultdict(list)
    for row in rows:
        by_group[(row["tree"], row["module"])].append(row)
    for (tree, module), group in sorted(by_group.items()):
        times = sorted(row["elapsed"] for row in group)
        killed = sum(row["status"] != 0 for row in group)
        print(
            f"{tree} {module}: {len(group)} runs, {sum(times) / 3600:.2f} h, "
            f"median {statistics.median(times):.1f} s, p95 {times[int(0.95 * (len(times) - 1))]:.1f} s, "
            f"max {times[-1]:.1f} s, killed {killed}"
        )
    print(f"\nslowest {SLOWEST_SHOWN} runs")
    for row in sorted(rows, key=lambda r: -r["elapsed"])[:SLOWEST_SHOWN]:
        print(
            f"  {row['elapsed']:7.1f} s  exit {row['status']:>3}  {row['tree']}  {row['mutated'][:150]}"
        )
    outcomes = collections.Counter(row["status"] for row in rows)
    print(f"\nexit statuses {dict(outcomes)}")


if __name__ == "__main__":
    summarize(
        read(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "timings.tsv")
    )
