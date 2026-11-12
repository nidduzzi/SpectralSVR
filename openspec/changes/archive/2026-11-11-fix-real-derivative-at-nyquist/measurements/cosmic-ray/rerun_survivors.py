import json, os, sqlite3, subprocess, sys, time
R = os.path.expanduser("~/.cache/spectralsvr-mutation")
W = f"{R}/nyquist-parent"
MODULES = {"fourier": "SpectralSVR/basis/FourierBasis.py", "base": "SpectralSVR/basis/__base.py"}
PYTEST = [f"{W}/.venv/bin/python", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--hypothesis-seed=0", "-n", "6", "tests"]

def pristine(path):
    return subprocess.run(["git", "-C", W, "show", f"HEAD:{path}"], capture_output=True, check=True).stdout

def run_suite():
    start = time.time()
    done = subprocess.run(PYTEST, cwd=W, capture_output=True, text=True)
    tail = [l for l in done.stdout.splitlines() if l.strip()][-1:]
    failed = [l for l in done.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")][:2]
    return done.returncode, round(time.time() - start), tail + failed

code, seconds, tail = run_suite()
print(json.dumps({"baseline": code, "seconds": seconds, "tail": tail}), flush=True)
if code:
    sys.exit("baseline fails")
for name, path in MODULES.items():
    original = pristine(path)
    db = sqlite3.connect(f"{R}/nyq-{name}.sqlite")
    rows = db.execute("select s.job_id, s.start_pos_row, s.operator_name, s.occurrence from mutation_specs s join work_results w on s.job_id=w.job_id where w.test_outcome='SURVIVED' order by s.start_pos_row, s.start_pos_col").fetchall()
    for job, line, operator, occurrence in rows:
        try:
            subprocess.run([f"{W}/.venv/bin/cosmic-ray", "apply", path, operator, str(occurrence)], cwd=W, check=True, capture_output=True)
            diff = subprocess.run(["git", "-C", W, "diff", "-U0", path], capture_output=True, text=True).stdout
            changed = [l for l in diff.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))]
            code, seconds, tail = run_suite()
        finally:
            with open(f"{W}/{path}", "wb") as handle:
                handle.write(original)
        print(json.dumps({"module": name, "line": line, "operator": operator, "occurrence": occurrence, "change": changed, "exit": code, "seconds": seconds, "tail": tail}), flush=True)
