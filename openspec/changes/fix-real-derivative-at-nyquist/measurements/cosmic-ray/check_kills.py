import subprocess
p = "SpectralSVR/basis/FourierBasis.py"
original = open(p).read()
signature = "modes: int, length: float, ord: int = 1, complex_funcs: bool = False"
switch = "res.step * mode,"
mutants = [(signature, signature.replace("ord: int = 1", "ord: int = 2")),
           (signature, signature.replace("ord: int = 1", "ord: int = 0")),
           (signature, signature.replace("= False", "= True"))]
mutants += [(switch, f"res.step {op} mode,") for op in ("+", "-", "//", "<<", "%", "|", "/", "^")]
mutants += [(switch, "res.step * (mode + 1 - mode % 2),")]
try:
    for old, new in mutants:
        assert original.count(old) == 1, old
        open(p, "w").write(original.replace(old, new))
        done = subprocess.run(["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--hypothesis-seed=0", "-n", "6",
                               "tests/basis/test_nufft.py", "tests/basis/test_fourier_basis.py"], capture_output=True, text=True)
        last = [l for l in done.stdout.splitlines() if l.strip()][-1]
        print("KILLED" if done.returncode else "SURVIVED", "|", new, "|", last, flush=True)
finally:
    open(p, "w").write(original)
