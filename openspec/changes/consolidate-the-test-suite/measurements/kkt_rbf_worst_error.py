import torch

from SpectralSVR import LSSVR

ATOL, RTOL = 1e-5, 1e-4
CASES = 2000


def gram(kernel, x, z, sigma):
    if kernel == "linear":
        return x @ z.T
    return torch.exp(-torch.cdist(x, z) ** 2 / (2 * sigma) ** 2)


def worst_ratio(kernel, n, d, o, c, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, o, generator=g, dtype=torch.float64)
    xq = torch.randn(7, d, generator=g, dtype=torch.float64)
    a = torch.zeros(n + 1, n + 1, dtype=torch.float64)
    a[0, 1:] = 1
    a[1:, 0] = 1
    a[1:, 1:] = gram(kernel, x, x, sigma) + torch.eye(n, dtype=torch.float64) / c
    rhs = torch.cat([torch.zeros(1, o, dtype=torch.float64), y])
    solution = torch.linalg.lstsq(a, rhs).solution
    expected = gram(kernel, xq, x, sigma) @ solution[1:] + solution[0]
    got = LSSVR(kernel=kernel, sigma=sigma, C=c, dtype=torch.float64).fit(x, y).predict(xq)
    scale = float(expected.abs().max().clamp(min=1.0))
    allowed = ATOL * scale + RTOL * expected.abs()
    return float(((got - expected).abs() / allowed).max())


if __name__ == "__main__":
    draw = torch.Generator().manual_seed(0)
    worst = {}
    for case in range(CASES):
        kernel = ("linear", "rbf")[int(torch.randint(2, (1,), generator=draw))]
        n = int(torch.randint(5, 21, (1,), generator=draw)) if case % 10 else 300
        sigma = float(0.5 + 2.5 * torch.rand(1, generator=draw))
        c = float(0.5 + 99.5 * torch.rand(1, generator=draw))
        d, o = int(torch.randint(1, 5, (1,), generator=draw)), int(torch.randint(1, 4, (1,), generator=draw))
        ratio = worst_ratio(kernel, n, d, o, c, sigma, case)
        key = (kernel, "300 rows" if n == 300 else "5-20 rows")
        worst[key] = max(worst.get(key, 0.0), ratio)
    for key, ratio in sorted(worst.items()):
        print(key, f"worst error as a fraction of the allowed {ratio:.3g}")
