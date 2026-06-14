import random


def auroc(pos, neg) -> float:
    """Probability that a random positive scores above a random negative.
    Equivalent to the area under the ROC curve. 0.5 = no separation."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def cohens_d(a, b) -> float:
    """Standardized mean difference (mean(a) - mean(b)) / pooled_sd."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    pooled = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return (ma - mb) / pooled if pooled > 0 else float("nan")


def bootstrap_ci(values, stat=lambda v: sum(v) / len(v), n=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a statistic over `values`."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    samples = []
    for _ in range(n):
        resample = [values[rng.randrange(k)] for _ in range(k)]
        samples.append(stat(resample))
    samples.sort()
    lo = samples[int((alpha / 2) * n)]
    hi = samples[int((1 - alpha / 2) * n)]
    return (lo, hi)
