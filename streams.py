"""Fixed, ordered streams shared across all experimental conditions."""
from itertools import islice

import numpy as np
from river import datasets, preprocessing
from sklearn.datasets import load_breast_cancer, load_diabetes

SEED = 42
SWITCH = 5001  # 1-based first changed observation
SYNTH_SIZE = 10_000
REGRESSION = ("TrumpApproval", "ChickWeights", "Diabetes", "FriedmanShift")
CLASSIFICATION = ("Elec2", "Phishing", "BreastCancer", "SEAShift")


def friedman():
    rng = np.random.default_rng(SEED)
    for i, x in enumerate(rng.random((SYNTH_SIZE, 10)), start=1):
        a, b, c, d, e = x[:5] if i < SWITCH else x[5:]
        y = 10 * np.sin(np.pi * a * b) + 20 * (c - .5) ** 2 + 10 * d + 5 * e
        y += (12 if i >= SWITCH else 0) + rng.normal()
        yield dict(enumerate(x)), y


def sea():
    rng = np.random.default_rng(SEED)
    for i, x in enumerate(rng.uniform(0, 10, size=(SYNTH_SIZE, 3)), start=1):
        threshold = 8 if i < SWITCH else 13
        y = int(x[0] + x[1] > threshold)
        if rng.random() < .05:
            y ^= 1
        yield dict(enumerate(x)), y


def sklearn_rows(data):
    for x, y in zip(data.data, data.target, strict=True):
        yield dict(enumerate(x)), y.item()


def load_rows(name):
    sources = {
        "TrumpApproval": datasets.TrumpApproval,
        "ChickWeights": datasets.ChickWeights,
        # Use raw diabetes features: sklearn's scaled=True uses full-dataset statistics.
        "Diabetes": lambda: sklearn_rows(load_diabetes(scaled=False)),
        "Elec2": lambda: islice(datasets.Elec2(), 15_000),
        "Phishing": datasets.Phishing,
        "BreastCancer": lambda: sklearn_rows(load_breast_cancer()),
        "FriedmanShift": friedman,
        "SEAShift": sea,
    }
    rows = [({str(k): float(v) for k, v in x.items()}, y) for x, y in sources[name]()]
    if name in CLASSIFICATION:
        rows = [(x, int(y)) for x, y in rows]
    return rows


def prior_scaled(rows):
    """Transform x using statistics from observations 1 through t-1."""
    scaler = preprocessing.StandardScaler()
    for x, y in rows:
        yield scaler.transform_one(x), y
        scaler.learn_one(x)


def clustering_rows(moving):
    """Three isotropic Gaussian groups. Shift every center at observation 3001."""
    rng = np.random.default_rng(SEED)
    centers = np.array([[0., 0.], [0., 3.], [3., 0.]])
    shift = np.array([[-2., 1.], [3., 1.], [1., -2.]])
    labels = rng.integers(0, len(centers), 6000)
    x = centers[labels] + rng.normal(0, .30, size=(len(labels), 2))
    if moving:
        x[3000:] += shift[labels[3000:]]
    return x, labels
