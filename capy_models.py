"""CapyMOA adapters for the shared numeric classification streams."""
from math import isqrt

import numpy as np
from capymoa.base import MOAClassifier
from capymoa.classifier import HoeffdingTree, NaiveBayes, SGDClassifier
from capymoa.core import LabeledInstance
from capymoa.stream import Schema
from moa.classifiers.meta import AdaptiveRandomForest


def classification_schema(rows, name):
    return Schema.from_custom(
        features=[*rows[0][0], "class"], target="class",
        categories={"class": ["0", "1"]}, name=name,
    )


def capy_rows(rows, schema, feature_names):
    return [
        (LabeledInstance.from_array(schema, np.array([x[key] for key in feature_names]), y), y)
        for x, y in rows
    ]


def classification_models(schema, seed, n_features):
    return {
        "HT": lambda: HoeffdingTree(
            schema, random_seed=seed, grace_period=100, confidence=1e-5,
            tie_threshold=.05, leaf_prediction="MajorityClass", disable_prepruning=False,
        ),
        "ARF": lambda: MOAClassifier(
            moa_learner=AdaptiveRandomForest, schema=schema, random_seed=seed, CLI=(
                "-l (ARFHoeffdingTree -g 100 -c 1e-5 -t 0.05 -l MC) "
                f"-s 5 -o 1 -m {isqrt(n_features)} -a 6 "
                "-x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.01) -j 1"
            ),
        ),
        "Logistic": lambda: SGDClassifier(
            schema, loss="log_loss", learning_rate="invscaling", eta0=.05, random_seed=seed,
        ),
        "GaussianNB": lambda: NaiveBayes(schema, random_seed=seed),
    }
