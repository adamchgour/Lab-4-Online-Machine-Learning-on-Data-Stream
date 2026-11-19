"""CapyMOA clustering and online model selection."""
import random
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from capymoa.automl import AutoClass, BanditClassifier, EpsilonGreedy, SuccessiveHalvingClassifier
from capymoa.classifier import HoeffdingAdaptiveTree, HoeffdingTree
from capymoa.cluster import ClusTree, Clustream_with_kmeans, Denstream_with_dbscan
from capymoa.core import LabeledInstance
from capymoa.stream import Schema
from sklearn.metrics import adjusted_rand_score

from benchmark import classification_scores, measure, score_blocks, write_table
from capy_models import capy_rows, classification_schema
from streams import SEED, SWITCH, clustering_rows, load_rows

CLUSTER_BLOCK = 250
CLUSTER_SWITCH = 3001


def get_centers(model):
    # The wrapper drops the final coordinate unconditionally and calls the
    # Java snapshot repeatedly. Fetch one snapshot and check its dimensionality.
    snapshot = (model.moa_learner.getClusteringResult() if model.implements_macro_clusters()
                else model.moa_learner.getMicroClusteringResult())
    centers = []
    if snapshot is not None:
        for item in snapshot.getClustering():
            center = np.array(item.getCenter())
            if len(center) not in (2, 3):
                raise ValueError(f"Unexpected cluster center dimensions: {len(center)}")
            if np.isfinite(center[:2]).all():
                centers.append(center[:2])
    return np.array(centers).reshape(-1, 2)


def run_clustering():
    schema = Schema.from_custom(
        ["x1", "x2", "dummy"], target="dummy",
        categories={"dummy": ["unused"]}, name="GaussianGroups",
    )
    methods = {
        "CluStream": lambda: Clustream_with_kmeans(
            schema, time_window=750, max_num_kernels=30, k_option=3,
        ),
        "DenStream": lambda: Denstream_with_dbscan(
            schema, horizon=750, epsilon=.55, beta=.2, mu=2, init_points=300,
            lambda_option=.25, speed=100,
        ),
        "ClusTree": lambda: ClusTree(schema, horizon=750, max_height=5),
    }
    summary, blocks, snapshots = [], [], []
    for moving in (False, True):
        x, hidden = clustering_rows(moving)
        stream = "GaussianShift" if moving else "GaussianStable"
        # Constant dummy class makes the Java header valid without revealing hidden labels.
        rows = [LabeledInstance.from_array(schema, point, 0) for point in x]
        for name, factory in methods.items():
            print(f"CapyMOA clustering {stream} {name}", flush=True)
            model = factory()
            identifiers = {"stream": stream, "model": name}
            fit_ns = np.empty(len(x))
            query_ns, ari_history = [], []
            centers = np.empty((0, 2))
            start = perf_counter_ns()
            for offset in range(0, len(x), CLUSTER_BLOCK):
                end = offset + CLUSTER_BLOCK
                ari, mean_distance = np.nan, np.nan
                # A snapshot from the previous block is used before any update on this block.
                if len(centers):
                    dist = np.linalg.norm(x[offset:end, None] - centers, axis=2)
                    nearest = dist.argmin(axis=1)
                    ari = adjusted_rand_score(hidden[offset:end], nearest)
                    mean_distance = dist.min(axis=1).mean()
                blocks.append(identifiers | {
                    "start": offset + 1, "end": end, "centers": len(centers),
                    "ari": ari, "mean_distance": mean_distance,
                })
                ari_history.append(ari)
                for i in range(offset, end):
                    tick = perf_counter_ns()
                    model.train(rows[i])
                    fit_ns[i] = perf_counter_ns() - tick
                tick = perf_counter_ns()
                centers = get_centers(model)
                query_ns.append(perf_counter_ns() - tick)
                snapshots.extend(identifiers | {"end": end, "x1": a, "x2": b} for a, b in centers)
            elapsed = (perf_counter_ns() - start) / 1e9
            summary.append(identifiers | {
                "n": len(x), "mean_ari": np.nanmean(ari_history),
                "scored_blocks": int(np.isfinite(ari_history).sum()),
                "first_post_shift_ari": ari_history[3000 // CLUSTER_BLOCK] if moving else "",
                "last_4_blocks_ari": np.nanmean(ari_history[-4:]), "final_centers": len(centers),
                "wall_s": elapsed, "mean_train_us": fit_ns.mean() / 1e3,
                "p95_train_us": np.quantile(fit_ns, .95) / 1e3,
                "mean_snapshot_ms": np.mean(query_ns) / 1e6,
            })
    write_table("clustering", summary)
    write_table("clustering_blocks", blocks)
    write_table("clustering_centers", snapshots)


def run_automl():
    stream = "SEAShift"
    rows = load_rows(stream)
    y = np.array([label for _, label in rows])
    schema = classification_schema(rows, stream)
    feature_names = tuple(rows[0][0])
    classes = (HoeffdingTree, HoeffdingAdaptiveTree)

    def candidates():
        return [cls(schema, random_seed=SEED, grace_period=grace)
                for cls in classes for grace in (50, 100, 200)]

    methods = {
        "HT (g=100)": lambda: HoeffdingTree(schema, random_seed=SEED, grace_period=100),
        "HAT (g=100)": lambda: HoeffdingAdaptiveTree(schema, random_seed=SEED, grace_period=100),
        "Bandit": lambda: BanditClassifier(
            schema=schema, random_seed=SEED, base_classifiers=candidates(),
            policy=EpsilonGreedy(epsilon=.15, burn_in=300),
        ),
        "Successive halving": lambda: SuccessiveHalvingClassifier(
            schema=schema, random_seed=SEED, base_classifiers=candidates(),
            budget=10_000, eta=2., min_models=1,
        ),
        "AutoClass": lambda: AutoClass(
            schema=schema, random_seed=SEED,
            configuration_json=str(Path(__file__).with_name("autoclass.json")),
            base_classifiers=list(classes),
        ),
    }
    summary, blocks = [], []
    for name, factory in methods.items():
        print(f"CapyMOA AutoML {name}", flush=True)
        random.seed(SEED)
        pred, cost, _ = measure(
            capy_rows(rows, schema, feature_names), factory, "classification", capy=True,
        )
        identifiers = {"stream": stream, "model": name}
        summary.append(identifiers | classification_scores(y, pred) | {
            "n": len(y),
            "first_1000_after": np.mean(y[SWITCH - 1:SWITCH + 999] == pred[SWITCH - 1:SWITCH + 999]),
        } | cost)
        blocks.extend(score_blocks(y, pred, "classification", identifiers))
    write_table("automl", summary)
    write_table("automl_blocks", blocks)
