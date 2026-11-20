"""Test-then-train benchmarks. Run each stage in a fresh Python process."""
import argparse
import csv
import gc
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
import psutil
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from river_models import classification_models, detectors, regression_models
from streams import CLASSIFICATION, REGRESSION, SEED, SWITCH, load_rows, prior_scaled

RESULTS = Path(__file__).resolve().parent / "results"
PROCESS = psutil.Process()
REPEATS = 3
BLOCK = 250
RSS_INTERVAL = 250
DETECTOR_START = 101
DELAY_HORIZON = 1000


def write_table(name, rows):
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def regression_scores(y, pred):
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": np.sqrt(mean_squared_error(y, pred)),
        "r2": r2_score(y, pred),
    }


def classification_scores(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y, pred, labels=[0, 1]),
    }


def measure(rows, factory, task, detector=None, reset=False, scale=1., capy=False):
    """Time input-independent prediction, optional drift handling and update."""
    gc.collect()
    rss_start = PROCESS.memory_info().rss
    rss_peak = rss_start
    model = factory()
    predict, train = (model.predict, model.train) if capy else (model.predict_one, model.learn_one)
    latencies = np.empty(len(rows))
    predictions = np.empty(len(rows))
    alarms = []
    start = perf_counter_ns()

    for i, (x, y) in enumerate(rows, start=1):
        tick = perf_counter_ns()
        pred = predict(x)
        if pred is None:
            pred = 0

        if detector is not None and i >= DETECTOR_START:
            loss = min(1., abs(y - pred) / scale) if task == "regression" else int(y != pred)
            detector.update(loss)
            if detector.drift_detected:
                alarms.append(i)
                if reset:
                    model = factory()
                    predict, train = model.predict_one, model.learn_one

        if capy:
            train(x)
        else:
            train(x, y)
        latencies[i - 1] = (perf_counter_ns() - tick) / 1e3
        predictions[i - 1] = pred
        if i % RSS_INTERVAL == 0:
            rss_peak = max(rss_peak, PROCESS.memory_info().rss)

    elapsed = (perf_counter_ns() - start) / 1e9
    rss_peak = max(rss_peak, PROCESS.memory_info().rss)
    cost = {
        "wall_s": elapsed,
        "mean_us": latencies.mean(),
        "p95_us": np.quantile(latencies, .95),
        "rss_start_mib": rss_start / 2**20,
        "rss_peak_mib": rss_peak / 2**20,
        "rss_growth_mib": (rss_peak - rss_start) / 2**20,
    }
    return predictions, cost, alarms


def score_blocks(y, pred, task, identifiers):
    loss = abs(y - pred) if task == "regression" else (y == pred)
    return [
        identifiers | {"end": min(i + BLOCK, len(y)), "score": loss[i:i + BLOCK].mean()}
        for i in range(0, len(y), BLOCK)
    ]


def evaluate(task, library):
    names = REGRESSION if task == "regression" else CLASSIFICATION
    scores = regression_scores if task == "regression" else classification_scores
    summary, timings, blocks = [], [], []
    drift_summary, drift_events, drift_blocks = [], [], []
    capy = library == "CapyMOA"

    for name in names:
        rows = load_rows(name)
        y = np.array([target for _, target in rows])
        scaled = list(prior_scaled(rows))
        print(f"{library} {task} {name}, n={len(rows)}", flush=True)
        if capy:
            from capy_models import capy_rows, classification_models as capy_models, classification_schema
            feature_names = tuple(rows[0][0])
            schema = classification_schema(rows, name)
            factories = capy_models(schema, SEED, len(feature_names))
        elif task == "regression":
            factories = regression_models(SEED)
        else:
            factories = classification_models(SEED)

        for model, factory in factories.items():
            scale_features = model in {"AdaGrad", "kNN", "Logistic"}
            inputs = scaled if scale_features else rows
            if capy:
                # Discarded model warms the Java/Python path. Each measured model starts empty.
                measure(capy_rows(inputs[:256], schema, feature_names), factory, task, capy=True)

            identifiers = {"library": library, "stream": name, "model": model}
            costs = []
            for repeat in range(1, REPEATS + 1):
                prepared = capy_rows(inputs, schema, feature_names) if capy else inputs
                pred, cost, _ = measure(prepared, factory, task, capy=capy)
                costs.append(cost)
                timings.append(identifiers | {"repeat": repeat} | cost)
                if repeat == 1:
                    first_predictions = pred
                elif not np.array_equal(pred, first_predictions):
                    raise RuntimeError(f"Non-deterministic predictions: {identifiers}")
            typical = {key: median(run[key] for run in costs) for key in costs[0]}
            summary.append(identifiers | {"n": len(y)} | scores(y, pred) | typical)
            blocks.extend(score_blocks(y, pred, task, identifiers))
            if model == "HT" and not capy:
                analyze_drift(
                    task, name, inputs, y, pred, factory, typical,
                    drift_summary, drift_events, drift_blocks,
                )

    prefix = f"{library.lower()}_{task}"
    write_table(prefix, summary)
    write_table(f"{prefix}_timings", timings)
    write_table(f"{prefix}_blocks", blocks)
    if not capy:
        write_table(f"{task}_drift", drift_summary)
        write_table(f"{task}_alarms", drift_events)
        write_table(f"{task}_drift_blocks", drift_blocks)


def analyze_drift(task, name, inputs, y, baseline, factory, typical, summary, events, blocks):
    scores = regression_scores if task == "regression" else classification_scores
    main_metric = "mae" if task == "regression" else "accuracy"
    scale = max(3 * np.std(y[:100]), 1e-8)
    has_switch = name.endswith("Shift")
    post = slice(SWITCH - 1, SWITCH - 1 + DELAY_HORIZON)

    conditions = [("None", "baseline", baseline, typical, [])]
    for detector, make_detector in detectors(task).items():
        for policy in ("monitor", "reset"):
            pred, cost, alarms = measure(
                inputs, factory, task, detector=make_detector(),
                reset=(policy == "reset"), scale=scale,
            )
            if policy == "monitor" and not np.array_equal(pred, baseline):
                raise RuntimeError("Monitor-only condition changed model predictions.")
            conditions.append((detector, policy, pred, cost, alarms))

    for detector, policy, pred, cost, alarms in conditions:
        timely = [p for p in alarms if SWITCH <= p < SWITCH + DELAY_HORIZON] if has_switch else []
        identifiers = {"stream": name, "detector": detector, "policy": policy}
        summary.append(identifiers | {
            "n": len(y), "alarms": len(alarms),
            "pre_change_alarms": sum(p < SWITCH for p in alarms) if has_switch else "",
            "delay": timely[0] - SWITCH if timely else "",
            "first_1000_after": scores(y[post], pred[post])[main_metric] if has_switch else "",
            "score": scores(y, pred)[main_metric],
            "wall_s": cost["wall_s"], "mean_us": cost["mean_us"], "p95_us": cost["p95_us"],
        })
        events.extend(identifiers | {"position": p} for p in alarms)
        blocks.extend(score_blocks(y, pred, task, identifiers))


def save_environment():
    java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    cpu = platform.processor() or platform.machine()
    ram = "unavailable"
    if platform.system() == "Darwin":
        hardware = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, check=False,
        )
        for line in hardware.stdout.splitlines():
            key, _, value = line.strip().partition(": ")
            if key == "Chip":
                cpu = value
            elif key == "Memory":
                ram = value
    else:
        ram = f"{psutil.virtual_memory().total / 2**30:.1f} GiB"
    write_table("environment", [
        {"item": "CPU", "value": cpu},
        {"item": "RAM", "value": ram},
        {"item": "Python", "value": platform.python_version()},
        {"item": "Platform", "value": f"{platform.system()} {platform.mac_ver()[0]} {platform.machine()}"},
        {"item": "Java", "value": java.stderr.splitlines()[0]},
        *({"item": module, "value": version(module)}
          for module in ("river", "capymoa", "numpy", "pandas", "scikit-learn", "matplotlib", "jpype1", "scipy", "psutil")),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["regression", "classification", "capymoa", "clustering", "automl", "environment"])
    stage = parser.parse_args().stage
    if stage in ("regression", "classification"):
        evaluate(stage, "River")
    elif stage == "capymoa":
        evaluate("classification", "CapyMOA")
    elif stage == "environment":
        save_environment()
    else:
        from unsupervised_automl import run_automl, run_clustering
        (run_clustering if stage == "clustering" else run_automl)()
