"""Regenerate the report figures from the measured CSVs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
ACCENT = "#006d74"
PALETTE = ("#006d74", "#b54919", "#6a4a9d", "#2568b2", "#916516")
SHORT = {
    "SeoulBikes": "Seoul bikes", "ApplianceEnergy": "Appliance energy", "MetroTraffic": "Metro traffic",
    "Occupancy": "Room occupancy", "EEGEyeState": "EEG eye state", "GasSensorDrift": "Gas sensor drift",
    "FriedmanShift": "Friedman shift", "SEAShift": "SEA shift",
    "OccupancySensors": "Room sensors", "GaussianShift": "Moving Gaussians",
    "PageHinkley": "Page–Hinkley", "HDDM_A": "HDDM-A",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.0, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#94a3b3",
    "axes.labelcolor": "#25374a", "text.color": "#17273a", "xtick.color": "#436074",
    "ytick.color": "#436074", "figure.facecolor": "white", "axes.grid": True,
    "grid.color": "#e6ecef", "grid.linewidth": .7, "axes.axisbelow": True,
    "legend.frameon": False,
})


def read(name):
    return pd.read_csv(ROOT / "results" / f"{name}.csv", keep_default_na=True)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=230, facecolor="white")
    plt.close(fig)


def facet_tradeoff(task):
    data = read(f"river_{task}")
    regression = task == "regression"
    metric = "mae" if regression else "accuracy"
    models = data.model.unique()
    limits = (data.mean_us.min() / 1.45, data.mean_us.max() * 1.45)
    low, high = np.floor(np.log10(limits)).astype(int)
    ticks = [
        m * 10. ** exponent
        for exponent in range(low, high + 1) for m in (1, 2, 5)
        if limits[0] <= m * 10. ** exponent <= limits[1]
    ]
    if len(ticks) > 5:
        ticks = ticks[::2]
    fig, axes = plt.subplots(1, 4, figsize=(8.1, 2.65), layout="constrained")
    for ax, (name, d) in zip(axes, data.groupby("stream", sort=False), strict=True):
        for model, color, marker in zip(models, PALETTE[:4], ("o", "s", "^", "D"), strict=True):
            row = d[d.model == model].iloc[0]
            ax.scatter(row.mean_us, row[metric], c=color, label=model, s=35,
                       marker=marker, edgecolors="white", linewidth=.6, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(*limits)
        ax.set_title(SHORT.get(name, name), fontsize=9.0, pad=7)
        ax.set_xlabel("Mean µs")
        ax.margins(y=.25)
        if not regression:
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{100*v:.1f}".rstrip("0").rstrip(".") + "%"
            ))
    axes[0].set_ylabel("MAE ↓" if regression else "Accuracy ↑")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4)
    save(fig, task)


def drift_figure(task):
    data = read(f"{task}_drift_blocks")
    synth = "FriedmanShift" if task == "regression" else "SEAShift"
    data = data[(data.stream == synth) & (data.policy != "monitor")].fillna({"detector": "None"})
    events = read(f"{task}_alarms")
    events = events[events.stream == synth]
    fig, (ax, raster) = plt.subplots(
        2, 1, sharex=True, figsize=(8.1, 3.12),
        gridspec_kw={"height_ratios": [2.3, 1.15]}, layout="constrained",
    )
    color_by_detector = {}
    for (detector, d), color in zip(data.groupby("detector", sort=False), ("#536477", *PALETTE)):
        label = "HT baseline" if detector == "None" else SHORT.get(detector, detector)
        ax.plot(d.end, d.score, color=color, label=label, linewidth=1.3,
                linestyle="--" if detector == "None" else "-")
        color_by_detector[detector] = color
    detectors = [d for d in color_by_detector if d != "None"]
    for idx, detector in enumerate(detectors):
        for policy, offset, marker in (("monitor", .14, "|"), ("reset", -.14, "o")):
            positions = events[(events.detector == detector) & (events.policy == policy)].position
            raster.scatter(positions, np.full(len(positions), idx + offset), c=color_by_detector[detector],
                           marker=marker, s=26 if policy == "monitor" else 9, alpha=.9)
    for axis in (ax, raster):
        axis.axvline(5001, color="#243246", linestyle=":", linewidth=1.1)
        axis.set_xlim(0, 10000)
    raster.set_yticks(range(len(detectors)), [SHORT.get(d, d) for d in detectors])
    raster.set_ylim(-.45, len(detectors) - .55)
    raster.invert_yaxis()
    raster.set_xlabel("Observation (1-based)")
    ax.set_ylabel("Block MAE" if task == "regression" else "Block accuracy")
    if task == "classification":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), fontsize=8.1)
    save(fig, f"{task}_drift")


def cross_library():
    river = read("river_classification")
    capy = read("capymoa_classification")
    data = river.merge(capy, on=["stream", "model"], suffixes=("_river", "_capy"))
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(8.1, 2.68), layout="constrained")
    for ax, (name, d) in zip(axes, data.groupby("stream", sort=False), strict=True):
        ax.axvline(1, color="#52677b", linestyle=":", linewidth=1)
        ax.axhline(0, color="#52677b", linestyle=":", linewidth=1)
        for row, color, marker in zip(d.itertuples(), PALETTE[:4], ("o", "s", "^", "D"), strict=True):
            ax.scatter(row.mean_us_river / row.mean_us_capy,
                       100 * (row.accuracy_capy - row.accuracy_river),
                       color=color, marker=marker, s=34, label=row.model, zorder=3)
        ax.set_xscale("log")
        ax.set_title(SHORT.get(name, name), fontsize=9)
    axes[0].set_ylabel("CapyMOA − River (pp)")
    fig.supxlabel("River / CapyMOA mean latency (>1 favors CapyMOA)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4)
    save(fig, "cross_library")


def clustering():
    data = read("clustering_blocks")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8.1, 2.45), layout="constrained")
    for ax, (name, d) in zip(axes, data.groupby("stream", sort=False), strict=True):
        for (model, rows), color in zip(d.groupby("model", sort=False), PALETTE):
            ax.plot(rows.start, rows.ari, color=color, label=model, marker=".", markersize=3)
        if name == "GaussianShift":
            ax.axvline(3001, color="#243246", linestyle=":", linewidth=1.1)
        ax.set_title(SHORT[name], fontsize=9)
        ax.set_xlabel("Scored block start")
        ax.set_ylim(min(-.04, float(data.ari.min()) - .03), 1.04)
    axes[0].set_ylabel("Adjusted Rand index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    save(fig, "clustering")

    data = read("clustering_centers")
    moving = data[(data.stream == "GaussianShift") & data.end.isin([3000, 3500, 6000])]
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.35), sharex=True, sharey=True, layout="constrained")
    for ax, (end, snapshots) in zip(axes, moving.groupby("end"), strict=True):
        for (model, d), color, marker in zip(
            snapshots.groupby("model", sort=False), PALETTE, ("o", "^", "+"),
        ):
            ax.scatter(d.x1, d.x2, label=model, color=color, marker=marker, s=19, alpha=.85)
        ax.set_title(f"After {end:,} observations", fontsize=8.7)
        ax.set_xlabel("Feature 1")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Feature 2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    save(fig, "cluster_centers")


def automl():
    data = read("automl_blocks")
    fig, ax = plt.subplots(figsize=(8.1, 2.65), layout="constrained")
    for i, ((model, rows), color) in enumerate(zip(data.groupby("model", sort=False), PALETTE, strict=True)):
        ax.plot(rows.end, rows.score, label=model, color=color, linewidth=1.45,
                linestyle="--" if i < 2 else "-")
    ax.axvline(5001, color="#243246", linestyle=":", linewidth=1.1)
    ax.set(xlabel="Observation (1-based)", ylabel="Block accuracy", xlim=(0, 10000))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=8)
    save(fig, "automl")


if __name__ == "__main__":
    for task in ("regression", "classification"):
        facet_tradeoff(task)
        drift_figure(task)
    cross_library()
    clustering()
    automl()
