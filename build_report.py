"""Refresh the LaTeX tables from the measured CSVs and compile the report."""
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf" / "lab-4-online-machine-learning.pdf"
NAMES = {
    "TrumpApproval": "Trump Approval", "ChickWeights": "Chick Weights",
    "FriedmanShift": "Friedman shift", "BreastCancer": "Breast Cancer",
    "SEAShift": "SEA shift", "GaussianStable": "Stationary", "GaussianShift": "Moving",
    "PageHinkley": "Page–Hinkley", "HDDM_A": "HDDM-A",
}


def name(key):
    return NAMES.get(key, key)


def data(key):
    return pd.read_csv(ROOT / "results" / f"{key}.csv")


def tex(value):
    special = {"%": r"\%", "&": r"\&", "_": r"\_", "µ": r"\ensuremath{\mu}",
               "²": r"\ensuremath{^{2}}", "×": r"\ensuremath{\times}", "–": "--", "—": "---"}
    return "".join(special.get(char, char) for char in str(value))


def table(key, columns, headers, rows, separators=()):
    layout = r"@{\extracolsep{\fill}}" + columns + "@{}"
    latex = [
        rf"\expandafter\newcommand\csname reportdata@{key}\endcsname{{%",
        rf"  \begin{{tabular*}}{{\linewidth}}{{{layout}}}",
        r"  \toprule",
        "  " + " & ".join(rf"\textbf{{{tex(h)}}}" for h in headers) + r" \\",
        r"  \midrule",
    ]
    for i, row in enumerate(rows):
        if i in separators:
            latex.append(r"  \addlinespace")
        latex.append("  " + " & ".join(map(tex, row)) + r" \\")
    latex.extend([r"  \bottomrule", r"  \end{tabular*}%", "}"])
    return "\n".join(latex)


def supervised_table(task):
    is_reg = task == "regression"
    headers = ["MAE", "RMSE", "R²"] if is_reg else ["Accuracy", "Macro F1", "Kappa"]
    rows = []
    for i, r in enumerate(data(f"river_{task}").itertuples()):
        metrics = [f"{r.mae:.3f}", f"{r.rmse:.3f}", f"{r.r2:.3f}"] if is_reg else [
            f"{r.accuracy:.2%}", f"{r.macro_f1:.3f}", f"{r.kappa:.3f}",
        ]
        rows.append([name(r.stream) if i % 4 == 0 else "", r.model, *metrics,
                     f"{r.wall_s:.4f}", f"{r.mean_us:.1f}", f"{r.p95_us:.1f}"])
    return table(task, "llrrrrrr", ["Stream", "Learner", *headers, "Wall (s)", "Mean (µs)", "p95 (µs)"],
                 rows, (4, 8, 12))


def drift_tables(task):
    is_reg = task == "regression"
    score_fmt = ".3f" if is_reg else ".1%"
    metric = "MAE" if is_reg else "acc."
    runs = data(f"{task}_drift").fillna({"detector": "None"})
    streams = runs.stream.unique()
    synthetic = runs[runs.stream == streams[-1]]

    def pair(monitor, reset, field):
        return " / ".join("—" if pd.isna(r[field]) else f"{r[field]:.0f}" for r in (monitor, reset))

    rows = []
    for detector, group in synthetic.groupby("detector", sort=False):
        monitor, reset = group.iloc[0], group.iloc[-1]
        rows.append([name(detector) if detector != "None" else "HT baseline",
                     pair(monitor, reset, "alarms"), pair(monitor, reset, "pre_change_alarms"),
                     pair(monitor, reset, "delay") if detector != "None" else "—",
                     format(reset.first_1000_after, score_fmt), format(reset.score, score_fmt),
                     f"{reset.mean_us:.1f}"])

    real_rows = []
    for stream in streams[:-1]:
        subset = runs[runs.stream == stream]
        base = subset.iloc[0].score
        cells = []
        for _, group in subset.iloc[1:].groupby("detector", sort=False):
            monitor, reset = group.iloc[0], group.iloc[1]
            change = reset.score - base
            suffix = f"{change:+.2f}" if is_reg else f"{100 * change:+.1f} pp"
            cells.append(f"{monitor.alarms}/{reset.alarms} ({suffix})")
        real_rows.append([name(stream), *cells])
    real_heads = [name(d) for d in synthetic.detector.unique()[1:]]
    return (
        table(f"{task}Drift", "lrrrrrr", ["Detector", "All alarms", "Pre-shift", "Delay (rows)",
              f"Post-1k {metric}", f"All {metric}", "Mean (µs)"], rows),
        table(f"{task}Real", "l" + "r" * len(real_heads), ["Real stream", *real_heads], real_rows),
    )


def library_table():
    river = data("river_classification")
    capy = data("capymoa_classification")
    runs = river.merge(capy, on=["stream", "model"], suffixes=("_r", "_c"), validate="one_to_one")
    rows = []
    for i, r in enumerate(runs.itertuples()):
        rows.append([name(r.stream) if i % 4 == 0 else "", r.model,
                     f"{r.accuracy_r:.2%}", f"{r.accuracy_c:.2%}", f"{r.wall_s_r:.4f}",
                     f"{r.wall_s_c:.4f}", f"{r.mean_us_r:.1f}", f"{r.mean_us_c:.1f}"])
    gains = 100 * (runs.accuracy_c - runs.accuracy_r)
    speedups = runs.mean_us_r / runs.mean_us_c
    fastest = runs.loc[speedups.idxmax()]
    best, worst = runs.loc[gains.idxmax()], runs.loc[gains.idxmin()]
    findings = (
        f"CapyMOA is faster in {(speedups > 1).sum()} of {len(runs)} pairs. "
        f"Its largest speedup is {speedups.max():.1f}× for {fastest.model} on {name(fastest.stream)}. "
        f"Its largest accuracy gain is {gains.max():+.2f} pp ({best.model}, {name(best.stream)}), "
        f"and its largest loss is {gains.min():+.2f} pp ({worst.model}, {name(worst.stream)})."
    )
    memory = (
        f"Sampled process peak RSS was {river.rss_peak_mib.min():.0f}–{river.rss_peak_mib.max():.0f} MiB "
        f"across River's classification runs and {capy.rss_peak_mib.min():.0f}–"
        f"{capy.rss_peak_mib.max():.0f} MiB for CapyMOA."
    )
    return (
        table("libraries", "llrrrrrr", ["Stream", "Learner", "River acc.", "Capy acc.", "River (s)",
              "Capy (s)", "River (µs)", "Capy (µs)"], rows, (4, 8, 12)),
        rf"\newcommand{{\libraryfindings}}{{{tex(findings)}}}",
        rf"\newcommand{{\librarymemory}}{{{tex(memory)}}}",
    )


def clustering_table():
    rows = [[name(r.stream), r.model, f"{r.mean_ari:.3f}", r.scored_blocks,
             "—" if pd.isna(r.first_post_shift_ari) else f"{r.first_post_shift_ari:.3f}",
             f"{r.last_4_blocks_ari:.3f}", r.final_centers, f"{r.mean_train_us:.1f}"]
            for r in data("clustering").itertuples()]
    return table("clustering", "llrrrrrr",
                 ["Stream", "Model", "Mean ARI", "Blocks", "Shift ARI", "Last 4 ARI", "Centers",
                  "Train (µs)"], rows, (3,))


def automl_table():
    rows = [[r.model, f"{r.accuracy:.2%}", f"{r.first_1000_after:.1%}", f"{r.wall_s:.3f}",
             f"{r.mean_us:.1f}", f"{r.p95_us:.1f}"] for r in data("automl").itertuples()]
    return table("automl", "lrrrrr", ["Method", "Accuracy", "Post-1k acc.", "Wall (s)", "Mean (µs)",
                 "p95 (µs)"], rows)


def environment():
    env = data("environment").set_index("item").value
    java = env["Java"].split('"')[1]
    text = (f"{env['CPU']}, {env['RAM']} RAM, {env['Platform']}, Python {env['Python']}, "
            f"OpenJDK {java}, River {env['river']} and CapyMOA {env['capymoa']}.")
    return rf"\newcommand{{\labenvironment}}{{{tex(text)}}}"


def main():
    compiler = shutil.which("tectonic")
    if not compiler:
        raise SystemExit("Tectonic is required to compile report.tex.")
    tables = [supervised_table("regression"), *drift_tables("regression"),
              supervised_table("classification"), *drift_tables("classification"),
              *library_table(), clustering_table(), automl_table(), environment()]
    (ROOT / "report-tables.tex").write_text(
        "% Generated by build_report.py from results/*.csv.\n\n" + "\n\n".join(tables) + "\n",
        encoding="utf-8",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".latex-", dir=OUT.parent) as tmp:
        subprocess.run([compiler, "-X", "compile", "--outdir", tmp, "report.tex"], cwd=ROOT, check=True)
        compiled = Path(tmp) / "report.pdf"
        os.replace(compiled, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
