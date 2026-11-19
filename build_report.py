"""Compose the nine-page PDF from the archived results and figures."""
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf" / "lab-4-online-machine-learning.pdf"
REPOSITORY = "https://github.com/adamchgour/Lab-4-Online-Machine-Learning-on-Data-Stream"
PAGE_W, PAGE_H = A4
MARGIN = 43
WIDTH = PAGE_W - 2 * MARGIN - 12
INK = colors.HexColor("#182b3c")
MUTED = colors.HexColor("#4e6576")
TEAL = colors.HexColor("#006d74")
RULE = colors.HexColor("#d7e3e8")
PALE = colors.HexColor("#edf5f6")

FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, suffix in (("DejaVu", ""), ("DejaVu-Bold", "-Bold"), ("DejaVu-Oblique", "-Oblique")):
    pdfmetrics.registerFont(TTFont(name, str(FONT_PATH / f"DejaVuSans{suffix}.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique")

STYLES = {
    "eyebrow": ParagraphStyle("eyebrow", fontName="DejaVu-Bold", fontSize=9, leading=12,
                              textColor=TEAL, spaceAfter=7),
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=28.5, leading=33,
                            textColor=INK, spaceAfter=11),
    "section": ParagraphStyle("section", fontName="DejaVu-Bold", fontSize=23, leading=27,
                              textColor=INK, spaceAfter=11),
    "subhead": ParagraphStyle("subhead", fontName="DejaVu-Bold", fontSize=11, leading=15,
                              textColor=TEAL, spaceBefore=5, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.15, leading=13.4,
                           textColor=INK, spaceAfter=7),
    "caption": ParagraphStyle("caption", fontName="DejaVu", fontSize=8.0, leading=10.8,
                              textColor=MUTED, spaceBefore=3, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=8.0, leading=10.1, textColor=INK),
    "num": ParagraphStyle("num", fontName="DejaVu", fontSize=7.8, leading=10.1,
                          alignment=TA_RIGHT, textColor=INK),
    "th": ParagraphStyle("th", fontName="DejaVu-Bold", fontSize=7.7, leading=10.1,
                         textColor=colors.white, alignment=TA_CENTER),
}

DISPLAY = {
    "TrumpApproval": "Trump Approval", "ChickWeights": "Chick Weights",
    "FriedmanShift": "Friedman shift", "BreastCancer": "Breast Cancer",
    "SEAShift": "SEA shift", "GaussianStable": "Stationary", "GaussianShift": "Moving",
    "PageHinkley": "Page–Hinkley", "HDDM_A": "HDDM-A",
}


def name(key):
    return DISPLAY.get(key, key)


def data(key):
    return pd.read_csv(ROOT / "results" / f"{key}.csv")


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def table(headers, rows, fractions, text_cols=(0,), separators=()):
    cells = [[p(escape(str(value)), "th") for value in headers]]
    cells.extend([p(escape(str(value)), "cell" if j in text_cols else "num")
                  for j, value in enumerate(row)] for row in rows)
    t = Table(cells, colWidths=[WIDTH * x for x in fractions], hAlign="LEFT", repeatRows=1)
    styles = [
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, -1), (-1, -1), .45, RULE),
    ]
    styles.extend(("LINEABOVE", (0, i), (-1, i), .65, RULE) for i in separators)
    t.setStyle(TableStyle(styles))
    return t


def pic(key, ratio):
    chart = Image(str(ROOT / "figures" / f"{key}.png"), width=WIDTH, height=WIDTH * ratio)
    chart.hAlign = "LEFT"
    return chart


def section(part, title):
    return [p(part, "eyebrow"), p(title, "section")]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN + 5, PAGE_H - 32, PAGE_W - MARGIN - 5, PAGE_H - 32)
    canvas.line(MARGIN + 5, 39, PAGE_W - MARGIN - 5, 39)
    canvas.setFillColor(MUTED)
    canvas.setFont("DejaVu", 8)
    canvas.drawString(MARGIN + 5, PAGE_H - 24, "M2 DATA SCIENCE  /  DATA STREAM PROCESSING")
    canvas.drawString(MARGIN + 5, 25, "ADAM CHGOUR  /  LAB 4")
    canvas.drawRightString(PAGE_W - MARGIN - 5, 25, f"{doc.page} / 9")
    canvas.restoreState()


def setup():
    env = data("environment").set_index("item").value
    rows = [
        ("Regression", "Trump Approval", "1,001", 6, "Approval estimate from poll ratings and date"),
        ("Regression", "Chick Weights", "578", 3, "Chick weight, ordered by time then chick"),
        ("Regression", "Diabetes", "442", 10, "Disease progression, source row order"),
        ("Regression", "Friedman shift", "10,000", 10, "Abrupt change in active features and mean"),
        ("Classification", "Elec2", "15,000", 8, "Price direction, first 15,000 ordered rows"),
        ("Classification", "Phishing", "1,250", 9, "Phishing indicator, source row order"),
        ("Classification", "Breast Cancer", "569", 30, "Benign or malignant, source row order"),
        ("Classification", "SEA shift", "10,000", 3, "Abrupt threshold change with label noise"),
    ]
    java = env["Java"].split('"')[1]
    return [
        p("INSTITUT POLYTECHNIQUE DE PARIS", "eyebrow"),
        p("Online machine learning<br/>with River and CapyMOA", "title"),
        p("<b>Lab 4</b>  /  Adam Chgour  /  19 November 2026"),
        p("This study compares incremental learners and drift responses using ordered data streams. "
          "River covers regression, classification and drift detection. CapyMOA provides a second "
          "implementation for classification and extends the study to unsupervised clustering and AutoML."),
        p("Streams", "subhead"),
        table(["Task", "Stream", "Rows", "Features", "Target and ordering"], rows,
              [.155, .175, .103, .105, .462], text_cols=(0, 1, 4), separators=(5,)),
        p("Six real streams and two seeded synthetic supervised streams. River supplies Trump Approval, "
          "Chick Weights, Elec2 and Phishing. Scikit-learn supplies unscaled Diabetes and Breast Cancer.", "caption"),
        p("Evaluation protocol", "subhead"),
        p("Predict with the state available at t−1, score against the observed target, then train on row t. "
          "All rows count, with 0 used when a learner abstains at startup. Preserve source order and use "
          "seed 42 wherever configurable. For linear learners and regression kNN, each feature vector "
          "is standardized using only past rows. Other learners receive the same raw numeric features."),
        p("Cost and environment", "subhead"),
        p(f"{env['CPU']}, {env['RAM']} RAM, {env['Platform']}, Python {env['Python']}, OpenJDK {java}, "
          f"River {env['river']} and CapyMOA {env['capymoa']}. Supervised baseline costs are componentwise "
          "medians of three sequential replays with fresh learners. CapyMOA gets a discarded 256-row "
          "warm-up per pair. Mean and p95 are per-row predict-plus-update latency. Wall time includes "
          "loop and instrumentation overhead. Imports, loading, feature preparation, learner setup and "
          "final metric aggregation are outside the timed loop."),
        p("Source code, exact configurations, notebook and raw results are in the "
          f'<link href="{REPOSITORY}" color="#006d74"><u>GitHub repository</u></link>.'),
    ]


def supervised(task):
    is_reg = task == "regression"
    runs = data(f"river_{task}")
    rows = []
    for i, r in enumerate(runs.itertuples()):
        metrics = [f"{r.mae:.3f}", f"{r.rmse:.3f}", f"{r.r2:.3f}"] if is_reg else [
            f"{r.accuracy:.2%}", f"{r.macro_f1:.3f}", f"{r.kappa:.3f}"]
        rows.append([name(r.stream) if i % 4 == 0 else "", r.model, *metrics,
                     f"{r.wall_s:.4f}", f"{r.mean_us:.1f}", f"{r.p95_us:.1f}"])
    title = "River regression" if is_reg else "River classification"
    part = "PART 1 / STEPS 2–3" if is_reg else "PART 1 / STEP 5"
    headers = ["MAE", "RMSE", "R²"] if is_reg else ["Accuracy", "Macro F1", "Kappa"]
    if is_reg:
        methods = ("<b>HT</b> grows a Hoeffding tree with mean leaves. <b>ARF</b> combines five resampled "
                   "trees with mean leaves, median aggregation and internal ADWIN-based replacement. "
                   "Both trees use a 100-observation grace period. <b>AdaGrad</b> fits a linear predictor "
                   "with initial coefficient rate 0.10 and intercept rate 0.05. <b>kNN</b> searches a "
                   "200-row window for five neighbors and averages their targets.")
        findings = ("On Friedman shift, ARF attained MAE 2.711 versus 2.846 for AdaGrad, "
                    "at about 36× the mean latency. AdaGrad also had the lowest Chick Weights MAE, "
                    "while kNN reduced RMSE from 37.63 to 35.36. On Diabetes and Trump Approval, kNN "
                    "had the lowest MAE but the highest mean latency. RMSE highlights the cost of early "
                    "large misses, especially for the Trump Approval linear fit.")
    else:
        methods = ("<b>HT</b> uses information gain, majority leaves, grace 100, split confidence 10⁻⁵ "
                   "and tie threshold 0.05. <b>ARF</b> uses five matching base trees with square-root "
                   "feature sampling and Poisson(6) resampling. <b>Logistic</b> fits log loss with "
                   "AdaGrad at 0.05 and intercept rate 0.05. <b>GaussianNB</b> maintains class priors "
                   "and per-class Gaussian feature statistics.")
        findings = ("Logistic had the highest accuracy on all three real datasets and the lowest mean "
                    "latency on every stream. On SEA shift, ARF improved accuracy by 0.49 percentage "
                    "points over Logistic at roughly 30× the mean latency. Macro F1 and Cohen’s kappa "
                    "gave the same model ranking on each of the four streams.")
    return [
        *section(part, title), p(methods),
        table(["Stream", "Learner", *headers, "Wall s", "Mean µs", "p95 µs"],
              rows, [.19, .15, .112, .112, .101, .11, .112, .113],
              text_cols=(0, 1), separators=(5, 9, 13)),
        p("Table 1. Regression errors and cost." if is_reg else "Table 3. Classification quality and cost.",
          "caption"),
        pic(task, 2.65 / 8.1),
        p(("Figure 1. MAE vs. mean cost." if is_reg else "Figure 3. Accuracy vs. mean cost.")
          + " Each panel has its own score scale. Latency uses a logarithmic axis.", "caption"),
        p(findings),
    ]


def drift_page(task):
    is_reg = task == "regression"
    runs = data(f"{task}_drift").fillna({"detector": "None"})
    streams = runs.stream.unique()
    synthetic = runs[runs.stream == streams[-1]]
    rows = []
    for detector, group in synthetic.groupby("detector", sort=False):
        reset = group.iloc[-1]
        monitor = group.iloc[0]
        delays = ("—" if pd.isna(monitor.delay) else f"{monitor.delay:.0f} / {reset.delay:.0f}")
        score_fmt = ".3f" if is_reg else ".1%"
        rows.append([
            name(detector) if detector != "None" else "HT baseline",
            f"{monitor.alarms} / {reset.alarms}", f"{monitor.pre_change_alarms:.0f} / {reset.pre_change_alarms:.0f}",
            delays, format(reset.first_1000_after, score_fmt), format(reset.score, score_fmt),
            f"{reset.mean_us:.1f}",
        ])
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
    if is_reg:
        intro = ("Friedman shift switches from the first five inputs to the last five and adds 12 to the "
                 "target at row 5,001. The detectors use absolute HT errors divided by three times the "
                 "target standard deviation from the first 100 rows and clipped at 1. ADWIN tests an "
                 "adaptive window mean (δ=0.002), Page–Hinkley accumulates upward deviations "
                 "(threshold 25, δ=0.005, minimum 30), and KSWIN compares empirical distributions "
                 "(α=0.005, window 100, reference size 30).")
        analysis = ("KSWIN’s 15-row alarm was earliest, with eight pre-change alarms in monitor-only "
                    "mode and about 5× the baseline HT mean latency under reset. Page–Hinkley sounded "
                    "once at +43 and reduced the first-1,000 MAE from 10.79 to 3.55. ADWIN fired later "
                    "in closed loop (+59 vs +27), illustrating how earlier resets alter future errors. "
                    "Diabetes shows the cost of a reset without a corresponding predictive benefit.")
    else:
        intro = ("SEA uses a threshold of 8 on x₁+x₂ through row 5,000 and 13 thereafter, with "
                 "5% independent label flips. Detectors observe the HT’s binary error. ADWIN compares "
                 "adaptive window means (δ=0.002), DDM tracks error-rate significance, EDDM monitors "
                 "distances between mistakes, and HDDM-A compares means using Hoeffding bounds. "
                 "DDM, EDDM and HDDM-A use River defaults.")
        analysis = ("HDDM-A detected SEA’s shift at +47 with no pre-change alarms and improved accuracy "
                    "from 77.86% to 80.72%. ADWIN lifted early recovery to 70.9% yet reduced overall "
                    "accuracy to 70.71%. On Elec2, every reset policy lost accuracy. EDDM and HDDM-A "
                    "generated over 160 closed-loop alarms each, consistent with resets inducing more "
                    "error changes. A detector’s sensitivity and a learner’s recovery need separate tests.")
    metric_name = "MAE" if is_reg else "accuracy"
    part = "PART 1 / STEP 4" if is_reg else "PART 1 / STEP 5"
    widths = [.25, *([.75 / len(real_heads)] * len(real_heads))]
    return [
        *section(part, f"{task.capitalize()} drift detection"), p(intro),
        p("From row 101, every detector runs in monitor-only and in reset mode. A reset replaces the "
          "tree after predicting the alarm row and trains the fresh tree on that row. Delay is measured "
          "from row 5,001 to the first alarm within the next 1,000 rows, including row 5,001."),
        table(["Detector", "All alarms", "Pre-shift", "Delay rows", f"Post-1k {metric_name}",
               f"All {metric_name}", "Mean µs"], rows,
              [.19, .115, .12, .13, .172, .152, .121]),
        p(("Table 2" if is_reg else "Table 4") + ". Synthetic results. Pairs show monitor / reset. "
          "Scores and latency use reset mode or the unmodified HT baseline.", "caption"),
        table(["Real stream", *real_heads], real_rows, widths, text_cols=tuple(range(len(widths)))),
        p("Real-stream counts show monitor/reset alarms and change from baseline under reset "
          f"({metric_name}). Alarm positions are in the CSVs. Real streams lack labeled change points.", "caption"),
        pic(f"{task}_drift", 3.12 / 8.1),
        p(("Figure 2" if is_reg else "Figure 4") + ". Synthetic blocks of 250 rows. Vertical line "
          "marks the true shift. Alarm ticks denote monitor mode and dots denote reset mode.", "caption"),
        p(analysis),
    ]


def library_page():
    river = data("river_classification")
    capy = data("capymoa_classification")
    runs = river.merge(capy, on=["stream", "model"], suffixes=("_r", "_c"))
    rows = []
    for i, r in enumerate(runs.itertuples()):
        rows.append([name(r.stream) if i % 4 == 0 else "", r.model,
                     f"{r.accuracy_r:.2%}", f"{r.accuracy_c:.2%}", f"{r.wall_s_r:.4f}",
                     f"{r.wall_s_c:.4f}", f"{r.mean_us_r:.1f}", f"{r.mean_us_c:.1f}"])
    gains = 100 * (runs.accuracy_c - runs.accuracy_r)
    ratios = runs.mean_us_r / runs.mean_us_c
    b = runs.loc[gains.idxmax()]
    worst = runs.loc[gains.idxmin()]
    best_speed = runs.loc[ratios.idxmax()]
    findings = (
        f"CapyMOA is faster in {(ratios > 1).sum()} of 16 pairs. Its largest speedup is "
        f"{ratios.max():.1f}× for {best_speed.model} on {name(best_speed.stream)}. "
        f"Its largest accuracy gain is {gains.max():+.2f} pp ({b.model}, {name(b.stream)}), "
        f"and its largest loss is {gains.min():+.2f} pp ({worst.model}, {name(worst.stream)}). "
        "Logistic compares AdaGrad in River with a log-loss scikit-learn SGD wrapper in CapyMOA "
        "(initial rate 0.05, inverse scaling and L2 by default). Its gap combines optimizer and wrapper effects."
    )
    memory = (
        f"Sampled process peak RSS was {river.rss_peak_mib.min():.0f}–{river.rss_peak_mib.max():.0f} MiB "
        f"across River’s classification runs and {capy.rss_peak_mib.min():.0f}–"
        f"{capy.rss_peak_mib.max():.0f} MiB for CapyMOA. These process measurements include input buffers, "
        "retained allocations and the JVM where applicable. Sampling every 250 rows can miss transient peaks."
    )
    return [
        *section("PART 2 / STEPS 7–8", "River versus CapyMOA"),
        p("Four pairs use the same ordered inputs, scoring, seed and learner family. HT and ARF align "
          "the split settings and majority leaves. ARF also aligns five trees, square-root feature count, "
          "Poisson(6) bagging, ADWIN drifts (δ=0.001) and warnings (δ=0.01) with one worker. "
          "Numeric observers, tie handling and implementation details can still differ."),
        table(["Stream", "Learner pair", "River acc.", "Capy acc.", "River wall s", "Capy wall s",
               "River µs", "Capy µs"], rows,
              [.178, .155, .111, .111, .111, .111, .111, .112],
              text_cols=(0, 1), separators=(5, 9, 13)),
        p("Table 5. End-to-end loop wall time and mean per-row latency. "
          "p95 latency, macro F1 and kappa are included in the per-library CSVs.", "caption"),
        pic("cross_library", 2.68 / 8.1),
        p("Figure 5. Positive values favor CapyMOA on accuracy. Values right of 1 favor CapyMOA on latency. "
          "The x-axis is logarithmic.", "caption"),
        p(findings), p(memory),
    ]


def clustering_page():
    rows = []
    runs = data("clustering")
    for r in runs.itertuples():
        rows.append([name(r.stream), r.model, f"{r.mean_ari:.3f}", str(r.scored_blocks),
                     "—" if pd.isna(r.first_post_shift_ari) else f"{r.first_post_shift_ari:.3f}",
                     f"{r.last_4_blocks_ari:.3f}", str(r.final_centers), f"{r.mean_train_us:.1f}"])
    return [
        *section("PART 2 / STEP 9", "Clustering evolving streams"),
        p("Two 6,000-row Gaussian streams have three groups in 2D (σ=0.30). Centers in one stream "
          "move abruptly at row 3,001. CluStream uses 30 micro-clusters, horizon 750 and K-means with "
          "k=3. DenStream uses a 750 horizon, ε=0.55, β=0.20, μ=2, 300 initialization points and "
          "density decay λ=0.25. ClusTree uses horizon 750 and height 5, returning its micro-clusters."),
        p("Give each clusterer features and a constant dummy class. Before training a 250-row block, "
          "assign its points to the nearest center from the previous snapshot. Use hidden groups only for "
          "adjusted Rand index (ARI). Skip blocks before the first nonempty snapshot."),
        table(["Stream", "Model", "Mean ARI", "Blocks", "First shift ARI", "Last 4 ARI",
               "Final centers", "Train µs"], rows,
              [.136, .138, .112, .093, .14, .125, .131, .125], text_cols=(0, 1), separators=(4,)),
        p("Table 6. ARI is 1 for identical partitions, with 0 representing chance agreement in expectation. "
          "Train latency excludes snapshots. Wall time, snapshot cost and nearest-center distance are in the CSVs.", "caption"),
        pic("clustering", 2.45 / 8.1),
        p("Figure 6. ARI on each new block. The dotted line marks the center shift.", "caption"),
        pic("cluster_centers", 2.35 / 8.1),
        p("Figure 7. Cluster centers at three snapshots of the moving stream. ClusTree reports "
          "microsummaries while CluStream and DenStream report macro-clusters.", "caption"),
        p("CluStream gave the strongest mean ARI on both streams. DenStream merged two stationary "
          "groups with these settings, yet separated all three after the move and achieved a final "
          "four-block ARI of 1.000. ClusTree retained 154 micro-centers on the stationary stream. "
          "Its low ARI partly reflects fragmentation from scoring micro-centers as final groups."),
    ]


def automl_page():
    runs = data("automl")
    rows = [[r.model, f"{r.accuracy:.2%}", f"{r.first_1000_after:.1%}", f"{r.wall_s:.3f}",
             f"{r.mean_us:.1f}", f"{r.p95_us:.1f}"] for r in runs.itertuples()]
    return [
        *section("PART 2 / STEP 10", "Online model selection"),
        p("Use the 10,000-row SEA shift again. Compare each selection policy with single Hoeffding "
          "Tree (HT) and Hoeffding Adaptive Tree (HAT) baselines at grace period 100. HAT monitors "
          "its branches with drift detectors and can replace them."),
        p("<b>Bandit.</b> Epsilon-greedy selection with ε=0.15 and burn-in 300 bandit pulls. "
          "Candidates are HT and HAT, each with grace periods 50, 100 and 200."),
        p("<b>Successive halving.</b> Same six candidates. Remove low-scoring models at each rung "
          "with η=2, a selection budget of 10,000 candidate updates and a minimum of one survivor."),
        p("<b>AutoClass.</b> Periodically resample HT and HAT configurations. Start at grace 100 "
          "and confidence 0.001 with search ranges 50–200 and 10⁻⁵–10⁻². Use a 500-row evaluation "
          "window, ensemble size 5 and three new configurations per cycle."),
        table(["Method", "Accuracy", "Post-1k acc.", "Wall s", "Mean µs", "p95 µs"], rows,
              [.23, .148, .167, .148, .154, .153]),
        p("Table 7. Bonus experiment. One prequential run per method. Post-1k starts at row 5,001.", "caption"),
        pic("automl", 2.65 / 8.1),
        p("Figure 8. Accuracy by 250-row block. The dotted line is the true shift. "
          "The HT and HAT curves are the fixed-configuration baselines.", "caption"),
        p("AutoClass had the highest accuracy (89.92%), 0.58 pp above the fixed HAT, with more than "
          "twice its wall time. Successive halving nearly matched HAT accuracy with far higher cost "
          "and tail latency. The bandit did poorly immediately after drift, when its historical rewards "
          "could favor a learner suited to the earlier regime. Fixed HAT is the economical starting "
          "point for this stream and candidate space."),
    ]


def conclusion():
    refs = [
        ("River 0.23.0 documentation and dataset reference", "https://riverml.xyz/0.23.0/"),
        ("CapyMOA documentation", "https://capymoa.org/"),
        ("CapyMOA clustering tutorial", "https://capymoa.org/notebooks/clusterer/clustering.html"),
        ("CapyMOA AutoML tutorial", "https://capymoa.org/notebooks/automl/automl.html"),
        ("Gomes et al. CapyMOA paper (arXiv:2502.07432)", "https://arxiv.org/abs/2502.07432"),
        ("Scikit-learn toy dataset reference", "https://scikit-learn.org/stable/datasets/toy_dataset.html"),
    ]
    return [
        *section("CONCLUSION", "Choosing a streaming method"),
        p("<b>Start with an adaptive linear learner when throughput matters.</b> River’s AdaGrad models "
          "were the lowest-cost members of both supervised comparisons. Use the nonlinear alternatives "
          "when the score improvement on the target stream justifies the measured latency. A bounded "
          "kNN window can help on local regression problems such as Diabetes and Trump Approval."),
        p("<b>Use selective adaptation when a model must recover quickly.</b> ARF offers an ensemble "
          "trade-off on the known synthetic shifts. For an external alerting policy, Page–Hinkley was "
          "effective on the abrupt regression loss increase and HDDM-A gave the best SEA reset result. "
          "Where labels arrive frequently and alarm ground truth is unavailable, validate the recovery "
          "effect on historical replay before enabling automatic resets."),
        p("<b>Match the library to the learner and deployment.</b> River makes it straightforward "
          "to insert online preprocessing or a detector in a per-sample pipeline. CapyMOA offers MOA "
          "classifiers, a streaming clustering API and automated selection. Its classifier costs depend "
          "on the algorithm, JVM and adapter. A Python callback loop and MOA’s optimized evaluation "
          "path represent different deployment choices."),
        p("<b>Choose clustering granularity and selection effort deliberately.</b> CluStream was most "
          "consistent when the expected number of groups was three. DenStream recovered the moving "
          "groups without a fixed k but depended on the chosen density scale. ClusTree can provide "
          "summaries for a separate macro-clustering step. For SEA, the fixed HAT was a strong default "
          "before paying for online hyperparameter searches."),
        p("Limitations", "subhead"),
        p("The long streams use a fixed seed and Elec2 uses a prefix. Repeated timings measure "
          "execution variability on identical streams. They do not give predictive confidence intervals. "
          "Only abrupt supervised drifts have known change points here. Static benchmark rows retain "
          "their dataset order, which can create order effects. The observations assume immediate "
          "labels and available contemporaneous features. Runtime excludes JVM startup and cannot "
          "isolate its contribution to memory use. Results are specific to these versions and settings."),
        p("References and reproducibility", "subhead"),
        *[p(f'<link href="{url}" color="#006d74"><u>{escape(label)}</u></link>', "caption")
          for label, url in refs],
        p(f'<link href="{REPOSITORY}" color="#006d74"><u>Lab repository</u></link> with pinned dependencies, '
          "experiment scripts, measured CSVs and figure sources."),
    ]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=46, bottomMargin=48,
        title="Online machine learning with River and CapyMOA", author="Adam Chgour",
    )
    pages = [
        setup(), supervised("regression"), drift_page("regression"),
        supervised("classification"), drift_page("classification"),
        library_page(), clustering_page(), automl_page(), conclusion(),
    ]
    body = []
    for i, page in enumerate(pages):
        if i:
            body.append(PageBreak())
        body.extend(page)
    doc.build(body, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
