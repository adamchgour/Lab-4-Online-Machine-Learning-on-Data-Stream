# Lab 4 · Online machine learning on data streams

Adam Chgour · M2 Data Science · Institut Polytechnique de Paris

- [PDF report](output/pdf/lab-4-online-machine-learning.pdf) · [LaTeX source](report.tex) — setup, results, interpretation and conclusions
- [Executed notebook](Lab_4.ipynb) · [HTML export](Lab_4.html) — tables, figures and reproduction commands
- [Measured results](results/) · [Figures](figures/)

## Reproduce the experiments

Use Python 3.12, a JDK and Tectonic on your PATH. The archived run used Python 3.12.7 and OpenJDK 17.0.14 on an Apple M2 with 16 GB RAM. See [environment.csv](results/environment.csv) for the measured versions. The experiments use CapyMOA’s CPU and Java methods.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
java -version
./reproduce.sh
```

If several JDKs are installed, set `JAVA_HOME` to the JDK CapyMOA should use. River downloads Elec2 on its first use. You can set `RIVER_DATA` to choose the cache directory. The library loaders provide the real datasets. [streams.py](streams.py) generates the synthetic streams.

The shell script runs each stage in a fresh process, then builds the figures, PDF, notebook and HTML. Run a stage individually with `python benchmark.py STAGE`, where `STAGE` is `environment`, `regression`, `classification`, `capymoa`, `clustering` or `automl`. Timings vary with hardware and system load. The AutoML stage took about a minute on the reference machine.

## Compile the report

`report.tex` contains the report text and layout. Run `python build_report.py` from the repository root with the Python dependencies installed to regenerate `report-tables.tex` from the archived CSVs and compile the PDF using Tectonic. This uses the existing figures and requires no experiment rerun. Tectonic may fetch its TeX bundle on the first compile.

## Experimental design

Predict each observation in order, score against its target and train on it. Every row contributes to the metrics. An abstention maps to 0. Seed 42 is used wherever configurable. Linear models and regression kNN receive features standardized using earlier observations. Trees and Gaussian Naive Bayes use raw features. Elec2 uses the supplied contemporaneous covariates. The static benchmark datasets are replayed in source order.

| Task | Real streams | Synthetic |
| --- | --- | --- |
| Regression | Trump Approval (1,001), Chick Weights (578), unscaled Diabetes (442) | Friedman shift (10,000), active features and target level change at row 5,001 |
| Classification | Elec2 (first 15,000), Phishing (1,250), Breast Cancer (569) | SEA shift (10,000), threshold 8 to 13 at row 5,001, 5% independent label noise |
| Clustering | | Two 6,000-row Gaussian streams with three groups, one with a center shift at row 3,001 |

- **Regression.** Compare a Hoeffding tree and five-tree adaptive random forest with mean leaves, AdaGrad linear regression, and exact kNN with five neighbors and a 200-row window. Metrics are MAE, RMSE and R².
- **Classification.** Compare a Hoeffding tree, five-tree adaptive random forest, logistic learning and Gaussian Naive Bayes. Metrics are accuracy, macro F1 and Cohen’s kappa. The two libraries use the same input sequence and match the main tree and forest settings. Each forest tests the same number of features per split, the rounded square root of the feature count. CapyMOA uses inverse-scaling SGD with log loss as its closest counterpart to River’s AdaGrad logistic model. Both start at a learning rate of 0.05. See [river_models.py](river_models.py) and [capy_models.py](capy_models.py), including the explicit MOA forest CLI.
- **Drift.** Apply ADWIN, Page–Hinkley and KSWIN to regression and ADWIN, DDM, EDDM and HDDM-A to classification. Starting at row 101, detectors receive the Hoeffding tree’s bounded absolute error in regression or 0/1 error in classification. Monitor mode records alarms. Reset mode replaces the learner after scoring the alarm row and trains the replacement on that row. Delay is measured from the first changed row to the first alarm in a 1,000-row window. All positions are 1-based. Real change points are unlabelled.
- **Clustering.** Compare CluStream + K-means, DenStream + DBSCAN and ClusTree. Assign the next block of 250 points to the nearest centers from the previous boundary. Score adjusted Rand index on hidden group IDs, and report mean distance and center count. Scoring ClusTree’s micro-centers as final groups penalizes fragmentation. The Java schema contains a dummy target.
- **AutoML.** Compare AutoClass, epsilon-greedy Bandit and Successive Halving on SEA shift with fixed Hoeffding and Hoeffding Adaptive Trees as baselines. Bandit and halving use a shared six-model grid, while AutoClass samples a wider parameter range in the same two learner families. See [autoclass.json](autoclass.json) and [unsupervised_automl.py](unsupervised_automl.py).

### Cost measurements

Supervised tables use componentwise medians across three timed replays with fresh models. CapyMOA receives one discarded warm-up of up to 256 rows per pair. Its per-row API includes any lazy conversion from Python to Java. Model creation, loading, feature preparation, JVM startup and final metric aggregation are outside the timed loop. Wall time includes loop and instrumentation overhead. Mean and p95 latency time prediction, training and detector work where applicable. RSS is process memory sampled every 250 rows and at the end, so it includes inputs, retained allocations and the JVM where applicable. The supervised replays check exact agreement of predictions across repeated runs.

### Result files

| CSV | Contents |
| --- | --- |
| `river_regression`, `river_classification`, `capymoa_classification` | One row per learner and stream with scores and median costs |
| `*_timings`, `*_blocks` | Timing replays and contiguous 250-row scores. The final block can be shorter. |
| `{regression,classification}_drift` | Alarm counts, pre-change counts, delay, 1,000-row and full-stream scores, costs by policy |
| `{regression,classification}_alarms` | Every alarm and its position |
| `{regression,classification}_drift_blocks` | Block scores by drift policy |
| `clustering*` | Aggregate and block ARI, mean distances, timing and center coordinates |
| `automl*` | Selector and baseline scores, costs and block accuracy |
| `environment` | Hardware and main package versions |

### References

- Mariam Barry, *Online Machine Learning on Data Streams*, course slides, 2026. Slides 16–19 cover streaming evaluation and drift, 29–32 cover Hoeffding splits, and 47–50 cover adaptive random forests.

- [River 0.23 documentation and dataset catalog](https://riverml.xyz/0.23.0/)
- [CapyMOA documentation](https://capymoa.org/)
- [CapyMOA clustering tutorial](https://capymoa.org/notebooks/clusterer/clustering.html)
- [CapyMOA AutoML tutorial](https://capymoa.org/notebooks/automl/automl.html)
- [Gomes et al., CapyMOA](https://arxiv.org/abs/2502.07432)
- [Scikit-learn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html)
