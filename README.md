# Lab 4 · Online machine learning on data streams

Adam Chgour · M2 Data Science · Institut Polytechnique de Paris

- [PDF report](output/pdf/lab-4-online-machine-learning.pdf)
- [Jupyter notebook](Lab_4.ipynb) · [Executed HTML export](Lab_4.html)
- [Measured results](results/) · [Figures](figures/)

## Run the notebook

Open `Lab_4.ipynb` in Jupyter with a Python 3.12 kernel and OpenJDK 17 available. Install the packages listed at the top of the notebook or run `python -m pip install -r requirements.txt` in the kernel's environment. Restart the kernel and run all cells in order. If several JDKs are installed, set `JAVA_HOME` to the one CapyMOA should use.

The notebook contains the data loaders, model definitions, evaluation loops and plots. The public UCI archives download into `.data/` on first use. Set `LAB_DATA_DIR` to use a different cache. Reruns save CSVs and figures under `lab4_run/`. The notebook and HTML include the saved tables and figures from the run used in the report. Read the HTML locally in a browser.

## Compile the report

Run `tectonic -X compile report.tex` from the repository root to produce a PDF using the archived `report-tables.tex` and `figures/`. Package and hardware details are in [environment.csv](results/environment.csv).

## Experimental design

Predict each observation in order, score against its target and train on it. Every row contributes to the metrics. An abstention maps to 0. Seed 42 is used wherever configurable. Linear models and regression kNN receive features standardized using earlier observations. Trees and Gaussian Naive Bayes use the supplied numeric features. The AdaGrad regressor also standardizes targets using its online state and transforms predictions back to the target's original units.

| Task | Streams and targets | Rows |
| --- | --- | ---: |
| Regression | [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), next-hour rental count | 8,759 |
| Regression | [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), next 10-minute appliance energy use (Wh) | 19,734 |
| Regression | [Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume), next-hour vehicle count, first 12,000 eligible predictions | 12,000 |
| Regression | Friedman shift, active features and target level change at row 5,001 | 10,000 |
| Classification | [Occupancy Detection](https://archive.ics.uci.edu/dataset/357/occupancy+detection), occupied versus empty from five environmental measurements | 20,560 |
| Classification | [EEG Eye State](https://archive.ics.uci.edu/dataset/264/eeg+eye+state), open versus closed from 14 channels in a single recording | 14,980 |
| Classification | [Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset), ethanol versus acetone from 16 normalized sensor responses | 5,574 |
| Classification | SEA shift, threshold changes from 8 to 13 at row 5,001, with 5% label noise | 10,000 |
| Clustering | First 6,000 occupancy rows using light and CO₂, plus 6,000 simulated points with a center shift at row 3,001 | 12,000 total |

Regression inputs combine the previous interval's measured target and selected sensors with four cyclic features for the known time and day of the prediction. Only adjacent intervals form forecast pairs. For Metro traffic, temperature must be within 200–330 K and the first record at each repeated timestamp is retained. Appliance energy uses preceding light, temperature and humidity readings from the house. Seoul uses the preceding weather readings.

The occupancy files are merged by timestamp, and EEG follows the published recording order. Gas sensors follow the ten dated batches. Each batch is shuffled with seed 42 to remove the source file's grouping by gas; the order of batches is preserved. The real replays have no labeled drift instants.

- **Regression.** Compare a Hoeffding tree and five-tree adaptive random forest with mean leaves, AdaGrad linear regression, and exact kNN with five neighbors and a 200-row window. Metrics are MAE, RMSE and R².
- **Classification.** Compare a Hoeffding tree, five-tree adaptive random forest, logistic learning and Gaussian Naive Bayes. Metrics are accuracy, macro F1 and Cohen’s kappa. The two libraries use the same input sequence and match the main tree and forest settings. Each forest tests the same number of features per split, the rounded square root of the feature count. CapyMOA uses inverse-scaling SGD with log loss as its closest counterpart to River’s AdaGrad logistic model. Both start at a learning rate of 0.05. Sections 2 and 5 of the notebook contain the model settings, including the MOA forest CLI.
- **Drift.** Apply ADWIN, Page–Hinkley and KSWIN to regression and ADWIN, DDM, EDDM and HDDM-A to classification. Starting at row 101, detectors receive the Hoeffding tree’s bounded absolute error in regression or 0/1 error in classification. Monitor mode records alarms. Reset mode replaces the learner after scoring the alarm row and trains the replacement on that row. Delay is measured from the first changed row to the first alarm in a 1,000-row window. All positions are 1-based. Real change points are unlabelled.
- **Clustering.** Compare CluStream + K-means, DenStream + DBSCAN and ClusTree. Assign the next block of 250 points to the nearest centers from the previous boundary. Score adjusted Rand index on hidden group IDs, and report mean distance and center count. Scoring ClusTree’s micro-centers as final groups penalizes fragmentation. The Java schema contains a dummy target. Room-sensor light and CO₂ are expressed in units of 100 lux and 250 ppm, respectively. CluStream uses two macro-clusters for room sensors and three for the simulation.
- **AutoML.** Compare AutoClass, epsilon-greedy Bandit and Successive Halving on SEA shift with fixed Hoeffding and Hoeffding Adaptive Trees as baselines. Bandit and halving use a shared six-model grid, while AutoClass samples a wider parameter range in the same two learner families. Section 7 of the notebook contains the candidate settings.

### Cost measurements

Supervised tables use componentwise medians across three timed replays with fresh models. CapyMOA receives one discarded warm-up of up to 256 rows per pair. Its per-row API includes any lazy conversion from Python to Java. Model creation, loading, feature preparation, JVM startup and final metric aggregation are outside the timed loop. Wall time includes loop and instrumentation overhead. Mean and p95 latency time prediction, training and detector work where applicable. RSS is notebook-process memory sampled every 250 rows and at the end. River runs before the JVM starts. CapyMOA measurements include the live JVM, notebook state and allocations retained from earlier runs. The supervised replays check exact agreement of predictions across repeated runs.

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
