"""Execute the lab results notebook and export its tables and plots as HTML."""
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent
md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell


def main():
    cells = [
        md(
            "# Lab 4 · Online machine learning on data streams\n\n"
            "Adam Chgour · M2 Data Science · Institut Polytechnique de Paris\n\n"
            "This notebook reads the archived experimental outcomes. "
            "See the [PDF report](output/pdf/lab-4-online-machine-learning.pdf) for interpretation and "
            "the [README](README.md) for datasets, model settings and reproduction instructions. "
            "Run `./reproduce.sh` in the repository root to replay the experiments and rebuild this notebook."
        ),
        code(
            "from pathlib import Path\n\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "def read(name):\n"
            "    return pd.read_csv(Path('results') / f'{name}.csv')\n\n"
            "def plot(name):\n"
            "    display(Image(filename=f'figures/{name}.png', width=800))\n\n"
            "pd.options.display.float_format = '{:,.4f}'.format\n"
            "read('environment')"
        ),
        md(
            "## River regression\n\n"
            "All four learners predict each row before updating. MAE and RMSE are in the target’s units. "
            "The panels show mean predict-plus-train latency versus MAE on each stream."
        ),
        code(
            "reg = read('river_regression')\n"
            "display(reg[['stream', 'model', 'n', 'mae', 'rmse', 'r2', 'wall_s', 'mean_us', 'p95_us']])\n"
            "plot('regression')"
        ),
        md(
            "## River regression drift\n\n"
            "Detector inputs are clipped absolute errors scaled using the first 100 targets. "
            "Monitor-only preserves the baseline learner. Reset replaces it after scoring an alarm row. "
            "The synthetic shift starts at row 5,001."
        ),
        code(
            "display(read('regression_drift').drop(columns=['n', 'wall_s', 'p95_us']))\n"
            "plot('regression_drift')"
        ),
        md(
            "## River classification\n\n"
            "Macro F1 weighs both classes equally. Kappa adjusts accuracy for expected agreement "
            "based on the predicted and observed class frequencies."
        ),
        code(
            "clf = read('river_classification')\n"
            "display(clf[['stream', 'model', 'n', 'accuracy', 'macro_f1', 'kappa', 'wall_s', 'mean_us', 'p95_us']])\n"
            "plot('classification')"
        ),
        md(
            "## River classification drift\n\n"
            "The detectors receive 0/1 prediction errors. SEA shifts at row 5,001. "
            "The alarm logs also include the real-stream results. All positions are 1-based."
        ),
        code(
            "display(read('classification_drift').drop(columns=['n', 'wall_s', 'p95_us']))\n"
            "plot('classification_drift')"
        ),
        code(
            "for task in ('regression', 'classification'):\n"
            "    alarms = read(f'{task}_alarms')\n"
            "    display(alarms.groupby(['stream', 'detector', 'policy']).position.agg(['count', 'min', 'max']))"
        ),
        md(
            "## River versus CapyMOA\n\n"
            "Compare each learner pair on identical observations. River uses AdaGrad logistic "
            "regression while the CapyMOA counterpart uses inverse-scaling SGD with log loss."
        ),
        code(
            "cost_keys = ['stream', 'model', 'accuracy', 'wall_s', 'mean_us', 'p95_us', 'rss_peak_mib']\n"
            "capy = read('capymoa_classification')\n"
            "paired = clf[cost_keys].merge(capy[cost_keys], on=['stream', 'model'], suffixes=('_river', '_capymoa'))\n"
            "display(paired)\n"
            "plot('cross_library')"
        ),
        md(
            "## CapyMOA clustering\n\n"
            "Score every new block with the previous snapshot’s centers. Compute adjusted Rand index "
            "against the hidden group IDs. ClusTree exposes micro-centers."
        ),
        code("display(read('clustering'))\nplot('clustering')\nplot('cluster_centers')"),
        md(
            "## Bonus · Online model selection\n\n"
            "Three CapyMOA selectors compete against fixed Hoeffding and Hoeffding Adaptive Trees on SEA. "
            "Post-change accuracy covers the first 1,000 observations after the switch."
        ),
        code(
            "display(read('automl')[['model', 'accuracy', 'macro_f1', 'kappa', "
            "'first_1000_after', 'wall_s', 'mean_us', 'p95_us']])\n"
            "plot('automl')"
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
    })
    ExecutePreprocessor(timeout=90, kernel_name="python3").preprocess(
        notebook, resources={"metadata": {"path": str(ROOT)}},
    )
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / "Lab_4.ipynb")
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (ROOT / "Lab_4.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
