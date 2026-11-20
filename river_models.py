"""River learner and detector configurations."""
from river import drift, forest, linear_model, naive_bayes, neighbors, optim, preprocessing, tree


def regression_models(seed):
    return {
        "HT": lambda: tree.HoeffdingTreeRegressor(grace_period=100, leaf_prediction="mean"),
        "ARF": lambda: forest.ARFRegressor(
            n_models=5, grace_period=100, leaf_prediction="mean", seed=seed,
        ),
        "AdaGrad": lambda: preprocessing.TargetStandardScaler(
            regressor=linear_model.LinearRegression(
                optimizer=optim.AdaGrad(.10), intercept_lr=.05,
            ),
        ),
        "kNN": lambda: neighbors.KNNRegressor(
            n_neighbors=5, engine=neighbors.LazySearch(window_size=200), aggregation_method="mean",
        ),
    }


def classification_models(seed):
    return {
        "HT": lambda: tree.HoeffdingTreeClassifier(
            grace_period=100, delta=1e-5, tau=.05, leaf_prediction="mc",
        ),
        "ARF": lambda: forest.ARFClassifier(
            n_models=5, max_features="sqrt", lambda_value=6, grace_period=100,
            delta=1e-5, tau=.05, leaf_prediction="mc", seed=seed,
        ),
        "Logistic": lambda: linear_model.LogisticRegression(
            optimizer=optim.AdaGrad(.05), intercept_lr=.05,
        ),
        "GaussianNB": naive_bayes.GaussianNB,
    }


def detectors(task):
    adwin = lambda: drift.ADWIN(delta=.002)
    if task == "regression":
        return {
            "ADWIN": adwin,
            "PageHinkley": lambda: drift.PageHinkley(min_instances=30, delta=.005, threshold=25, mode="up"),
            "KSWIN": lambda: drift.KSWIN(alpha=.005, window_size=100, stat_size=30, seed=42),
        }
    return {
        "ADWIN": adwin,
        "DDM": drift.binary.DDM,
        "EDDM": drift.binary.EDDM,
        "HDDM_A": drift.binary.HDDM_A,
    }
