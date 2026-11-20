"""Load the ordered real-data replays from official UCI dataset archives."""
import os
import shutil
from gzip import GzipFile
from io import TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import arff

UCI = "https://archive.ics.uci.edu/static/public"
ARCHIVES = {
    "seoul": "560/seoul+bike+sharing+demand.zip",
    "appliances": "374/appliances+energy+prediction.zip",
    "traffic": "492/metro+interstate+traffic+volume.zip",
    "occupancy": "357/occupancy+detection.zip",
    "eeg": "264/eeg+eye+state.zip",
    "gas": "224/gas+sensor+array+drift+dataset.zip",
}
OCCUPANCY_FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]


def archive(key):
    folder = Path(os.environ.get("LAB_DATA_DIR", Path(__file__).parent / ".data"))
    path = folder / f"{key}.zip"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=folder, delete=False) as temp:
            downloaded = Path(temp.name)
            try:
                with urlopen(f"{UCI}/{ARCHIVES[key]}", timeout=60) as response:
                    shutil.copyfileobj(response, temp)
                temp.flush()
                with ZipFile(downloaded):
                    pass
                downloaded.replace(path)
            finally:
                downloaded.unlink(missing_ok=True)
    return ZipFile(path)


def records(features, target):
    names = tuple(features.columns)
    for values, label in zip(features.itertuples(index=False, name=None), target, strict=True):
        yield dict(zip(names, values, strict=True)), label


def next_interval(frame, target, feature_names, interval, limit=None):
    """Forecast the next adjacent interval from current observations and its calendar."""
    ordered = frame.sort_values("time", kind="stable").drop_duplicates("time")
    target_time = ordered.time.shift(-1)
    contiguous = (target_time - ordered.time) == pd.Timedelta(interval)
    observed = ordered.loc[contiguous]
    times = target_time.loc[contiguous].dt
    hour = times.hour + times.minute / 60
    features = observed[feature_names].copy()
    features["lag_target"] = observed[target]
    for name, angle in (("hour", 2 * np.pi * hour / 24),
                        ("weekday", 2 * np.pi * times.dayofweek / 7)):
        features[f"{name}_sin"] = np.sin(angle)
        features[f"{name}_cos"] = np.cos(angle)
    return records(features.iloc[:limit], ordered[target].shift(-1).loc[contiguous].iloc[:limit])


def seoul_bikes():
    with archive("seoul") as z, z.open("SeoulBikeData.csv") as f:
        frame = pd.read_csv(f, encoding="latin-1")
    frame["time"] = pd.to_datetime(frame["Date"], format="%d/%m/%Y") + pd.to_timedelta(frame.Hour, unit="h")
    weather = [
        "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)",
        "Dew point temperature(°C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)",
    ]
    return next_interval(frame, "Rented Bike Count", weather, "1h")


def appliance_energy():
    with archive("appliances") as z, z.open("energydata_complete.csv") as f:
        frame = pd.read_csv(f)
    frame["time"] = pd.to_datetime(frame.date)
    sensors = ["lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3"]
    return next_interval(frame, "Appliances", sensors, "10min")


def metro_traffic():
    with archive("traffic") as z, z.open("Metro_Interstate_Traffic_Volume.csv.gz") as f:
        with GzipFile(fileobj=f) as uncompressed:
            frame = pd.read_csv(uncompressed)
    frame["time"] = pd.to_datetime(frame.date_time)
    # Zero Kelvin in this file means an invalid observation, not a temperature.
    frame = frame.loc[frame.temp.between(200, 330)].copy()
    frame["raining"] = (frame.rain_1h > 0).astype(int)
    frame["snowing"] = (frame.snow_1h > 0).astype(int)
    sensors = ["temp", "clouds_all", "raining", "snowing"]
    return next_interval(frame, "traffic_volume", sensors, "1h", limit=12_000)


def occupancy_frame():
    chunks = []
    with archive("occupancy") as z:
        for name in ("datatest.txt", "datatraining.txt", "datatest2.txt"):
            with z.open(name) as f:
                chunks.append(pd.read_csv(f, index_col=0, parse_dates=["date"]))
    return pd.concat(chunks, ignore_index=True).sort_values("date", kind="stable")


def occupancy():
    frame = occupancy_frame()
    return records(frame[OCCUPANCY_FEATURES], frame.Occupancy)


def occupancy_clustering():
    frame = occupancy_frame().iloc[:6000]
    # Express light in hundreds of lux and CO2 in units of 250 ppm.
    return frame[["Light", "CO2"]].to_numpy() / [100., 250.], frame.Occupancy.to_numpy()


def eeg_eye_state():
    with archive("eeg") as z, z.open("EEG Eye State.arff") as f:
        observations, _ = arff.loadarff(TextIOWrapper(f, encoding="ascii"))
    frame = pd.DataFrame(observations)
    return records(frame.drop(columns="eyeDetection"), frame.eyeDetection.astype(int))


def gas_sensor_drift(seed):
    rng = np.random.default_rng(seed)
    response_indices = range(2, 129, 8)  # Normalized steady-state response of each sensor.
    labels = {"1": 0, "5": 1}  # Ethanol vs acetone.
    with archive("gas") as z:
        for batch in range(1, 11):
            observations = []
            with z.open(f"Dataset/batch{batch}.dat") as f:
                for line in f:
                    label, *values = line.decode("ascii").split()
                    if label in labels:
                        full = dict(entry.split(":") for entry in values)
                        x = {f"sensor_{i}": float(full[str(index)])
                             for i, index in enumerate(response_indices, 1)}
                        observations.append((x, labels[label]))
            # The archive groups measurements by gas. Mix within each dated batch;
            # the order of batches, and hence the long-term drift, is preserved.
            for i in rng.permutation(len(observations)):
                yield observations[i]
