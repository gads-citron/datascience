import datetime
import os
import time
import urllib.parse

import clickhouse_connect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clickhouse_connect.driver.exceptions import OperationalError
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from umap import UMAP

DATASET = 'datamart_sous_comptage.distributed_dataset_hourly_payloads_merged'
PAYLOAD_DATASET = 'datawarehouse_prod.fact_payload'
STATS_OUTPUT = "statistiques_fenetre_4_semaines.csv"
FEATURES_OUTPUT = "anomalies_variables.csv"
CLUSTERED_OUTPUT = "anomalies_clustered.csv"
STATS_COLUMNS = [
    "device_id", "sensor_id", "sensor_name", "year", "week",
    "mean_eval_prev_hour", "median_eval_prev_hour", "std_eval_prev_hour",
    "min_eval_prev_hour", "max_eval_prev_hour",
    "q25_eval_prev_hour", "q75_eval_prev_hour", "n_observations_semestre",
]
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10

CATEGORICAL_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
OTHER_COLOR = "#898781"
SEQUENTIAL_PALETTE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]

# NUMERIC_FEATURES = [
#    "mean_eval_prev_hour", "median_eval_prev_hour", "std_eval_prev_hour",
#     "min_eval_prev_hour", "eval_prev_hour", "threshold",
#     "diff_eval_threshold", "q25_eval_prev_hour", "q75_eval_prev_hour",
#     "ecart_mediane", "ecart_iqr", "seuil_relatif", "amplitude_iqr",
#     "ratio_valeur_seuil", "anomaly_score", "threshold_score", "relative_deviation",
#     "iqr_median", "q75_q25_ratio", "anomaly_median_ratio", "anomaly_q75_ratio", "anomaly_q25_ratio",
#     "max_eval_prev_hour"]

# NUMERIC_FEATURES = [
#     "ecart_mediane", "ecart_iqr", "seuil_relatif", "amplitude_iqr",
#     "ratio_valeur_seuil", "relative_deviation",
#     "iqr_median", "q75_q25_ratio", "anomaly_median_ratio", "anomaly_q75_ratio", "anomaly_q25_ratio",
# ]
# reduit apres analyse de correlation (spearman) : plusieurs des 11 variables ci-dessus
# sont quasi-dupliquees (corr >= 0.95) - ecart_mediane/anomaly_median_ratio, iqr_median/q75_q25_ratio,
# ecart_iqr/anomaly_q75_ratio, ecart_iqr/seuil_relatif, ecart_mediane/anomaly_q25_ratio.
# on garde un representant par axe : ecart a la mediane, ecart normalise par l'IQR,
# dispersion brute, position par rapport au seuil.
NUMERIC_FEATURES = ["ecart_mediane", "ecart_iqr", "amplitude_iqr", "ratio_valeur_seuil"]
# CATEGORICAL_FEATURES = ["sensor_name", "customer_account_name"]
CATEGORICAL_FEATURES = []

MERGED_COLUMNS = [
    "device_id", "sensor_id", "site_id",
    "mean_eval_prev_hour", "median_eval_prev_hour", "std_eval_prev_hour",
    "min_eval_prev_hour", "max_eval_prev_hour", "q25_eval_prev_hour", "q75_eval_prev_hour",
    "payload_expected_time", "eval_prev_hour", "smoothed_value", "q25", "q75",
    "k", "threshold", "sensor_name", "unit", "customer_account_name",
]

ZERO_REPLACEMENT = 0.001


def db_connect(type='prod'):
    load_dotenv()
    username = urllib.parse.quote_plus(os.environ.get("USER_NAME"))
    if type == 'prod':
        host = urllib.parse.quote_plus(os.environ.get("clk_prod"))
        password = urllib.parse.quote_plus(os.environ.get('clk_prod_pass'))
    elif type == 'stage':
        host = urllib.parse.quote_plus(os.environ.get("clk_stg"))
        password = urllib.parse.quote_plus(os.environ.get('clk_stg_pass'))
    return clickhouse_connect.get_client(host=host, username=username, port=443, password=password)

client = db_connect()

def week_bounds(year, week):
    monday = datetime.date.fromisocalendar(year, week, 1)
    start = monday - datetime.timedelta(weeks=3)
    end = monday + datetime.timedelta(weeks=1)
    return start.isoformat(), end.isoformat()

def query_window_stats(device_id, sensor_id, year, week):
    start, end = week_bounds(year, week)
    query = f"""SELECT
        avg(eval_prev_hour) AS mean_eval_prev_hour,
        median(eval_prev_hour) AS median_eval_prev_hour,
        stddevSamp(eval_prev_hour) AS std_eval_prev_hour,
        min(eval_prev_hour) AS min_eval_prev_hour,
        max(eval_prev_hour) AS max_eval_prev_hour,
        quantile(0.25)(eval_prev_hour) AS q25_eval_prev_hour,
        quantile(0.75)(eval_prev_hour) AS q75_eval_prev_hour,
        count(eval_prev_hour) AS n_observations_semestre
    FROM {DATASET}
    WHERE device_id = '{device_id}'
      AND sensor_id = '{sensor_id}'
      AND payload_expected_time >= '{start}'
      AND payload_expected_time < '{end}'
    settings max_rows_to_read=10000000000000
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.query_df(query)
        except OperationalError:
            if attempt == MAX_RETRIES:
                raise
            print(f"requete echouee (tentative {attempt}/{MAX_RETRIES}), nouvel essai dans {RETRY_DELAY_SECONDS}s")
            time.sleep(RETRY_DELAY_SECONDS)


def query_payloads(device_id, sensor_id, year):
    query = f"""SELECT
        payload_timestamp,
        toFloat64OrNull(payload_value) AS payload_value
    FROM {PAYLOAD_DATASET}
    WHERE device_id = '{device_id}'
      AND payload_metric_id = '{sensor_id}'
      AND payload_timestamp >= '{year}-01-01'
      AND payload_timestamp < '{year + 1}-01-01'
    ORDER BY payload_timestamp
    settings max_rows_to_read=10000000000000
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.query_df(query)
        except OperationalError:
            if attempt == MAX_RETRIES:
                raise
            print(f"requete echouee (tentative {attempt}/{MAX_RETRIES}), nouvel essai dans {RETRY_DELAY_SECONDS}s")
            time.sleep(RETRY_DELAY_SECONDS)


def query_device_info(device_ids):
    device_list = ",".join(f"'{d}'" for d in device_ids)
    query = f"""SELECT
        device_id,
        any(device_name) AS device_name,
        any(usage) AS usage
    FROM {DATASET}
    WHERE device_id IN ({device_list})
    GROUP BY device_id
    settings max_rows_to_read=10000000000000
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.query_df(query)
        except OperationalError:
            if attempt == MAX_RETRIES:
                raise
            print(f"requete echouee (tentative {attempt}/{MAX_RETRIES}), nouvel essai dans {RETRY_DELAY_SECONDS}s")
            time.sleep(RETRY_DELAY_SECONDS)


def add_device_info(clustered_df):
    device_ids = clustered_df["device_id"].unique().tolist()
    info = query_device_info(device_ids)
    return clustered_df.merge(info, on="device_id", how="left")


def query_hourly_eval(device_id, sensor_id, year):
    query = f"""SELECT
        payload_expected_time,
        eval_prev_hour
    FROM {DATASET}
    WHERE device_id = '{device_id}'
      AND sensor_id = '{sensor_id}'
      AND payload_expected_time >= '{year}-01-01'
      AND payload_expected_time < '{year + 1}-01-01'
    ORDER BY payload_expected_time
    settings max_rows_to_read=10000000000000
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.query_df(query)
        except OperationalError:
            if attempt == MAX_RETRIES:
                raise
            print(f"requete echouee (tentative {attempt}/{MAX_RETRIES}), nouvel essai dans {RETRY_DELAY_SECONDS}s")
            time.sleep(RETRY_DELAY_SECONDS)


def plot_device_eval_timeseries(device_id, sensor_id, year, anomalies_path="anomalies_all.csv"):
    hourly = query_hourly_eval(device_id, sensor_id, year)
    if hourly.empty:
        print(f"aucune donnee eval_prev_hour pour {device_id}/{sensor_id} en {year}")
        return None

    anomalies = pd.read_csv(anomalies_path, encoding="utf-8")
    anomalies["payload_expected_time"] = pd.to_datetime(anomalies["payload_expected_time"])
    anomalies_sub = anomalies[
        (anomalies["device_id"] == device_id)
        & (anomalies["sensor_id"] == sensor_id)
        & (anomalies["payload_expected_time"].dt.year == year)
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hourly["payload_expected_time"], hourly["eval_prev_hour"], color=CATEGORICAL_PALETTE[0], linewidth=1)
    ax.scatter(
        anomalies_sub["payload_expected_time"], anomalies_sub["eval_prev_hour"],
        color="#d03b3b", s=30, zorder=5, label="anomalie",
    )

    ax.set_xlabel("payload_expected_time")
    ax.set_ylabel("eval_prev_hour")
    ax.set_title(f"{device_id} / {sensor_id} - {year}")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()
    return fig


def plot_device_eval_timeseries_clustered(device_id, sensor_id, year, clustered_df, anomalies_path="anomalies_all.csv"):
    hourly = query_hourly_eval(device_id, sensor_id, year)
    if hourly.empty:
        print(f"aucune donnee eval_prev_hour pour {device_id}/{sensor_id} en {year}")
        return None

    anomalies = pd.read_csv(anomalies_path, encoding="utf-8")
    anomalies["payload_expected_time"] = pd.to_datetime(anomalies["payload_expected_time"])
    anomalies_sub = anomalies[
        (anomalies["device_id"] == device_id)
        & (anomalies["sensor_id"] == sensor_id)
        & (anomalies["payload_expected_time"].dt.year == year)
    ].copy()

    cluster_lookup = clustered_df[
        (clustered_df["device_id"] == device_id) & (clustered_df["sensor_id"] == sensor_id)
    ][["payload_expected_time", "cluster"]]
    anomalies_sub = anomalies_sub.merge(cluster_lookup, on="payload_expected_time", how="left")
    anomalies_sub["cluster_label"] = anomalies_sub["cluster"].apply(
        lambda c: "non clusterise" if pd.isna(c) else ("bruit" if c == -1 else f"cluster {int(c)}")
    )

    color_map = {}
    palette_idx = 0
    for label in sorted(anomalies_sub["cluster_label"].unique()):
        if label == "non clusterise":
            color_map[label] = "#c3c2b7"
        elif label == "bruit":
            color_map[label] = OTHER_COLOR
        else:
            color_map[label] = CATEGORICAL_PALETTE[palette_idx % len(CATEGORICAL_PALETTE)]
            palette_idx += 1

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hourly["payload_expected_time"], hourly["eval_prev_hour"], color="#898781", linewidth=1)
    for label, group in anomalies_sub.groupby("cluster_label"):
        ax.scatter(
            group["payload_expected_time"], group["eval_prev_hour"],
            color=color_map[label], s=30, zorder=5, label=label,
        )

    ax.set_yscale("log")
    ax.set_xlabel("payload_expected_time")
    ax.set_ylabel("eval_prev_hour (echelle log)")
    ax.set_title(f"{device_id} / {sensor_id} - {year}")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()
    return fig


def plot_device_timeseries(device_id, sensor_id, year, anomalies_path="anomalies_all.csv"):
    payloads = query_payloads(device_id, sensor_id, year)
    if payloads.empty:
        print(f"aucune donnee payload pour {device_id}/{sensor_id} en {year}")
        return None

    anomalies = pd.read_csv(anomalies_path, encoding="utf-8")
    anomalies["payload_expected_time"] = pd.to_datetime(anomalies["payload_expected_time"])
    anomalies_sub = anomalies[
        (anomalies["device_id"] == device_id)
        & (anomalies["sensor_id"] == sensor_id)
        & (anomalies["payload_expected_time"].dt.year == year)
    ].sort_values("payload_expected_time")

    anomalies_sub = pd.merge_asof(
        anomalies_sub, payloads.sort_values("payload_timestamp"),
        left_on="payload_expected_time", right_on="payload_timestamp",
        direction="nearest",
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(payloads["payload_timestamp"], payloads["payload_value"], color=CATEGORICAL_PALETTE[0], linewidth=1)
    ax.scatter(
        anomalies_sub["payload_expected_time"], anomalies_sub["payload_value"],
        color="#d03b3b", s=30, zorder=5, label="anomalie",
    )

    ax.set_xlabel("payload_timestamp")
    ax.set_ylabel("payload_value")
    ax.set_title(f"{device_id} / {sensor_id} - {year}")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()
    return fig


def build_combos(anomalies_path="anomalies_all.csv"):
    anomalies = pd.read_csv(anomalies_path, encoding="utf-8")
    anomalies["payload_expected_time"] = pd.to_datetime(anomalies["payload_expected_time"])
    anomalies = anomalies[
        (anomalies["eval_prev_hour"] > 1000)
        & (anomalies["payload_expected_time"] > "2023-01-01")
    ]
    iso = anomalies["payload_expected_time"].dt.isocalendar()
    anomalies["year"] = iso["year"]
    anomalies["week"] = iso["week"]
    return anomalies[["device_id", "sensor_id", "sensor_name", "year", "week"]].drop_duplicates()

def load_done_combos():
    if not os.path.exists(STATS_OUTPUT):
        return set()
    existing = pd.read_csv(STATS_OUTPUT, encoding="utf-8")
    return set(zip(existing["device_id"], existing["sensor_id"], existing["year"], existing["week"]))

def fetch_stats_from_db():
    combos = build_combos()
    total = len(combos)
    print(f"{total} combos device/sensor/annee/semaine a traiter")

    done = load_done_combos()
    if done:
        print(f"{len(done)} combos deja traites, reprise du run")
    file_exists = os.path.exists(STATS_OUTPUT)

    n_written = 0
    for i, combo in enumerate(combos.itertuples(index=False), start=1):
        if (combo.device_id, combo.sensor_id, combo.year, combo.week) in done:
            continue

        stats = query_window_stats(combo.device_id, combo.sensor_id, combo.year, combo.week)

        if stats.empty or stats.loc[0, "n_observations_semestre"] == 0:
            continue

        row = stats.iloc[0].to_dict()
        row["device_id"] = combo.device_id
        row["sensor_id"] = combo.sensor_id
        row["sensor_name"] = combo.sensor_name
        row["year"] = combo.year
        row["week"] = combo.week

        row_df = pd.DataFrame([row])[STATS_COLUMNS]
        row_df.to_csv(STATS_OUTPUT, mode="a", header=not file_exists, index=False, encoding="utf-8")
        file_exists = True
        n_written += 1

        if i % 200 == 0:
            print(f"{i}/{total} combos traites")

    print(f"{n_written} nouvelles lignes sauvegardees dans {STATS_OUTPUT}")


def load_pca_dataset(anomalies_path="anomalies_all.csv", stats_path="statistiques_fenetre_4_semaines.csv"):
    anomalies = pd.read_csv(anomalies_path, encoding="utf-8")
    stats = pd.read_csv(stats_path, encoding="utf-8")

    anomalies["payload_expected_time"] = pd.to_datetime(anomalies["payload_expected_time"])
    anomalies = anomalies[anomalies["eval_prev_hour"] > 1000]
    iso = anomalies["payload_expected_time"].dt.isocalendar()
    anomalies["year"] = iso["year"]
    anomalies["week"] = iso["week"]

    merged = anomalies.merge(stats, on=["device_id", "sensor_id", "year", "week"], suffixes=("", "_stats"))
    return merged[MERGED_COLUMNS].reset_index(drop=True)


def _no_zero(series):
    return series.replace(0, ZERO_REPLACEMENT)


def add_derived_features(df):
    data = df.copy()

    valeur = data["eval_prev_hour"]
    mediane = data["median_eval_prev_hour"]
    moyenne = data["mean_eval_prev_hour"]
    seuil = data["threshold"]
    q25 = data["q25_eval_prev_hour"]
    q75 = data["q75_eval_prev_hour"]
    iqr = q75 - q25

    mediane_safe = _no_zero(mediane)
    moyenne_safe = _no_zero(moyenne)
    seuil_safe = _no_zero(seuil)
    q25_safe = _no_zero(q25)
    q75_safe = _no_zero(q75)
    iqr_safe = _no_zero(iqr)

    data["ecart_mediane"] = (valeur - mediane) / mediane_safe
    data["ecart_iqr"] = (valeur - mediane) / iqr_safe
    data["seuil_relatif"] = (seuil - mediane) / iqr_safe
    data["amplitude_iqr"] = iqr
    data["ratio_valeur_seuil"] = valeur / seuil_safe
    data["anomaly_score"] = valeur
    data["threshold_score"] = seuil
    data["relative_deviation"] = (valeur - moyenne) / moyenne_safe
    data["iqr_median"] = iqr / mediane_safe
    data["q75_q25_ratio"] = q75 / q25_safe
    data["anomaly_median_ratio"] = valeur / mediane_safe
    data["anomaly_q75_ratio"] = valeur / q75_safe
    data["anomaly_q25_ratio"] = valeur / q25_safe

    return data

def _signed_log1p(series):
    return np.sign(series) * np.log1p(series.abs())


def _build_scaled_features(df):
    data = df.copy()
    data["diff_eval_threshold"] = data["eval_prev_hour"] - data["threshold"]
    data = data.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)

    numeric = data[NUMERIC_FEATURES].apply(_signed_log1p)
    if CATEGORICAL_FEATURES:
        categorical = pd.get_dummies(data[CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES)
        features = pd.concat([numeric, categorical], axis=1)
    else:
        features = numeric
    scaled = RobustScaler().fit_transform(features)

    return data, scaled, features.columns.tolist()


def run_pca(df, n_components=2):
    data, scaled, feature_names = _build_scaled_features(df)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)

    components_df = pd.DataFrame(components, columns=[f"PC{i + 1}" for i in range(n_components)])
    components_df["device_id"] = data["device_id"]
    components_df["sensor_id"] = data["sensor_id"]
    components_df["site_id"] = data["site_id"]
    components_df["sensor_name"] = data["sensor_name"]
    components_df["unit"] = data["unit"]
    components_df["customer_account_name"] = data["customer_account_name"]
    components_df["payload_expected_time"] = data["payload_expected_time"]
    for feature in NUMERIC_FEATURES:
        components_df[feature] = data[feature]

    print("variance expliquee par composante:", pca.explained_variance_ratio_)
    print("variance expliquee cumulee:", pca.explained_variance_ratio_.cumsum())

    return components_df, pca, feature_names


def run_hdbscan(df, source="pca", min_cluster_size=15, min_samples=None):
    if source == "pca":
        component_columns = [c for c in df.columns if c.startswith("PC")]
        features = df[component_columns]
        identifiers = df
    elif source == "stats":
        data, scaled, _ = _build_scaled_features(df)
        features = scaled
        identifiers = data
    else:
        raise ValueError(f"source inconnu: {source!r}, attendu 'pca' ou 'stats'")

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(features)

    result = identifiers.copy()
    result["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"{n_clusters} clusters trouves, {n_noise} points consideres comme bruit")

    return result, clusterer


def profile_clusters(clustered_df, features=None):
    if features is None:
        features = NUMERIC_FEATURES

    total = len(clustered_df)
    rows = []
    for cluster_id, group in clustered_df.groupby("cluster"):
        row = {
            "cluster": cluster_id,
            "n": len(group),
            "pct": round(len(group) / total * 100, 1),
            "sensor_name_dominant": group["sensor_name"].value_counts().idxmax(),
            "customer_dominant": group["customer_account_name"].value_counts().idxmax(),
        }
        for feature in features:
            row[f"{feature}_mediane"] = group[feature].median()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def run_umap(df, source="stats", n_neighbors=15, min_dist=0.1, n_components=2):
    if source == "pca":
        component_columns = [c for c in df.columns if c.startswith("PC")]
        features = df[component_columns]
        identifiers = df
    elif source == "stats":
        data, scaled, _ = _build_scaled_features(df)
        features = scaled
        identifiers = data
    else:
        raise ValueError(f"source inconnu: {source!r}, attendu 'pca' ou 'stats'")

    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, init="random")
    embedding = reducer.fit_transform(features)

    result = identifiers.copy().reset_index(drop=True)
    for i in range(n_components):
        result[f"UMAP{i + 1}"] = embedding[:, i]

    return result, reducer


def plot_correlation_circle(pca, feature_names, axes=(0, 1)):
    pc_x, pc_y = axes
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(plt.Circle((0, 0), 1, facecolor="none", edgecolor="grey", linestyle="--"))

    for i, name in enumerate(feature_names):
        x, y = loadings[i, pc_x], loadings[i, pc_y]
        ax.arrow(0, 0, x, y, head_width=0.03, length_includes_head=True, color="steelblue")
        ax.text(x * 1.1, y * 1.1, name, fontsize=8, ha="center", va="center")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC{pc_x + 1} ({pca.explained_variance_ratio_[pc_x]:.1%})")
    ax.set_ylabel(f"PC{pc_y + 1} ({pca.explained_variance_ratio_[pc_y]:.1%})")
    ax.set_title("Cercle des correlations")
    fig.tight_layout()
    plt.show()
    return fig


def plot_pca_scatter(components_df, color_by="sensor_name", pca=None, x_col="PC1", y_col="PC2", clip_percentiles=None, symlog=False):
    main_categories = components_df[color_by].value_counts().head(len(CATEGORICAL_PALETTE)).index.tolist()
    color_map = dict(zip(main_categories, CATEGORICAL_PALETTE))

    fig, ax = plt.subplots(figsize=(9, 7))

    for category in main_categories:
        subset = components_df[components_df[color_by] == category]
        ax.scatter(
            subset[x_col], subset[y_col],
            s=20, alpha=0.6, color=color_map[category], label=category, edgecolors="none",
        )

    other = components_df[~components_df[color_by].isin(main_categories)]
    if not other.empty:
        ax.scatter(
            other[x_col], other[y_col],
            s=20, alpha=0.6, color=OTHER_COLOR, label="Autre", edgecolors="none",
        )

    if pca is not None and x_col == "PC1" and y_col == "PC2":
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    else:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    if clip_percentiles is not None:
        ax.set_xlim(np.nanpercentile(components_df[x_col], clip_percentiles))
        ax.set_ylim(np.nanpercentile(components_df[y_col], clip_percentiles))

    if symlog:
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")

    ax.set_title(f"Anomalies projetees ({x_col}/{y_col}, couleur = {color_by})")
    ax.legend(loc="best", fontsize=8, markerscale=1.5, frameon=False)
    fig.tight_layout()
    plt.show()
    return fig


def plot_feature_heatmap(df, feature, x_col="UMAP1", y_col="UMAP2", clip_percentiles=(1, 99)):
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_PALETTE)
    vmin, vmax = np.nanpercentile(df[feature], clip_percentiles)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        df[x_col], df[y_col],
        c=df[feature], cmap=cmap, vmin=vmin, vmax=vmax, s=20, alpha=0.8, edgecolors="none",
    )
    fig.colorbar(scatter, ax=ax, label=feature, extend="both")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col}/{y_col} colore par {feature}")
    fig.tight_layout()
    plt.show()
    return fig


def plot_all_feature_heatmaps(df, features=None, x_col="UMAP1", y_col="UMAP2"):
    if features is None:
        features = NUMERIC_FEATURES

    return [plot_feature_heatmap(df, feature, x_col=x_col, y_col=y_col) for feature in features]


def main():
    if not os.path.exists(STATS_OUTPUT):
        fetch_stats_from_db()
    if not os.path.exists(FEATURES_OUTPUT):
        merged = load_pca_dataset()
        dataset = add_derived_features(merged)
        dataset.to_csv(FEATURES_OUTPUT, index=False, encoding="utf-8")
        print(f"{len(dataset)} lignes sauvegardees dans {FEATURES_OUTPUT}")
    else:
        dataset = pd.read_csv(FEATURES_OUTPUT, encoding="utf-8")
        dataset = dataset[dataset["eval_prev_hour"] > 1000].reset_index(drop=True)
        components_df, pca, feature_names = run_pca(dataset)
        clustered, clusterer = run_hdbscan(components_df, source="pca", min_cluster_size=150, min_samples=50)
        clustered = add_device_info(clustered)
        clustered.to_csv(CLUSTERED_OUTPUT, index=False, encoding="utf-8")
        print(f"{len(clustered)} lignes sauvegardees dans {CLUSTERED_OUTPUT}")
        profile = profile_clusters(clustered)
        print(profile.to_string())
        plot_pca_scatter(clustered, color_by="cluster", x_col="PC1", y_col="PC2")
        plot_pca_scatter(clustered, color_by="cluster", x_col="amplitude_iqr", y_col="ecart_mediane", symlog=True)
        # plot_pca_scatter(clustered, color_by="cluster", x_col="relative_deviation", y_col="seuil_relatif", symlog=True)
        embedding, reducer = run_umap(dataset, source="stats")
        # meme dropna deterministe sur le meme dataset des deux cotes -> lignes alignees
        embedding["cluster"] = clustered["cluster"].values
        plot_pca_scatter(embedding, color_by="sensor_name", x_col="UMAP1", y_col="UMAP2")
        plot_pca_scatter(embedding, color_by="cluster", x_col="UMAP1", y_col="UMAP2")
        plot_all_feature_heatmaps(embedding)

if __name__ == "__main__":
    main()
