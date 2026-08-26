"""
L1 Detection -- ML scorer producing a fraud confidence score per transaction / per SIM-swap event.
Two separate models, not one merged model -- transactions and sim_swap_events differ in grain,
features, and label rate. Train on the config.TIME_SPLIT_DATE train split only.

Diturunin dari notebooks/02_L1_detection_baseline_and_model.ipynb. Model transaksi = Random
Forest hasil GridSearchCV; model SIM swap = Logistic Regression. Beda model bukan karena
selera: yang transaksi udah dibandingin sama LR dan RF manual di notebook, yang swap belum.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Hasil GridSearchCV di notebook 02 (scoring='average_precision', TimeSeriesSplit(n_splits=3)
# -- 5 fold nggak bisa, fold paling awal nol fraud). Ditulis di sini sebagai angka supaya
# training ulang nggak perlu ngulang grid search-nya; kalau fiturnya berubah, cari lagi.
TRANSACTION_MODEL_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 1,
    "class_weight": "balanced",  # fraud cuma 2.69% -- tanpa ini model milih diem aja
    "random_state": 42,
}

SIM_SWAP_MODEL_PARAMS: dict = {
    "class_weight": "balanced",  # fraud 10% di tabel swap
    "max_iter": 1000,            # default 100 sering belum konvergen
    "random_state": 42,
}


def train_transaction_model(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
) -> RandomForestClassifier:
    """Latih Random Forest transaksi pakai TRANSACTION_MODEL_PARAMS."""
    model = RandomForestClassifier(**TRANSACTION_MODEL_PARAMS)
    model.fit(train_features, train_labels)
    return model


def train_sim_swap_model(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
) -> LogisticRegression:
    """Latih Logistic Regression SIM swap pakai SIM_SWAP_MODEL_PARAMS."""
    model = LogisticRegression(**SIM_SWAP_MODEL_PARAMS)
    model.fit(train_features, train_labels)
    return model


def predict_score(model, features: pd.DataFrame) -> pd.Series:
    """
    Skor kepercayaan fraud (0-1) per baris, bukan kelas prediksi.

    L3 butuh angkanya, bukan 0/1: `policy_rules.score_band()` yang mutusin ambangnya, dan
    ambang itu beda per model. `predict()` udah nempelin ambang 0.5 diam-diam -- angka yang
    nggak pernah dipilih siapa-siapa.
    """
    return pd.Series(model.predict_proba(features)[:, 1], index=features.index)


def save_model(model, path: str) -> str:
    """Simpan model ke disk, balikin path-nya. Ditimpa kalau udah ada -- artefak, bukan dokumen."""
    joblib.dump(model, path)
    return path


def load_model(path: str):
    """Baca model yang udah disimpan. Dipakai layer lain biar nggak usah latih ulang."""
    return joblib.load(path)
