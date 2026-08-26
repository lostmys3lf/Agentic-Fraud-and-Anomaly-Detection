"""
L1 Detection -- feature engineering for SIM-swap fraud scoring.
Encodes the SOP-001 indicators as numeric features for src/detection/baseline_rules.py
and src/detection/model.py. Separate from transaction_features.py: different grain
(1 row per swap event, not per transaction) and no case_id links the two tables.

Diturunin dari notebooks/02_L1_detection_baseline_and_model.ipynb (Bagian 2).
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import config

# Empat indikator SOP-001, satu kolom flag 0/1 masing-masing. Urutannya sengaja dipatok:
# `indicator_count` itu jumlah dari keempat kolom ini, jadi nambah/ngurangin kolom di sini
# langsung ngubah arti baseline SOP-001 di detection/baseline_rules.py.
SOP001_FLAG_COLS: list[str] = [
    "flag_distance",
    "flag_lost_phone",
    "flag_login_change",
    "flag_device_change",
]

# Dipakai model ML, bukan baseline. `device_changes_last_12mo` sengaja dipakai mentah
# (bukan flag-nya) biar model bisa nemu batas sendiri, nggak dipaksa ikut batas SOP.
SWAP_CATEGORICAL_COLS: list[str] = ["reason_stated", "swap_location_city"]
SWAP_NUMERIC_COLS: list[str] = ["device_changes_last_12mo"]

LABEL_COL = "is_fraud_label"


def build_features(sim_swap_events: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Bikin 4 flag indikator SOP-001 + `indicator_count` per event swap.

    `device_changes_last_12mo` ditarik dari tabel customers -- satu-satunya indikator
    SOP-001 yang datanya nggak ada di tabel swap.

    Dua catatan yang harus ikut kebaca tiap kali fungsi ini dipakai (CLAUDE.md data gaps):
      - `hours_since_last_login_change` cuma **proksi** buat aturan "reset password atau
        login finansial dalam 2 jam" di SOP-001. Bukan aturan yang sebenernya.
      - Indikator "device belum pernah terdaftar" nggak bisa diitung sama sekali: nggak ada
        riwayat device per nasabah. Yang bisa dicek cuma alasan yang dinyatain
        (`reason_stated == 'lost_phone'`), jadi `flag_lost_phone` cuma separo indikator itu.
    """
    merged = sim_swap_events.merge(
        customers[["customer_id", "device_changes_last_12mo"]],
        on="customer_id",
        how="left",
    )

    merged["flag_distance"] = (
        merged["distance_from_home_km"] > config.SIM_SWAP_DISTANCE_KM_THRESHOLD
    ).astype(int)
    merged["flag_lost_phone"] = (merged["reason_stated"] == "lost_phone").astype(int)
    merged["flag_login_change"] = (
        merged["hours_since_last_login_change"] < config.SIM_SWAP_LOGIN_CHANGE_HOURS_THRESHOLD
    ).astype(int)
    merged["flag_device_change"] = (
        merged["device_changes_last_12mo"] > config.SIM_SWAP_DEVICE_CHANGES_12MO_THRESHOLD
    ).astype(int)

    merged["indicator_count"] = merged[SOP001_FLAG_COLS].sum(axis=1)
    return merged


def encode_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    One-hot `reason_stated` + `swap_location_city`, gabung sama kolom numeriknya.

    Sama kayak di transaction_features: encoder-nya di-`fit` cuma di train.

    Kolom flag-nya sengaja **nggak** ikut masuk ke fitur model. Flag itu jawaban aturan
    SOP-001, dan `flag_lost_phone` sendiri hampir sama persis sama label di data ini (semua
    20 swap penipuan nyatain `lost_phone`) -- masukin dia ke model sama aja ngasih contekan.
    Perbandingan yang jujur di sini: baseline aturan vs model dari kolom mentah.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[SWAP_CATEGORICAL_COLS])
    names = encoder.get_feature_names_out(SWAP_CATEGORICAL_COLS)

    def _encode(part: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(
            encoder.transform(part[SWAP_CATEGORICAL_COLS]),
            columns=names,
            index=part.index,
        )
        return pd.concat([part[SWAP_NUMERIC_COLS], encoded], axis=1)

    return _encode(train), _encode(test), encoder
