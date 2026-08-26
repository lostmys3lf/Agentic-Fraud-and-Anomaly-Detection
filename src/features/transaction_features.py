"""
L1 Detection -- feature engineering for transaction-level fraud scoring.
Feeds src/detection/baseline_rules.py and src/detection/model.py.

Two builders on purpose: the leakage patterns documented in CLAUDE.md (gopay + 01:00-04:59 +
amount divisible by 50,000; the DEV_SHARED_9F21A device; every promo row is label 1) make this
label trivially separable. Keeping legit and leaked features in separate functions makes it
possible to report both honestly instead of accidentally training on the generator's fingerprints.

Diturunin dari notebooks/02_L1_detection_baseline_and_model.ipynb (Bagian 1). Di notebook,
gabungan transaksi + customer udah kesimpen sebagai outputs/data_clean.csv; di sini
gabungannya dibikin ulang dari CSV mentah biar modulnya nggak nempel ke artefak notebook 01.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import config

# Kolom yang dianggap aman dipelajari. device_id nggak masuk (satu device dipakai 25 akun --
# itu sidik jari generator), begitu juga jam dan kelipatan amount (lihat build_leaked_features).
LEGIT_CATEGORICAL_COLS: list[str] = [
    "transaction_type",
    "payment_method",
    "channel",
    "location_city",
    "home_city",
    "segment",
]
LEGIT_NUMERIC_COLS: list[str] = [
    "tenure_months",
    "avg_monthly_topup_idr",
    "device_changes_last_12mo",
]

# Kolom customer yang ditarik ke tabel transaksi. Jangan tarik semuanya: full_name dan
# phone_number nggak nambah apa-apa buat model dan cuma bikin frame-nya berat.
CUSTOMER_COLS: list[str] = [
    "customer_id",
    "home_city",
    "segment",
    "tenure_months",
    "avg_monthly_topup_idr",
    "device_changes_last_12mo",
]

# Dua kolom one-hot yang menutup kebocoran promo: SETIAP baris promo_bundle_redeem dan
# SETIAP pembayaran promo_credit itu label 1 (CLAUDE.md, "Leakage warning"). Modelnya bukan
# belajar fraud, dia belajar "promo = fraud".
LEAKED_ONE_HOT_COLS: list[str] = [
    "transaction_type_promo_bundle_redeem",
    "payment_method_promo_credit",
]

LABEL_COL = "is_fraud_label"


def build_legit_features(transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature set considered safe to learn from.

    Balikannya masih mentah (belum di-encode): kolom kategorikal masih teks, plus
    `timestamp` dan `is_fraud_label` supaya yang manggil masih bisa nge-split berdasarkan
    waktu. Encoding-nya dipisah di `encode_features()` karena encoder cuma boleh di-fit di
    data latih, dan fungsi ini nggak tau mana train mana test.
    """
    merged = transactions.merge(customers[CUSTOMER_COLS], on="customer_id", how="left")

    keep = ["transaction_id", "customer_id", "timestamp"]
    keep += LEGIT_CATEGORICAL_COLS + LEGIT_NUMERIC_COLS
    if LABEL_COL in merged.columns:
        keep.append(LABEL_COL)
    return merged[keep]


def encode_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drop_promo_leak: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode the categorical columns and glue the numeric ones back on.

    Encoder-nya di-`fit` **cuma di train**, test cuma di-`transform`. Kalau di-fit ke
    dua-duanya, kategori yang cuma ada di Juni ikut kepelajari dan itu bocoran halus yang
    nggak keliatan di metrik manapun. `handle_unknown='ignore'` yang ngurus kategori baru
    di test -- dia jadi baris nol, bukan error.

    `drop_promo_leak=True` ngebuang dua kolom di LEAKED_ONE_HOT_COLS. Itu default-nya karena
    model yang direkomendasiin repo ini adalah yang tanpa bocoran; setel False cuma buat
    ngukur seberapa besar efek bocorannya (notebook 02 nyandingin dua-duanya).

    Balikannya `(X_train, X_test, encoder)`. Encoder-nya ikut dibalikin karena L1 nanti harus
    nge-transform satu transaksi baru pakai kategori yang sama persis.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[LEGIT_CATEGORICAL_COLS])
    names = encoder.get_feature_names_out(LEGIT_CATEGORICAL_COLS)

    def _encode(part: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.DataFrame(
            encoder.transform(part[LEGIT_CATEGORICAL_COLS]),
            columns=names,
            index=part.index,
        )
        return pd.concat([encoded, part[LEGIT_NUMERIC_COLS]], axis=1)

    X_train = _encode(train)
    X_test = _encode(test)

    if drop_promo_leak:
        leaked = [c for c in LEAKED_ONE_HOT_COLS if c in X_train.columns]
        X_train = X_train.drop(columns=leaked)
        X_test = X_test.drop(columns=leaked)

    return X_train, X_test, encoder


def transform_features(
    features: pd.DataFrame,
    encoder: OneHotEncoder,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Encode a batch of rows for SCORING, using an encoder that was already fitted on train.

    Bedanya sama `encode_features()`: fungsi itu buat waktu latih (dia yang nge-`fit`
    encoder-nya), yang ini buat waktu nyekor (encoder-nya udah jadi, tinggal dipakai).

    `feature_columns` itu urutan kolom persis waktu latih, dan urutan itu wajib sama.
    Kalau kolomnya di-*reindex*, model tetep jalan tapi nyocokin nilai ke kolom yang
    salah -- skornya ngaco tanpa error, dan itu bug yang paling susah keliatan.
    """
    encoded = pd.DataFrame(
        encoder.transform(features[LEGIT_CATEGORICAL_COLS]),
        columns=encoder.get_feature_names_out(LEGIT_CATEGORICAL_COLS),
        index=features.index,
    )
    combined = pd.concat([encoded, features[LEGIT_NUMERIC_COLS]], axis=1)
    return combined[feature_columns]


def build_leaked_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature set that DOES include the documented leakage patterns, for
    quantifying the leakage effect in evaluation only -- never for a model we'd recommend using.

    Tiga kolomnya juga kepakai buat baseline SOP-002 di `detection/baseline_rules.py`:
    aturan SOP-nya emang nyebut jam 01:00-05:00, amount bulat, dan device yang dipakai
    banyak akun. Bedanya cuma niat: sebagai aturan dia jujur, sebagai fitur model dia bocor.
    """
    hours = pd.to_datetime(transactions["timestamp"]).dt.hour

    device_account_count = (
        transactions.groupby("device_id")["customer_id"].transform("nunique")
    )

    return pd.DataFrame(
        {
            "hour": hours,
            "is_night": (
                (hours >= config.NIGHT_HOUR_START) & (hours < config.NIGHT_HOUR_END)
            ).astype(int),
            "is_round_amount": (
                transactions["amount_idr"] % config.ROUND_AMOUNT_DIVISOR == 0
            ).astype(int),
            "device_account_count": device_account_count,
        },
        index=transactions.index,
    )
