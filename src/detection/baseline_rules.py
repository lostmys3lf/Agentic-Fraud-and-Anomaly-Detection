"""
L1 Detection -- rule-based baseline scorer (no ML). Directly encodes SOP indicator counts.
This is the number src/detection/model.py has to beat -- CLAUDE.md Modeling bar says
baseline first, and every performance claim needs a comparison against it.

Dua-duanya balikin **jumlah indikator yang nyala**, bukan 0/1. Alasannya biar dia bisa
diperlakuin sama persis kayak skor model: satu angka, terus dibandingin sama satu ambang
lewat `detection.evaluate.evaluate()`. Ambang SOP-nya ada di config, bukan di sini.
"""

import pandas as pd

import config


def score_transactions_baseline(features: pd.DataFrame) -> pd.Series:
    """
    Baseline SOP-002 per transaksi: jumlah indikator yang nyala (0-3).

    Input-nya hasil `features.transaction_features.build_leaked_features()` --
    butuh `device_account_count`, `is_night`, `is_round_amount`.

    Indikatornya: device dipakai >= SHARED_DEVICE_ACCOUNTS_HIGH akun, transaksi jam
    01:00-05:00, dan amount kelipatan 50.000.

    **Ini nggak diturunin di notebook 02** -- Bagian 1 langsung ke model, baseline aturannya
    cuma dibikin buat SIM swap. Jadi angka pembandingnya belum pernah diukur.
    """
    shared_device = (
        features["device_account_count"] >= config.SHARED_DEVICE_ACCOUNTS_HIGH
    ).astype(int)
    return shared_device + features["is_night"] + features["is_round_amount"]


def score_sim_swap_baseline(features: pd.DataFrame) -> pd.Series:
    """
    Baseline SOP-001 per event swap: jumlah indikator yang nyala (0-4).

    Input-nya hasil `features.sim_swap_features.build_features()`, yang udah nyediain
    kolom `indicator_count`. Fungsi ini sengaja tetep ada walau isinya cuma ngambil satu
    kolom: yang manggil mestinya nggak perlu tau nama kolomnya, dan aturan "2 indikator =
    HIGH" (config.SIM_SWAP_INDICATORS_HIGH) jadi punya satu tempat buat dipasang.
    """
    return features["indicator_count"]
