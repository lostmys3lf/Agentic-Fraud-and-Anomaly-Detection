"""
Generic evaluation utilities -- metrics and comparison tables. Shared by both the
transaction and SIM-swap models so results are reported consistently. Never report
accuracy alone -- CLAUDE.md Modeling bar (class imbalance is 2.69% and 10%).

Kenapa nggak ada accuracy di sini sama sekali: nebak "bukan fraud" buat semua baris
transaksi udah dapet accuracy 97.3%, dan angka itu nggak salah -- cuma nggak ada artinya.
"""

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict:
    """
    Precision, recall, F1 di satu ambang + PR-AUC di semua ambang.

    `y_score` harus skor kontinu (probabilitas model atau jumlah indikator baseline),
    bukan kelas 0/1. Precision/recall/F1 diitung dari `y_score >= threshold`, sementara
    PR-AUC baca skornya langsung -- dia ngukur mutu urutannya, lepas dari ambang manapun.
    Itu satu-satunya angka di tabel ini yang nggak berubah waktu ambangnya digeser.

    `zero_division=0` dipasang biar ambang yang kelewat tinggi (nggak ada baris yang
    ke-flag) balik 0.0, bukan warning plus nan yang nyusup ke tabel perbandingan.
    """
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Beberapa hasil `evaluate()` jadi satu tabel, satu baris per metode.

    `results` bentuknya {nama metode: hasil evaluate()}. Nama metodenya jadi index, jadi
    tabelnya kebaca sebagai perbandingan -- dan sebuah klaim performa tanpa baris pembanding
    emang nggak lolos standar repo ini (CLAUDE.md Modeling bar).
    """
    return pd.DataFrame(results).T
