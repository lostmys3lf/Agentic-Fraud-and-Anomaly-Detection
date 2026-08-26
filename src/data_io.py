"""
Generic data loading and parsing. Used by every layer (L1-L4) and by notebooks --
holds no fraud-specific logic, just consistent reads of the 4 raw CSVs.
"""

import pandas as pd

import config
from config import CUSTOMERS_CSV, TRANSACTIONS_CSV, SIM_SWAP_EVENTS_CSV, COMPLAINT_NOTES_CSV

# Nomor telepon di CSV kesimpen sebagai int64, jadi angka 0 di depannya ilang (11 digit,
# semuanya mulai dari 8). Dibalikin dengan zfill, bukan dengan "0" + s, biar tetep bener
# kalau suatu saat ada nomor yang panjangnya beda.
PHONE_NUMBER_DIGITS = 12


def _read_events(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Baca satu tabel event: parse kolom waktunya, urutin, reset index.

    Tiga tabel event dibaca persis sama caranya -- kalau urutannya beda-beda, hasil
    `time_based_split()` ikut beda dan itu nggak bakal keliatan sampai metriknya aneh.
    """
    df = pd.read_csv(path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    return df.sort_values(timestamp_col).reset_index(drop=True)


def load_customers() -> pd.DataFrame:
    """
    Load customers.csv.

    `phone_number` dibalikin jadi string 12 digit (0 di depan dikembaliin) -- lihat
    "Known data gaps" di CLAUDE.md. Dia identitas, bukan angka: nggak pernah dijumlahin,
    dan sebagai int dia salah waktu dicetak.
    """
    customers = pd.read_csv(CUSTOMERS_CSV)
    customers["phone_number"] = (
        customers["phone_number"].astype(str).str.zfill(PHONE_NUMBER_DIGITS)
    )
    return customers


def load_transactions() -> pd.DataFrame:
    """Load transactions.csv, sorted by timestamp (event table -- order matters for splitting)."""
    return _read_events(TRANSACTIONS_CSV)


def load_sim_swap_events() -> pd.DataFrame:
    """Load sim_swap_events.csv, sorted by timestamp."""
    return _read_events(SIM_SWAP_EVENTS_CSV)


def load_complaint_notes() -> pd.DataFrame:
    """Load complaint_notes.csv, sorted by timestamp."""
    return _read_events(COMPLAINT_NOTES_CSV)


def time_based_split(
    events: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split an event-level DataFrame into (train, test) using config.TIME_SPLIT_DATE.

    Never use a random split for transactions/sim_swap_events -- CLAUDE.md Modeling bar.
    Alasannya: split acak naruh transaksi Juni di data latih, jadi model "tau" masa depan
    dan angkanya keliatan bagus tanpa alasan.

    Kolom waktunya di-parse ulang di sini biar fungsinya tetep bener walau yang manggil
    ngasih DataFrame hasil `read_csv()` mentah (timestamp-nya masih string).
    """
    timestamps = pd.to_datetime(events[timestamp_col])
    split_date = pd.to_datetime(config.TIME_SPLIT_DATE)

    train = events[timestamps < split_date].copy()
    test = events[timestamps >= split_date].copy()
    return train, test
