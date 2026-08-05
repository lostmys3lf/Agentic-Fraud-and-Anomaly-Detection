"""
Generic data loading and parsing. Used by every layer (L1-L4) and by notebooks --
holds no fraud-specific logic, just consistent reads of the 4 raw CSVs.
"""

import pandas as pd

from config import CUSTOMERS_CSV, TRANSACTIONS_CSV, SIM_SWAP_EVENTS_CSV, COMPLAINT_NOTES_CSV


def load_customers() -> pd.DataFrame:
    """
    Load customers.csv.
    TODO: cast phone_number to string and restore the leading zero (see CLAUDE.md data gaps).
    """
    raise NotImplementedError


def load_transactions() -> pd.DataFrame:
    """Load transactions.csv, sorted by timestamp (event table -- order matters for splitting)."""
    raise NotImplementedError


def load_sim_swap_events() -> pd.DataFrame:
    """Load sim_swap_events.csv, sorted by timestamp."""
    raise NotImplementedError


def load_complaint_notes() -> pd.DataFrame:
    """Load complaint_notes.csv, sorted by timestamp."""
    raise NotImplementedError


def time_based_split(events: pd.DataFrame, timestamp_col: str = "timestamp") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split an event-level DataFrame into (train, test) using config.TIME_SPLIT_DATE.
    Never use a random split for transactions/sim_swap_events -- CLAUDE.md Modeling bar.
    TODO: rows with timestamp_col < TIME_SPLIT_DATE go to train, the rest to test.
    """
    raise NotImplementedError
