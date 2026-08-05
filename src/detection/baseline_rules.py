"""
L1 Detection -- rule-based baseline scorer (no ML). Directly encodes SOP indicator counts.
This is the number src/detection/model.py has to beat -- CLAUDE.md Modeling bar says
baseline first, and every performance claim needs a comparison against it.
"""

import pandas as pd


def score_transactions_baseline(features: pd.DataFrame) -> pd.Series:
    """
    TODO: SOP-002-style rule score -- shared device/instrument count across accounts in 24h
    (config.SHARED_DEVICE_ACCOUNTS_MEDIUM/HIGH), night-hour flag, round-amount flag.
    """
    raise NotImplementedError


def score_sim_swap_baseline(features: pd.DataFrame) -> pd.Series:
    """
    TODO: count SOP-001 indicators per event (distance, reason_stated, login-change proxy,
    device_changes_last_12mo) and return the 0 / 1 / 2+ risk level from config thresholds.
    """
    raise NotImplementedError
