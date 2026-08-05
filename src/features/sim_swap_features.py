"""
L1 Detection -- feature engineering for SIM-swap fraud scoring.
Encodes the SOP-001 indicators as numeric features for src/detection/baseline_rules.py
and src/detection/model.py. Separate from transaction_features.py: different grain
(1 row per swap event, not per transaction) and no case_id links the two tables.
"""

import pandas as pd


def build_features(sim_swap_events: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: reason_stated == 'lost_phone' flag, distance_from_home_km, device_changes_last_12mo
    (from customers). hours_since_last_login_change is a proxy for SOP-001's 2-hour rule --
    restate that assumption here per CLAUDE.md data gaps, don't treat it as the real thing.
    """
    raise NotImplementedError
