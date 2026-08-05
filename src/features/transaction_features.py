"""
L1 Detection -- feature engineering for transaction-level fraud scoring.
Feeds src/detection/baseline_rules.py and src/detection/model.py.

Two builders on purpose: the leakage patterns documented in CLAUDE.md (gopay + 01:00-04:59 +
amount divisible by 50,000; the DEV_SHARED_9F21A device; every promo row is label 1) make this
label trivially separable. Keeping legit and leaked features in separate functions makes it
possible to report both honestly instead of accidentally training on the generator's fingerprints.
"""

import pandas as pd


def build_legit_features(transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature set considered safe to learn from.
    TODO: encode transaction_type/payment_method/channel/location_city, join customer
    attributes (home_city, segment, tenure_months, avg_monthly_topup_idr, device_changes_last_12mo).
    Excludes device_id and any hour-of-day/amount-modulo features -- those are leaked.
    """
    raise NotImplementedError


def build_leaked_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature set that DOES include the documented leakage patterns, for
    quantifying the leakage effect in evaluation only -- never for a model we'd recommend using.
    TODO: hour-of-day, amount_idr % 50000 == 0 flag, device_id shared-account count.
    """
    raise NotImplementedError
