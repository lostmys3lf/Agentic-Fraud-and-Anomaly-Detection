"""
Generic evaluation utilities -- metrics and comparison tables. Shared by both the
transaction and SIM-swap models so results are reported consistently. Never report
accuracy alone -- CLAUDE.md Modeling bar (class imbalance is 2.69% and 10%).
"""

import pandas as pd


def evaluate(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict:
    """TODO: precision, recall, F1, PR-AUC at the given threshold."""
    raise NotImplementedError


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """TODO: build a DataFrame comparing baseline vs model, keyed by method name."""
    raise NotImplementedError
