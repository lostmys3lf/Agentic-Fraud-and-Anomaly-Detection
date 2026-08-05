"""
L1 Detection -- ML scorer producing a fraud confidence score per transaction / per SIM-swap event.
Two separate models, not one merged model -- transactions and sim_swap_events differ in grain,
features, and label rate. Train on the config.TIME_SPLIT_DATE train split only.
"""

import pandas as pd


def train_transaction_model(train_features: pd.DataFrame, train_labels: pd.Series):
    """TODO: e.g. LogisticRegression(class_weight='balanced') or RandomForestClassifier."""
    raise NotImplementedError


def train_sim_swap_model(train_features: pd.DataFrame, train_labels: pd.Series):
    """TODO: same pattern as train_transaction_model, fit separately on sim-swap features."""
    raise NotImplementedError


def predict_score(model, features: pd.DataFrame) -> pd.Series:
    """TODO: return predict_proba(...)[:, 1] -- a confidence score, not just the predicted class."""
    raise NotImplementedError
