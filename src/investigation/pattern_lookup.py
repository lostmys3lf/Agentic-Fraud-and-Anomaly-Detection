"""
L2 Investigation -- historical pattern lookup over the event tables.

The other half of L2. sop_retriever.py answers "what does policy say?"; this module
answers "what has this customer actually done?". L3 needs both: a policy clause with no
evidence decides nothing, and evidence with no clause cannot be cited.

Scope note (CLAUDE.md data gaps -- do not invent values for these):
  - There is no case_id linking transactions, swap events and complaints, so every
    lookup here is keyed on customer_id and nothing else.
  - SOP-002's "same e-wallet id or card number" does not exist in the data.
    payment_method is a category only; device_id is the one usable shared fingerprint.
  - SOP-003's "cancel and re-subscribe within 7 days" and "no usage between redemptions"
    are not computable -- there is no subscription or usage table. Count redemptions and
    say plainly that the cancel-recreate half of the rule is unverified.
  - complaint_text is 47 repeated templates across 280 rows. Match it as categories,
    not as free text, and treat related_to_fraud_case as a proxy label.

Timestamps arrive as strings from read_csv unless parsed -- parse before comparing.
"""

import pandas as pd

import config


def find_shared_device_accounts(
    transactions: pd.DataFrame,
    device_id: str,
    window_hours: int = config.SHARED_DEVICE_WINDOW_HOURS,
) -> pd.DataFrame:
    """
    SOP-002 section 2: how many distinct accounts used this device inside one window?

    TODO:
      1. Keep only rows for this device_id.
      2. For each transaction, count distinct customer_id within window_hours of it
         (the SOP says "within a 24-hour window", not "in the whole dataset" -- a device
         seen on 3 accounts across 3 months is not the same finding).
      3. Return one row per account with its transaction count and first/last timestamp,
         sorted so the busiest account is on top.
      4. The caller compares the account count against config.SHARED_DEVICE_ACCOUNTS_
         MEDIUM / _HIGH -- do not hardcode 2 or 3 in here.

    Hint: `df[df[col] == value]`, `pd.Timedelta(hours=...)`, `sort_values()`,
          `groupby().agg()`, `nunique()`
    """
    raise NotImplementedError


def count_promo_redemptions(
    transactions: pd.DataFrame,
    customer_id: str,
    window_days: int = config.PROMO_WINDOW_DAYS,
) -> int:
    """
    SOP-003 section 2: promo bundle redemptions by this customer in the last window_days.

    Careful -- promo_bundle_redeem is the leaked column from CLAUDE.md: every such row is
    label 1. That makes it useless as an L1 model feature, but it is still the correct
    thing to count here, because SOP-003 literally counts redemptions. Different jobs.
    Note the distinction in the notebook rather than reusing the L1 feature function.

    TODO:
      1. Filter to this customer and transaction_type == 'promo_bundle_redeem'.
      2. Keep rows inside the window (the dataset spans 90 days total, so with the
         default window this is every row -- state that, do not pretend it is a filter).
      3. Return the count as a plain int.

    Hint: `str.eq()`, `&`, `df.loc[]`, `len()` or `shape[0]`
    """
    raise NotImplementedError


def get_customer_complaints(
    complaint_notes: pd.DataFrame,
    customer_id: str,
) -> pd.DataFrame:
    """
    Complaints filed by this customer -- supporting evidence for the SOP-004 report.

    TODO:
      1. Filter to this customer, sort by timestamp.
      2. Return complaint_id, timestamp, channel, complaint_text, related_to_fraud_case.
      3. Do not aggregate the text or compute a sentiment -- 47 templates cannot support
         that. The complaint_id list is what the report cites.

    Hint: `df[df[col] == value]`, `sort_values()`, column selection with a list
    """
    raise NotImplementedError


def build_customer_profile(
    customer_id: str,
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    sim_swap_events: pd.DataFrame,
    complaint_notes: pd.DataFrame,
) -> dict:
    """
    Bundle every L2 finding for one customer into the evidence dict L3 consumes.

    This is the contract between L2 and L3, so decide the keys here and keep them
    stable. Suggested shape -- each value is either a number L3 compares to a config
    threshold, or a list of ids the L4 report cites:

        {
          "customer_id": ...,
          "home_city": ..., "tenure_months": ..., "device_changes_last_12mo": ...,
          "n_transactions": ..., "n_night_transactions": ...,
          "promo_redemptions_90d": ...,
          "shared_device_ids": [...], "max_accounts_per_shared_device": ...,
          "n_sim_swaps": ..., "max_swap_distance_km": ...,
          "complaint_ids": [...], "transaction_ids": [...], "swap_event_ids": [...],
        }

    TODO:
      1. Pull the customer row from `customers` (raises if the id is unknown -- fail
         loudly, a silent empty profile would become a silently wrong decision).
      2. Call the three functions above, plus this customer's SIM swap events.
      3. Return the dict. No risk level and no action here -- scoring is L3's job
         (src/decision/), and mixing them makes the decision untraceable.

    Hint: `df.loc[df[col] == value]`, `.iloc[0]`, `.to_dict()`, `tolist()`
    """
    raise NotImplementedError
