"""
L3 Decision -- SOP thresholds read against L2 evidence. One SOP, one function.

Where this sits: L1 produces a confidence score, L2 produces evidence
(pattern_lookup.build_customer_profile) and policy text (sop_retriever). This module has
exactly one job -- turn evidence into a risk level per SOP, naming the indicators that
fired. It never chooses an action; that is decide.py.

Keeping those two apart is what makes the layer auditable: afterwards you can still tell
whether a number came from the evidence or from the rule that read it. Merge them and a
wrong Auto-Approve has no single line to blame.

Every function here is pure -- dict in, RiskFinding out. No pandas, no Chroma, no file
I/O. Two reasons: the same finding has to be reproducible from a stored profile alone,
and changing a threshold has to be testable without touching the data layer.

Profile keys each rule needs (all from build_customer_profile):
  SOP-002  max_accounts_per_shared_device, shared_device_ids, n_night_transactions
  SOP-003  promo_redemptions_90d
  SOP-001  n_sim_swaps, max_swap_distance_km, device_changes_last_12mo,
           swap_reasons_stated, min_hours_since_login_change

Measured properties of this dataset that no threshold can fix (both from notebook 04 --
do not re-derive them, do report them in the L3 evaluation):
  - Only 15 customers ever redeem a promo and the minimum is already 4, which is
    PROMO_REDEMPTIONS_HIGH. SOP-003 therefore rates 100% of them HIGH; the LOW and
    MEDIUM bands are unreachable. The rule is right, the data has no grey cases.
  - DEV_SHARED_9F21A is the only shared device at all: 25 accounts, 22 of them inside a
    single hour, every other device 1:1. SOP-002 has exactly one positive to fire on.
"""

from dataclasses import dataclass

import config


# --- vocabulary ---------------------------------------------------------------
# Strings, not an Enum: these values get written into a JSON case file by L4, and a
# report that has to be de-serialised to be read is not a report.

RISK_NONE = "NONE"
RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"

# Ordered weakest to strongest. combine_risk_levels() uses the position, so the only
# correct way to add a level later is to insert it at the right place in this list.
RISK_LEVEL_ORDER = [RISK_NONE, RISK_LOW, RISK_MEDIUM, RISK_HIGH]

# One category per SOP document. decide.sop_doc_id_for() maps these to a doc_id, which is
# what keeps retrieval pointed at the right policy file.
CATEGORY_SIM_SWAP = "sim_swap"
CATEGORY_SHARED_DEVICE = "shared_device"
CATEGORY_PROMO_ABUSE = "promo_abuse"

# Bands for the L1 score. Deliberately not the same words as the risk levels -- "the model
# is confident" and "policy says this is high risk" are different claims and the report
# must not blur them.
L1_BAND_LOW = "LOW"
L1_BAND_BORDERLINE = "BORDERLINE"
L1_BAND_HIGH = "HIGH"


@dataclass(frozen=True)
class RiskFinding:
    """
    One SOP's verdict on one customer. The data contract for the whole layer, so keep the
    fields stable: decide.py crosses risk_level with the L1 band, L4 prints the rest.

    Frozen on purpose. A finding is a record of what the evidence said at decision time;
    if a later step needs to add something (citations), it builds a new one with
    dataclasses.replace() and the original stays intact in the log.
    """

    sop_id: str                    # "SOP-002"
    category: str                  # one of the CATEGORY_* constants
    risk_level: str                # one of RISK_LEVEL_ORDER
    indicators_matched: tuple[str, ...]   # human-readable, one per rule that fired
    unverified: tuple[str, ...]    # rule halves this data cannot check -- see below
    evidence: dict                 # the raw numbers behind the level, for L4
    citations: tuple[str, ...] = ()  # filled later by decide.attach_sop_citations()

    @property
    def is_actionable(self) -> bool:
        """True once the level is above NONE. Sugar, but it keeps decide.py readable."""
        return self.risk_level != RISK_NONE


def score_band(confidence_score: float | None, l1_source: str = "transaction") -> str:
    """
    L1's number -> one of the L1_BAND_* strings.

    l1_source names the model that produced the score, because the cut points belong to a
    model's calibration and not to this repo -- see config.L1_SCORE_BANDS.

    Decisions taken here, all three deliberate:

      1. A missing score is BORDERLINE, never LOW. None means L1 never ran for this
         customer, and reading that as "the model cleared them" would let an unscored
         case reach Auto-Approve. BORDERLINE forces Escalate everywhere in the matrix,
         which is the honest answer to "we do not know".
      2. The LOW cut is exclusive and the HIGH cut is inclusive: score < low -> LOW,
         score >= high -> HIGH. A score sitting exactly on the low cut is therefore
         BORDERLINE, not LOW. LOW is the only band that can lead to Auto-Approve, so the
         boundary case goes to the side that keeps a human in the loop.
      3. A model with no trained thresholds raises. Borrowing another model's cut points
         would read one model's probabilities against another's calibration and leave no
         trace that it happened.
    """
    if l1_source not in config.L1_SCORE_BANDS:
        raise KeyError(
            f"unknown l1_source {l1_source!r}; expected one of "
            f"{sorted(config.L1_SCORE_BANDS)}"
        )

    low, high = config.L1_SCORE_BANDS[l1_source]
    if low is None or high is None:
        raise ValueError(
            f"no score bands configured for l1_source {l1_source!r}: that model has not "
            f"been trained, so its scores cannot be banded"
        )

    if confidence_score is None:
        return L1_BAND_BORDERLINE
    if confidence_score >= high:
        return L1_BAND_HIGH
    if confidence_score < low:
        return L1_BAND_LOW
    return L1_BAND_BORDERLINE


def evaluate_shared_device(profile: dict) -> RiskFinding:
    """
    SOP-002 -- one device fingerprint across several accounts inside a 24-hour window.

    The account count is read off the profile, never recounted here: the "within 24 hours"
    definition already lives in find_shared_device_accounts() and a second copy would
    eventually disagree with the first.

    Night volume is recorded as an indicator but is not allowed to raise the level on its
    own. Reason is specific to this data, not a general rule: all 118 transactions on
    DEV_SHARED_9F21A already fall inside NIGHT_HOUR_START..NIGHT_HOUR_END, so on the only
    shared device in the dataset the two indicators always fire together. Letting the
    second one lift a band would be the same evidence counted twice.
    """
    n_accounts = profile["max_accounts_per_shared_device"]
    n_night = profile["n_night_transactions"]

    # Strongest band first: 25 accounts satisfies the MEDIUM test too, so whichever branch
    # is tested first wins.
    if n_accounts >= config.SHARED_DEVICE_ACCOUNTS_HIGH:
        risk_level = RISK_HIGH
    elif n_accounts >= config.SHARED_DEVICE_ACCOUNTS_MEDIUM:
        risk_level = RISK_MEDIUM
    else:
        # A device used by one account is not a finding. SOP-002 has no LOW band.
        risk_level = RISK_NONE

    indicators: list[str] = []
    if risk_level != RISK_NONE:
        indicators.append(
            f"device {profile['shared_device_ids']} shared by {n_accounts} accounts "
            f"within {config.SHARED_DEVICE_WINDOW_HOURS}h"
        )
        if n_night > 0:
            indicators.append(
                f"{n_night} transactions between "
                f"{config.NIGHT_HOUR_START:02d}:00 and {config.NIGHT_HOUR_END:02d}:00"
            )

    unverified: list[str] = []
    if risk_level != RISK_NONE:
        unverified.append(
            "same e-wallet id / card number across accounts (payment_method is a "
            "category only, device_id is the whole fingerprint here)"
        )

    return RiskFinding(
        sop_id="SOP-002",
        category=CATEGORY_SHARED_DEVICE,
        risk_level=risk_level,
        indicators_matched=tuple(indicators),
        unverified=tuple(unverified),
        evidence={
            "max_accounts_per_shared_device": n_accounts,
            "shared_device_ids": profile["shared_device_ids"],
            "n_night_transactions": n_night,
            "window_hours": config.SHARED_DEVICE_WINDOW_HOURS,
            "threshold_medium": config.SHARED_DEVICE_ACCOUNTS_MEDIUM,
            "threshold_high": config.SHARED_DEVICE_ACCOUNTS_HIGH,
        },
    )


def evaluate_promo_abuse(profile: dict) -> RiskFinding:
    """
    SOP-003 -- promo bundle redemptions inside a 90-day period.

    Two decisions worth knowing before you change a number here.

    Zero redemptions is RISK_NONE, not RISK_LOW. SOP-003's LOW band starts at one
    redemption; a customer who never touched a promo has no finding, not a small one.

    A count at or above PROMO_REDEMPTIONS_HIGH is rated HIGH even though the written rule
    is "4 or more redemptions *with* cancel-and-recreate" and the cancel-recreate half is
    not computable here (no subscription table, no usage table). The alternative -- capping
    at MEDIUM until a human confirms -- was rejected because MEDIUM is defined as exactly
    three redemptions, so filing an eight-redemption customer there would misstate the
    evidence rather than soften the verdict. The missing half is recorded in `unverified`
    instead, which keeps the report honest without lying about the count.

    This is not an edge case: in this dataset it decides 100% of the promo customers.
    """
    n_redemptions = profile["promo_redemptions_90d"]

    if n_redemptions >= config.PROMO_REDEMPTIONS_HIGH:
        risk_level = RISK_HIGH
    elif n_redemptions >= config.PROMO_REDEMPTIONS_MEDIUM:
        risk_level = RISK_MEDIUM
    elif n_redemptions > 0:
        # PROMO_REDEMPTIONS_LOW is the *ceiling* of the LOW band (1-2 per 90d), not a
        # floor, so it is not used as a >= cut. "> 0" is an existence check, not a policy
        # number, which is why it is inline rather than in config.py.
        risk_level = RISK_LOW
    else:
        risk_level = RISK_NONE

    indicators: list[str] = []
    if n_redemptions > 0:
        indicators.append(
            f"{n_redemptions} promo redemptions within {config.PROMO_WINDOW_DAYS} days"
        )

    unverified: list[str] = []
    if risk_level == RISK_HIGH:
        unverified.append("cancel-and-resubscribe within 7 days (no subscription table)")
        unverified.append("no usage between redemptions (no usage table)")

    return RiskFinding(
        sop_id="SOP-003",
        category=CATEGORY_PROMO_ABUSE,
        risk_level=risk_level,
        indicators_matched=tuple(indicators),
        unverified=tuple(unverified),
        evidence={
            "promo_redemptions_90d": n_redemptions,
            "window_days": config.PROMO_WINDOW_DAYS,
            "threshold_medium": config.PROMO_REDEMPTIONS_MEDIUM,
            "threshold_high": config.PROMO_REDEMPTIONS_HIGH,
        },
    )


def evaluate_sim_swap(profile: dict) -> RiskFinding:
    """
    SOP-001 -- indicator *counting*, not a single number. 0 = clear, 1 = verify and hold,
    2 or more = block and escalate.

    Three decisions, all of which change outcomes:

      1. A customer with zero swap events exits with RISK_NONE. SOP-001 is a procedure for
         verifying a swap request; with no request there is nothing to verify, which is a
         different statement from "a request that passed verification".
      2. Half-checkable indicators still count toward the total, and the gap is recorded in
         `unverified`. Two of the four cannot be fully checked here -- there is no
         per-customer device history, so "device not previously registered" is unknowable,
         and hours_since_last_login_change is a proxy for the 2-hour rule rather than the
         rule. Excluding them would cap SOP-001 at two reachable indicators and make the
         2+ band nearly unreachable, which silently disables the block the SOP orders.
      3. One indicator maps to RISK_MEDIUM, not RISK_LOW. SOP-001 words that band as an
         action -- "verify and hold 1 h" -- and a mandatory hold is not a small finding.
         Consequence to report: SOP-001 never produces RISK_LOW.
    """
    n_swaps = profile["n_sim_swaps"]

    if n_swaps == 0:
        return RiskFinding(
            sop_id="SOP-001",
            category=CATEGORY_SIM_SWAP,
            risk_level=RISK_NONE,
            indicators_matched=(),
            unverified=(),
            evidence={"n_sim_swaps": 0},
        )

    max_distance = profile["max_swap_distance_km"]
    min_hours = profile["min_hours_since_login_change"]
    device_changes = profile["device_changes_last_12mo"]
    reasons = profile["swap_reasons_stated"]

    # SOP-001 words these as "more than 100 km" and "more than 2 device changes", so the
    # comparison is strict; the login rule is "within 2 hours", so that one is inclusive.
    far_from_home = max_distance > config.SIM_SWAP_DISTANCE_KM_THRESHOLD
    lost_phone_claimed = "lost_phone" in reasons
    login_changed_fast = (
        min_hours is not None
        and min_hours <= config.SIM_SWAP_LOGIN_CHANGE_HOURS_THRESHOLD
    )
    many_device_changes = device_changes > config.SIM_SWAP_DEVICE_CHANGES_12MO_THRESHOLD

    indicators: list[str] = []
    if far_from_home:
        indicators.append(
            f"swap {max_distance} km from home city "
            f"(> {config.SIM_SWAP_DISTANCE_KM_THRESHOLD})"
        )
    if lost_phone_claimed:
        indicators.append("reason stated is lost_phone")
    if login_changed_fast:
        indicators.append(
            f"login/password change {min_hours}h from the swap "
            f"(<= {config.SIM_SWAP_LOGIN_CHANGE_HOURS_THRESHOLD})"
        )
    if many_device_changes:
        indicators.append(
            f"{device_changes} device changes in 12 months "
            f"(> {config.SIM_SWAP_DEVICE_CHANGES_12MO_THRESHOLD})"
        )

    unverified: list[str] = []
    if lost_phone_claimed:
        unverified.append(
            "device not previously registered: no per-customer device history exists, so "
            "only the stated reason could be checked"
        )
    if login_changed_fast:
        unverified.append(
            "hours_since_last_login_change is a proxy for the SOP-001 2-hour rule, not "
            "the rule itself"
        )

    n_indicators = len(indicators)
    if n_indicators >= 2:
        risk_level = RISK_HIGH
    elif n_indicators == 1:
        risk_level = RISK_MEDIUM
    else:
        risk_level = RISK_NONE

    return RiskFinding(
        sop_id="SOP-001",
        category=CATEGORY_SIM_SWAP,
        risk_level=risk_level,
        indicators_matched=tuple(indicators),
        unverified=tuple(unverified),
        evidence={
            "n_sim_swaps": n_swaps,
            "n_indicators": n_indicators,
            # The four values compared, not just the count: "2 indicators" with no numbers
            # behind it is not evidence a reviewer can check.
            "max_swap_distance_km": max_distance,
            "swap_reasons_stated": reasons,
            "min_hours_since_login_change": min_hours,
            "device_changes_last_12mo": device_changes,
            "threshold_distance_km": config.SIM_SWAP_DISTANCE_KM_THRESHOLD,
            "threshold_login_change_hours": config.SIM_SWAP_LOGIN_CHANGE_HOURS_THRESHOLD,
            "threshold_device_changes_12mo": config.SIM_SWAP_DEVICE_CHANGES_12MO_THRESHOLD,
        },
    )


def combine_risk_levels(findings: list[RiskFinding]) -> str:
    """
    Several RiskFindings -> the one risk level for the case.

    The strongest level present wins, compared by position in RISK_LEVEL_ORDER rather than
    by string: alphabetically "HIGH" < "LOW", which is exactly backwards.

    Levels are never added, averaged or weighted. Two MEDIUMs from two different SOPs are
    not a HIGH -- each band is defined inside its own document against its own indicators,
    and no clause anywhere makes them add up. If a case like that deserves escalation,
    that rule belongs in decide.py where it is visible, not hidden in arithmetic here.
    """
    if not findings:
        # Normal for most customers, not an error.
        return RISK_NONE

    return max(findings, key=lambda f: RISK_LEVEL_ORDER.index(f.risk_level)).risk_level
