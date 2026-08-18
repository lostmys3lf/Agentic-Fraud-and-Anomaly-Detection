"""
L3 Decision -- cross L1 against L2, attach the SOP citation, emit the action.

policy_rules.py answers "how bad is this, per SOP?". This module answers "so what do we
do?", and it is the only place in the repo allowed to say Auto-Approve, Escalate or Block.

The rule this layer implements (decided 2026-08-10, option C -- two keys, plus one narrow
override):

  - L1 and L2 are crossed in a matrix, not resolved by whoever scores higher. Both LOW
    -> Auto-Approve. Both high -> Block. **They disagree -> Escalate, never
    Auto-Approve.** Disagreement is not noise to be averaged away; it is the case a human
    should see.
  - One override, for shared-device findings only: an L2 HIGH from SOP-002 forces Block
    even when L1 scored LOW. Reason, and it is structural rather than a preference -- the
    L1 model scores one transaction from that transaction's own columns and has no
    cross-account feature at all, so it *cannot* see that a device is shared by 25
    accounts. Its low score is silence, not a denial.
  - Promo findings deliberately get no override. L1 does see promo, in the worst possible
    way: promo_bundle_redeem and promo_credit are the documented leak and every such row
    carries label 1. So if L1 scores a promo case LOW, the disagreement is informative and
    a human should look at it -- exactly what Escalate is for.

Everything a Decision carries must survive json.dumps() -- L4 writes it into the SOP-004
case file. No numpy scalars, no DataFrames, no timestamps that are not strings.

Why decide() takes the Chroma collection and not a ready-made list of hits: which policy
document to retrieve from depends on which finding fired, so retrieval cannot happen
before the rules have run. That ordering is also the fix for a measured problem --
notebook 03 found correct chunks scoring 0.284-0.573 and wrong ones 0.251-0.702, so
config.RAG_MIN_SIMILARITY cannot filter on its own. Pairing it with a doc_id filter is
what makes the citation trustworthy.

Import-order note, and it is load-bearing on Windows: importing this module pulls in
sop_retriever, which imports chromadb. If pandas is imported into the process before
chromadb, collection.upsert() kills the interpreter with exit 139 and no traceback. Any
notebook that touches this module must import chromadb above pandas in its bootstrap.
"""

from dataclasses import dataclass, replace

import config
from decision import policy_rules
from decision.policy_rules import RiskFinding
from investigation import sop_retriever


# --- vocabulary ---------------------------------------------------------------
# The only three actions this project emits. Spelled exactly as SOP-004 expects them,
# because L4 prints these strings verbatim into the "recommended action" field.

ACTION_AUTO_APPROVE = "Auto-Approve"
ACTION_ESCALATE = "Escalate"
ACTION_BLOCK = "Block"

# Categories allowed to override a LOW L1 score. Keep this list short and justify every
# entry in the docstring above -- an override is a claim that the model is structurally
# blind to something, which is true for cross-account patterns and not much else.
OVERRIDE_CATEGORIES = (policy_rules.CATEGORY_SHARED_DEVICE,)

# category -> the policy file that defines it.
_CATEGORY_DOC_IDS: dict[str, str] = {
    policy_rules.CATEGORY_SIM_SWAP: "SOP-001",
    policy_rules.CATEGORY_SHARED_DEVICE: "SOP-002",
    policy_rules.CATEGORY_PROMO_ABUSE: "SOP-003",
}

# What to search the policy corpus with, per category. Policy language, on purpose: the
# embedding has never seen "CUST10061" and "25" appears in no SOP, so the customer id and
# the raw numbers would only add noise. Each phrase names "escalation levels" explicitly
# because of a failure measured in notebook 03 -- queries phrased as "what risk level"
# land on the *Risk Indicators* section instead, in both retrievers. That is a chunking
# problem, and asking for the clause by name is the cheap way around it.
_CATEGORY_QUERY_TERMS: dict[str, str] = {
    policy_rules.CATEGORY_SIM_SWAP: (
        "SIM swap verification risk indicators and escalation levels"
    ),
    policy_rules.CATEGORY_SHARED_DEVICE: (
        "same device fingerprint or payment instrument used across multiple accounts, "
        "risk indicators and escalation levels"
    ),
    policy_rules.CATEGORY_PROMO_ABUSE: (
        "promo bundle redemption abuse thresholds and escalation levels"
    ),
}

# The 3 x 4 matrix: (L1 band, L2 risk level) -> action. Written out in full rather than
# derived from a rule, because every box is a decision someone has to be able to argue
# with. Three invariants are fixed and not open for re-litigation:
#   - Auto-Approve requires BOTH sides quiet. It appears in exactly two boxes.
#   - Disagreement always yields Escalate, never Auto-Approve.
#   - Block needs an L2 HIGH. An L1 HIGH on its own never blocks -- that is why the HIGH
#     cut in config.py is sized as an Escalate-grade threshold, not a block-grade one.
# BORDERLINE is Escalate across the whole row on purpose: it is also where an unscored
# case (confidence_score=None) lands, and "we do not know" must never auto-approve.
_MATRIX: dict[tuple[str, str], str] = {
    (policy_rules.L1_BAND_LOW, policy_rules.RISK_NONE): ACTION_AUTO_APPROVE,
    (policy_rules.L1_BAND_LOW, policy_rules.RISK_LOW): ACTION_AUTO_APPROVE,
    (policy_rules.L1_BAND_LOW, policy_rules.RISK_MEDIUM): ACTION_ESCALATE,
    (policy_rules.L1_BAND_LOW, policy_rules.RISK_HIGH): ACTION_ESCALATE,

    (policy_rules.L1_BAND_BORDERLINE, policy_rules.RISK_NONE): ACTION_ESCALATE,
    (policy_rules.L1_BAND_BORDERLINE, policy_rules.RISK_LOW): ACTION_ESCALATE,
    (policy_rules.L1_BAND_BORDERLINE, policy_rules.RISK_MEDIUM): ACTION_ESCALATE,
    (policy_rules.L1_BAND_BORDERLINE, policy_rules.RISK_HIGH): ACTION_ESCALATE,

    # L1 HIGH + L2 NONE is Escalate, not Block: with the measured precision of 0.262 the
    # model is wrong about roughly three alarms in four, and L2 NONE here often means "no
    # rule could check this pattern" rather than "there is no pattern".
    (policy_rules.L1_BAND_HIGH, policy_rules.RISK_NONE): ACTION_ESCALATE,
    (policy_rules.L1_BAND_HIGH, policy_rules.RISK_LOW): ACTION_ESCALATE,
    (policy_rules.L1_BAND_HIGH, policy_rules.RISK_MEDIUM): ACTION_ESCALATE,
    (policy_rules.L1_BAND_HIGH, policy_rules.RISK_HIGH): ACTION_BLOCK,
}


@dataclass(frozen=True)
class Decision:
    """
    One case, decided. This is the hand-off to L4: every SOP-004 field is either here or
    reachable from the findings inside it.

    l1_band and l2_risk_level are both kept alongside the final risk_level on purpose. The
    matrix is only auditable if the two inputs it consumed are still visible afterwards --
    otherwise "Escalate" gives a reviewer nothing to disagree with.
    """

    customer_id: str
    confidence_score: float | None      # from L1, None when L1 has not run
    l1_source: str                      # which L1 model produced it, key of config.L1_SCORE_BANDS
    l1_band: str                        # policy_rules.L1_BAND_*
    l2_risk_level: str                  # combine_risk_levels() over the findings
    risk_level: str                     # final level after the matrix and any override
    action: str                         # one of the ACTION_* constants
    findings: tuple[RiskFinding, ...]   # every SOP evaluated, including the quiet ones
    rationale: str                      # one sentence a reviewer can argue with
    override_applied: bool = False      # True when OVERRIDE_CATEGORIES changed the action


def sop_doc_id_for(category: str) -> str:
    """
    A policy_rules CATEGORY_* -> the doc_id of the SOP that defines it ("SOP-002").

    An unknown category raises rather than returning None. Returning None and letting
    retrieval run unfiltered is the tempting option and the wrong one: it would cite
    whichever chunk happened to be nearest, from any document, and an empty or misdirected
    citation list looks exactly like "policy had nothing to say".
    """
    if category not in _CATEGORY_DOC_IDS:
        raise KeyError(
            f"unknown finding category {category!r}; expected one of "
            f"{sorted(_CATEGORY_DOC_IDS)}"
        )
    return _CATEGORY_DOC_IDS[category]


def attach_sop_citations(
    finding: RiskFinding,
    collection,
    top_k: int = config.RAG_TOP_K,
) -> RiskFinding:
    """
    Look up the policy text behind a finding and return a copy carrying its citations.

    Both retrieval keys are used together, and that is not belt-and-braces: the similarity
    threshold alone cannot separate right from wrong hits on this corpus (measured in
    notebook 03, correct chunks 0.284-0.573 against wrong ones 0.251-0.702 -- they overlap
    almost completely), and the doc_id filter is what closes that gap.

    A finding with nothing above the threshold comes back with citations still empty, and
    decide() records that in the rationale. Never write a citation the retriever did not
    return: a fabricated SOP reference is worse than none, because it is the part a
    reviewer trusts without checking.
    """
    query = _CATEGORY_QUERY_TERMS[finding.category]
    doc_id = sop_doc_id_for(finding.category)

    hits = sop_retriever.retrieve_vector(query, collection, top_k=top_k, doc_id=doc_id)
    citations = tuple(
        hit.chunk.citation for hit in hits if hit.score >= config.RAG_MIN_SIMILARITY
    )

    # RiskFinding is frozen, so this is a new object and the original stays intact in the
    # log -- that is the whole point of freezing it.
    return replace(finding, citations=citations)


def decide(
    confidence_score: float | None,
    profile: dict,
    collection,
    l1_source: str = "transaction",
) -> Decision:
    """
    The whole of L3 in one call: L1 score + L2 profile + SOP corpus -> one Decision.

    confidence_score arrives as a parameter because L1 has not been migrated into src/
    yet. When it is, this signature does not change -- the caller wires the model in.

    l1_source has to travel with the score: one number is meaningless without knowing
    which model's calibration it should be read against (config.L1_SCORE_BANDS). A case
    that has both a swap event and topups genuinely has two L1 scores at two different
    grains -- this signature takes one, so the caller decides which. Today only
    'transaction' has a trained model behind it.
    """
    # Every SOP is evaluated, and the quiet ones are kept. "SOP-001 was checked and found
    # nothing" is a required line in an audit trail -- drop it and a reader cannot tell
    # checked-and-clean from never-checked.
    findings = [
        policy_rules.evaluate_sim_swap(profile),
        policy_rules.evaluate_shared_device(profile),
        policy_rules.evaluate_promo_abuse(profile),
    ]

    l2_risk_level = policy_rules.combine_risk_levels(findings)
    l1_band = policy_rules.score_band(confidence_score, l1_source)

    action = _MATRIX[(l1_band, l2_risk_level)]

    # The override. A cross-account finding at HIGH blocks regardless of what L1 said,
    # because the L1 model has no cross-account feature and therefore cannot have
    # disagreed -- it was never looking.
    override_applied = False
    override_categories = [
        f.category
        for f in findings
        if f.category in OVERRIDE_CATEGORIES and f.risk_level == policy_rules.RISK_HIGH
    ]
    if override_categories and action != ACTION_BLOCK:
        action = ACTION_BLOCK
        override_applied = True

    # Citations only for the findings that fired; a RISK_NONE finding has no clause to cite
    # and a retrieval per quiet SOP would be three wasted queries per customer.
    cited: list[RiskFinding] = []
    for finding in findings:
        if finding.is_actionable:
            cited.append(attach_sop_citations(finding, collection))
        else:
            cited.append(finding)

    fired = [f for f in cited if f.is_actionable]
    uncited = [f.sop_id for f in fired if not f.citations]

    parts = [
        f"L1 {l1_band} (score={confidence_score}, model={l1_source})",
        f"L2 {l2_risk_level}",
    ]
    if fired:
        clauses = "; ".join(
            f"{f.sop_id} {f.risk_level} [{', '.join(f.citations) or 'no citation'}]"
            for f in fired
        )
        parts.append(f"findings: {clauses}")
    else:
        parts.append("no SOP fired")
    if override_applied:
        parts.append(
            f"override: {', '.join(sorted(set(override_categories)))} HIGH forces "
            f"{ACTION_BLOCK} regardless of the L1 band"
        )
    if uncited:
        # Said out loud rather than dropped: an uncited level is a weaker claim than a
        # cited one, and the reviewer has to be able to see which they are reading.
        parts.append(f"uncited findings: {', '.join(uncited)}")
    rationale = f"{' | '.join(parts)} -> {action}."

    return Decision(
        customer_id=profile["customer_id"],
        confidence_score=confidence_score,
        l1_source=l1_source,
        l1_band=l1_band,
        l2_risk_level=l2_risk_level,
        # The final level is the L2 level, not a new number: the L1 band is a model
        # confidence and does not define a policy risk level. The override can raise it,
        # which is the one case where the action outranks the evidence level.
        risk_level=(
            policy_rules.RISK_HIGH if override_applied else l2_risk_level
        ),
        action=action,
        findings=tuple(cited),
        rationale=rationale,
        override_applied=override_applied,
    )
