"""
View layer (Streamlit) -- halaman limitasi.

Isinya angka yang UDAH DIUKUR di notebook 02-07 dan dicatat di CLAUDE.md. Nggak ada yang
dihitung ulang di halaman ini, dan nggak ada angka yang dikarang. Ambang apa pun diambil
dari `config.py`, bukan diketik ulang di sini -- kalau ambangnya digeser, halaman ini ikut
berubah sendiri.

Kenapa halaman ini ada sama sekali: tiga aturan SOP di repo ini cuma bisa dicek separo, dan
datanya nggak punya kasus abu-abu. Dashboard yang diam soal itu kelihatan lebih yakin
daripada buktinya, dan itu kesalahan paling mahal di laporan fraud.
"""

from __future__ import annotations

import os
import sys

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import bootstrap  # noqa: F401,E402  -- sys.path repo + .env, harus paling duluan

import streamlit as st  # noqa: E402

import config  # noqa: E402
from reporting.case_file import FIXED_LIMITATIONS  # noqa: E402

st.title("Limitations")
st.caption(
    "Every number on this page was measured in notebooks 02-07 and recorded in CLAUDE.md. "
    "Nothing here is recomputed at page load."
)

st.subheader("1. The data is synthetic")
st.markdown(
    """
- Generated for this case study: 1200 customers, four event tables spanning
  2026-04-01 to 2026-06-30. It is **not** real, customer, or production data, and no
  conclusion here transfers to a live telco portfolio.
- Labels exist per transaction (205 positives, 2.69%) and per SIM-swap event (20 positives,
  10.0%) only. There is **no customer-level or case-level label**, so "one customer = one
  case" is a decision this project made, not something the data states.
- There is no `case_id` linking transactions, swap events and complaints to one another.
- `complaint_text` is 47 repeated templates across 280 rows. It is not diverse free text.
"""
)

st.subheader("2. The label is leaked, and the leak is documented")
st.markdown(
    """
Four patterns make the label trivially separable. A model scoring near-perfect on this
dataset has learned the generator, not fraud:

- Every `promo_bundle_redeem` row and every `promo_credit` payment carries label 1.
- Every fraudulent topup is `e-wallet_gopay`, falls between 01:00 and 04:59, and has an
  amount divisible by 50000.
- All 20 fraudulent swaps state `lost_phone`, sit far from home, and have a near-zero
  `hours_since_last_login_change`.
- Exactly one device, `DEV_SHARED_9F21A`, is shared across 25 accounts. Every other
  device is 1:1.

The transaction model used by this dashboard is trained with the promo leak removed. On the
June test split it scores precision 0.262, recall 1.00, F1 0.415, PR-AUC 0.788 -- one alarm
in four is a real fraud, so an L1 HIGH is a reason to look, never a reason to block.
"""
)
st.warning(
    f"The L1 score bands (LOW below {config.L1_TX_SCORE_LOW}, HIGH at or above "
    f"{config.L1_TX_SCORE_HIGH}) are provisional placeholders -- the only guessed numbers "
    "left in config.py. They have not been read off a measured operating point yet."
)

st.subheader("3. Only 6 of the 12 L3 matrix boxes are reachable")
st.markdown(
    f"""
L3 crosses the L1 band (LOW / BORDERLINE / HIGH) with the L2 risk level
(NONE / LOW / MEDIUM / HIGH), which is 12 boxes. Measured over 119 customers and two L1
score scenarios:

- The whole L2 `LOW` column is dead. No SOP can emit LOW on this data: SOP-002 has no LOW
  band, SOP-003's LOW band is unreachable because the minimum promo count is already
  {config.PROMO_REDEMPTIONS_HIGH} (`PROMO_REDEMPTIONS_HIGH`), and SOP-001 maps a single
  indicator to MEDIUM because the SOP words that band as a mandatory hold.
- That kills `LOW` x `LOW`, one of only two Auto-Approve boxes.
- Exactly **1 of 119** customers ever reaches L2 MEDIUM. Everyone else is HIGH or NONE.
- The shared-device override fires 25 times at L1 LOW -- exactly the `DEV_SHARED_9F21A`
  accounts -- and 0 times at L1 HIGH, where the box is already Block.

**None of the three SOPs has a single borderline case in this data.** L3 is verified correct
against its own specification and is *not* verified to make good decisions on hard cases.
"""
)

st.subheader("4. One customer = one case file, so the ring is reported 25 times")
st.markdown(
    """
The case grain was fixed at one customer per report because `decide()` runs on one customer
profile. The cost is visible in the output: the 25 accounts sharing `DEV_SHARED_9F21A`
produce 25 near-identical case files -- one unique `risk_level`, one unique
`recommended_action`, and six variants of the triggering evidence that differ only in a
night-transaction count. **No field anywhere says those 25 belong to one event.**

The burst itself is one event, not a shared family device: all 118 transactions on that
device fall on 2026-05-16 between 01:07 and 04:12, all topups, all `e-wallet_gopay`, and
22 of the 25 accounts appear within a single hour.
"""
)

st.subheader("5. RAG here buys traceability, not precision")
st.markdown(
    f"""
The L2 retriever is a real vector store (Chroma, all-MiniLM-L6-v2, 384 dimensions) over a
corpus of 4 SOP documents split into 13 chunks. Measured against a 13-question evaluation
set:

- Vector recall@1 **9/13** against TF-IDF's **8/13** -- a one-question gap, i.e. noise.
- Vector recall@3 **13/13** against TF-IDF's **10/13**. Only that gap justifies the vector
  store. Dense embeddings are not inherently better on a corpus this small.
- Both retrievers fail the same way: "what risk level" questions land on *Risk Indicators*
  instead of *Escalation Levels*. That is a chunking problem, not a model problem.
- Correct chunks scored 0.284-0.573 and wrong chunks 0.251-0.702 -- almost complete overlap,
  so `RAG_MIN_SIMILARITY` ({config.RAG_MIN_SIMILARITY}) cannot filter on its own. It is
  paired with a `doc_id` filter, and that filter is what makes a citation trustworthy.
- In practice the threshold removes nothing: each SOP has only 4 chunks and the weakest
  still scores 0.236 (SOP-001), 0.438 (SOP-002) and 0.479 (SOP-003). With
  `RAG_TOP_K = {config.RAG_TOP_K}`, a citation list is effectively "3 of the 4 sections of
  the right document".

There is also **no generative model in the retrieval path**. The "generation" in these case
files is deterministic templating; the only LLM in this repo writes the optional narrative
paragraph in L4, from the finished case file.
"""
)

st.subheader("6. Half of each SOP is not computable on this data")
st.markdown(
    f"""
- SOP-001 "device not previously registered": there is no per-customer device history.
- SOP-001's {config.SIM_SWAP_LOGIN_CHANGE_HOURS_THRESHOLD}-hour rule is applied through
  `hours_since_last_login_change`, which is a **proxy** for "password reset or financial
  login within 2 hours", not the event itself.
- SOP-002 "same e-wallet id or card number": `payment_method` is a category only.
  `device_id` is the sole usable shared fingerprint.
- SOP-003 "cancel and re-subscribe within 7 days" and "no usage between redemptions":
  there is no subscription or usage table.
- `PROMO_WINDOW_DAYS = {config.PROMO_WINDOW_DAYS}` on a 90-day dataset means the promo
  window filter removes nothing. The code is correct and the step is inert.

Half-verifiable indicators still count toward the SOP-001 total. Excluding them would leave
only 2 of 4 indicators reachable and quietly disable the block the SOP orders -- so the gap
is recorded in each finding's `unverified` list instead, and printed in every case file.
"""
)

st.subheader("7. Carried into every case file")
for line in FIXED_LIMITATIONS:
    st.markdown(f"- {line}")

st.subheader("8. Known cosmetic defect")
st.markdown(
    """
`indicators_matched` renders the device list as a Python repr, for example
`device ['DEV_SHARED_9F21A']`, which leaks into the report text. It is an L3 formatting bug,
not a dashboard bug, and it is shown here verbatim rather than patched in the view layer --
the dashboard is not allowed to rewrite what L3 decided.
"""
)
