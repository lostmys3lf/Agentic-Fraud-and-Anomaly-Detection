"""
View layer (Streamlit) -- halaman pendaratan: ringkasan hasil batch yang udah ada.

Prinsipnya drill-down: halaman ini jawab "keadaannya gimana" (KPI, distribusi action,
sebaran skor, worklist urut risiko) TANPA nyuruh user milih apa-apa dulu. Detail satu
nasabah baru dibuka di halaman Investigate, lewat tombol "Open case".

Sumber datanya `outputs/pipeline_run.csv` doang -- hasil run notebook 07 yang udah
kesimpen. Halaman ini nggak pernah manggil pipeline: 1200 nasabah = 1200 panggilan
OpenAI buat satu klik. Kalau berkasnya belum ada, halamannya bilang notebook mana yang
harus dijalanin, bukan diam-diam ngitung sendiri.

Semua angka KPI dihitung dari CSV-nya, nggak ada yang diketik langsung. Ambang skor L1
diambil dari `config.py`.
"""

from __future__ import annotations

import os
import sys

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
import resources  # noqa: E402

# --- konstanta tampilan -------------------------------------------------------------
# Warna status (bukan warna seri): Block = critical, Escalate = warning, Auto-Approve =
# good. Trio ini lolos validasi CVD (deltaE terburuk 11.3) -- jangan diganti asal.
# Amber-nya di bawah kontras 3:1 di latar terang, mitigasinya label nilai langsung di
# atas tiap batang + worklist tabel di bawah chart, jadi warnanya nggak pernah sendirian
# bawa makna. Urutannya urutan keparahan, dipakai juga buat sortir worklist -- ini urutan
# TAMPILAN doang, keputusannya sendiri udah final dari L3.
ACTION_ORDER = ["Block", "Escalate", "Auto-Approve"]
ACTION_COLORS = {"Block": "#d03b3b", "Escalate": "#fab219", "Auto-Approve": "#0ca30c"}
SCORE_BAR_COLOR = "#3987e5"   # satu hue buat magnitude (histogram), bukan warna status
RULE_COLOR = "#6b6a67"        # garis ambang: tinta redup, bukan warna data

st.title("Agentic Fraud & Anomaly Investigation")
st.caption(
    "Synthetic dataset generated for this case study -- 1200 customers, 2026-04-01 to "
    "2026-06-30. Not real, customer, or production data."
)

frame = resources.load_batch_csv("pipeline_run.csv")

if frame is None:
    st.warning(
        "outputs/pipeline_run.csv not found, so there is no saved batch to summarize. "
        "Run notebook 07 first to generate it, then reload this page. "
        "You can still investigate any single customer on the Investigate page."
    )
    if st.button("Go to Investigate", type="primary"):
        st.switch_page("views/investigate.py")
    st.stop()

# Baris error (nasabah gagal diproses) nggak punya action -- dipisah, bukan dibuang diam.
failed = frame[frame["error"].notna()] if "error" in frame.columns else frame.iloc[0:0]
scored = frame.drop(failed.index)
action_counts = scored["action"].value_counts()

# --- KPI ----------------------------------------------------------------------------
col_total, col_block, col_escalate, col_approve = st.columns(4)
col_total.metric("Customers reviewed", len(scored))
col_block.metric("Block", int(action_counts.get("Block", 0)))
col_escalate.metric("Escalate", int(action_counts.get("Escalate", 0)))
col_approve.metric("Auto-Approve", int(action_counts.get("Auto-Approve", 0)))

st.caption(
    f"From the saved batch in outputs/pipeline_run.csv ({len(frame)} of 1200 customers; "
    "the rest were never queued -- use the Investigate page for any of them). "
    "One event dominates this batch: a single shared device, DEV_SHARED_9F21A, links 25 "
    "accounts in one night -- see the Limitations page."
)
if not failed.empty:
    st.warning(f"{len(failed)} customer(s) in the batch failed to process -- see Batch Data.")

# --- dua chart: distribusi action + sebaran skor L1 ---------------------------------
chart_left, chart_right = st.columns(2)

with chart_left:
    st.markdown("**Recommended actions**")
    counts = (
        scored["action"]
        .value_counts()
        .reindex(ACTION_ORDER, fill_value=0)
        .rename_axis("action")
        .reset_index(name="customers")
    )
    base = alt.Chart(counts).encode(
        x=alt.X("action:N", sort=ACTION_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("customers:Q", title="Customers"),
    )
    bars = base.mark_bar(
        size=48, cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        color=alt.Color(
            "action:N",
            scale=alt.Scale(domain=ACTION_ORDER, range=[ACTION_COLORS[a] for a in ACTION_ORDER]),
            legend=None,  # identitas udah ada di sumbu x, legend cuma dobel
        )
    )
    # Label nilai langsung -- ini mitigasi wajib buat amber yang kontrasnya rendah.
    labels = base.mark_text(dy=-8).encode(text="customers:Q")
    st.altair_chart(bars + labels, width="stretch")

with chart_right:
    st.markdown("**Model confidence scores (L1, transaction model)**")
    low, high = config.L1_SCORE_BANDS["transaction"]
    hist = (
        alt.Chart(scored.dropna(subset=["confidence_score"]))
        .mark_bar(color=SCORE_BAR_COLOR)
        .encode(
            x=alt.X(
                "confidence_score:Q",
                bin=alt.Bin(step=0.05),
                title="Max score per customer",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("count()", title="Customers"),
        )
    )
    cuts = pd.DataFrame(
        {"cut": [low, high], "label": [f"LOW < {low}", f"HIGH >= {high}"]}
    )
    rules = alt.Chart(cuts).mark_rule(strokeDash=[4, 3], color=RULE_COLOR).encode(x="cut:Q")
    # Posisi y label pakai nilai piksel tetap DI DALAM area plot -- dy negatif dari
    # baseline bikin teksnya kepotong tepi atas chart.
    rule_labels = alt.Chart(cuts).mark_text(
        align="left", dx=4, color=RULE_COLOR
    ).encode(x="cut:Q", y=alt.value(12), text="label")
    st.altair_chart(hist + rules + rule_labels, width="stretch")
    st.caption(
        "The two cuts come from config.py and are provisional placeholders, "
        "not tuned operating points."
    )

# --- worklist: siapa yang harus dilihat duluan --------------------------------------
st.subheader("Worklist")
st.caption(
    "Every reviewed customer, most urgent first: Block, then Escalate, then Auto-Approve, "
    "highest model score first within each group. Values are shown exactly as decided -- "
    "this page recomputes nothing."
)

# Urutan keparahan buat SORTIR TAMPILAN doang. Keputusannya nggak disentuh.
severity = {action: rank for rank, action in enumerate(ACTION_ORDER)}
worklist = scored.copy()
worklist["_severity"] = worklist["action"].map(severity)
worklist = worklist.sort_values(
    ["_severity", "confidence_score"], ascending=[True, False]
)

display = worklist[
    ["customer_id", "action", "risk_level", "confidence_score", "l1_band",
     "l2_risk_level", "override_applied"]
].rename(
    columns={
        "customer_id": "Customer",
        "action": "Action",
        "risk_level": "Risk level",
        "confidence_score": "Model score",
        "l1_band": "L1 band",
        "l2_risk_level": "L2 risk",
        "override_applied": "Override",
    }
)
display["Model score"] = display["Model score"].round(3)
st.dataframe(display, width="stretch", hide_index=True, height=400)

# Jalur drill-down-nya: pilih dari worklist -> halaman Investigate. Pakai selectbox +
# tombol (bukan klik baris dataframe) biar jalurnya kelihatan dan bisa dites.
col_pick, col_open = st.columns([3, 1], vertical_alignment="bottom")
picked = col_pick.selectbox(
    "Open a case from the worklist",
    worklist["customer_id"].tolist(),
    key="worklist_pick",
)
if col_open.button("Open case", type="primary", key="open_case_button"):
    # Halaman Investigate yang jalanin pipeline-nya (dan yang bayar satu panggilan
    # OpenAI kalau belum ada di cache) -- halaman ini cuma ngoper id-nya.
    st.session_state["pending_investigation"] = picked
    st.switch_page("views/investigate.py")

st.caption("Raw batch artifacts (all three CSVs) are on the Batch Data page.")
