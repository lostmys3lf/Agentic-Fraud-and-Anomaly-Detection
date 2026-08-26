"""
View layer (Streamlit) -- ngegambar hasil investigasi satu nasabah.

Strukturnya ngikutin prinsip drill-down: `render_headline()` naruh KEPUTUSANNYA di paling
atas (action, risk level, rationale, override), dan bukti per layer (L1-L4) tinggal di
expander yang dibuka kalau pembacanya mau ngecek. Jawaban dulu, bukti belakangan.

Modul ini cuma nulis apa yang udah diputusin layer lain. Aturan kerasnya sama persis kayak
`reporting/case_file.py`: nggak ada `if` di sini yang boleh ngubah `risk_level` atau
`recommended_action`. Milih warna banner berdasarkan action itu styling, bukan keputusan --
nilainya sendiri dicetak apa adanya.

Angka ambang nggak pernah ditulis di file ini -- semuanya dari `config.py`. Teks yang
muncul di layar bahasa Inggris; komentar dan docstring bahasa Indonesia.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config
from reporting.case_file import render_case_file_markdown

# Cerminan dari `reporting/narrative.py`: teks penanda waktu narasi gagal digenerate.
# Sengaja dibandingin ke teksnya, bukan ditebak dari ada-tidaknya API key -- narasi juga
# bisa gagal gara-gara timeout walau key-nya ada.
NARRATIVE_FALLBACK_TEXT = "Narrative not generated."


def render_error(state: dict) -> None:
    """Jalur `handle_error`: nasabah nggak bisa diproses, nggak ada case file yang ditulis."""
    st.error(f"Investigation stopped: {state['error']}")
    st.caption(
        "No case file was written for this customer. Pick an ID that exists in "
        "data/customers.csv, or use the manual field to test this path on purpose."
    )


def render_headline(state: dict) -> None:
    """
    Keputusan di paling atas: action sebagai banner berwarna status + ikon, lalu angka
    pendukungnya, lalu rationale. Warna + ikon + teks bareng-bareng (nggak pernah warna
    doang yang bawa makna).
    """
    decision = state["decision"]
    score = state.get("confidence_score")

    # Mapping action -> gaya banner. STYLING doang: teks action-nya dicetak verbatim.
    if decision.action == "Block":
        st.error(f"Recommended action: **{decision.action}**", icon="🚫")
    elif decision.action == "Escalate":
        st.warning(f"Recommended action: **{decision.action}**", icon="⚠️")
    else:
        st.success(f"Recommended action: **{decision.action}**", icon="✅")

    # Narasi LLM langsung di bawah banner, bukan dikubur di expander case file:
    # pembaca yang cuma butuh gambaran cepet berhenti di sini. Teksnya disalin apa
    # adanya dari case dict -- L4 yang bikin, headline cuma majang.
    narrative = state["case"].get("narrative_summary")
    if narrative == NARRATIVE_FALLBACK_TEXT:
        st.warning(
            "Narrative not generated. The LLM summary is optional: either no "
            "OPENAI_API_KEY was available or the API call failed. Every field below "
            "is complete without it."
        )
    elif narrative:
        st.markdown(f"**Summary:** {narrative}")
        st.caption(
            "Written by an LLM from the finished case file only. It never changes the "
            "risk level or the recommended action."
        )

    col_risk, col_l1, col_l2, col_score = st.columns(4)
    col_risk.metric("Risk level", decision.risk_level)
    col_l1.metric("L1 band", decision.l1_band)
    col_l2.metric("L2 risk level", decision.l2_risk_level)
    # Skor None itu nasabah nol transaksi -- bukan error, dan L3 bacanya BORDERLINE.
    col_score.metric("Model score (max)", "n/a" if score is None else f"{score:.4f}")

    st.markdown(f"**Why:** {decision.rationale}")

    if decision.override_applied:
        st.warning(
            "Override applied: a HIGH shared-device finding forces Block even when the "
            "model scored LOW. The model scores one transaction at a time and has no "
            "cross-account feature, so a low score there is silence, not a denial."
        )


def render_l1(state: dict) -> None:
    """L1 Detection: skor kepercayaan + cara bacanya (band-nya dari L3, bukan diitung ulang)."""
    decision = state["decision"]
    score = state.get("confidence_score")
    low, high = config.L1_SCORE_BANDS[decision.l1_source]

    col_score, col_band, col_source = st.columns(3)
    col_score.metric("Confidence score (max)", "n/a" if score is None else f"{score:.4f}")
    col_band.metric("L1 band", decision.l1_band)
    col_source.metric("Model", decision.l1_source)

    if low is None or high is None:
        st.caption(
            f"No score bands are configured for the '{decision.l1_source}' model, "
            "so this score cannot be read as LOW / BORDERLINE / HIGH."
        )
    else:
        st.caption(
            f"Bands for this model: LOW below {low}, HIGH at or above {high}. "
            "Both cuts are provisional placeholders, not tuned operating points."
        )

    if score is None:
        st.info(
            "This customer has no transactions to score, so the model never ran. "
            "A missing score is read as BORDERLINE, never as cleared."
        )

    # Skor per transaksi diambil max, bukan rata-rata: fraud itu satu transaksi buruk.
    st.caption("The score is the highest of this customer's per-transaction scores.")


def render_l2(state: dict, complaints: pd.DataFrame) -> None:
    """L2 Investigation: profil nasabah dari pattern_lookup + komplain terkait."""
    profile = state["profile"]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Transactions", profile["n_transactions"])
    col_b.metric("Night transactions", profile["n_night_transactions"])
    col_c.metric("Promo redemptions", profile["promo_redemptions_90d"])
    col_d.metric("SIM swap events", profile["n_sim_swaps"])

    st.markdown("**Shared device (SOP-002)**")
    if profile["shared_device_ids"]:
        st.write(
            "Devices shared with other accounts: "
            + ", ".join(profile["shared_device_ids"])
        )
        st.write(
            f"Largest account group on one device within "
            f"{config.SHARED_DEVICE_WINDOW_HOURS} hours: "
            f"{profile['max_accounts_per_shared_device']} accounts "
            f"(MEDIUM at {config.SHARED_DEVICE_ACCOUNTS_MEDIUM}, "
            f"HIGH at {config.SHARED_DEVICE_ACCOUNTS_HIGH})."
        )
    else:
        st.write("No device of this customer is shared with another account.")

    st.markdown("**Promo (SOP-003)**")
    st.write(
        f"{profile['promo_redemptions_90d']} redemptions in the last "
        f"{config.PROMO_WINDOW_DAYS} days "
        f"(MEDIUM at {config.PROMO_REDEMPTIONS_MEDIUM}, "
        f"HIGH at {config.PROMO_REDEMPTIONS_HIGH})."
    )

    st.markdown("**SIM swap inputs (SOP-001)**")
    # Nilai mentah doang. Indikator mana yang NYALA itu hasil L3 dan ditampilin di bagian
    # findings -- ngitung ulang di sini artinya bikin sumber kebenaran kedua.
    swap_inputs = {
        "Swap events": profile["n_sim_swaps"],
        "Max distance from home (km)": profile["max_swap_distance_km"],
        "Device changes, last 12 months": profile["device_changes_last_12mo"],
        "Reasons stated": ", ".join(profile["swap_reasons_stated"]) or "(none)",
        "Min hours since login change": (
            "(no swap on record)"
            if profile["min_hours_since_login_change"] is None
            else profile["min_hours_since_login_change"]
        ),
    }
    st.table(pd.DataFrame({"Value": [str(v) for v in swap_inputs.values()]},
                          index=list(swap_inputs)))
    st.caption(
        f"SOP-001 thresholds: distance above {config.SIM_SWAP_DISTANCE_KM_THRESHOLD} km, "
        f"login change within {config.SIM_SWAP_LOGIN_CHANGE_HOURS_THRESHOLD} hours, "
        f"more than {config.SIM_SWAP_DEVICE_CHANGES_12MO_THRESHOLD} device changes in 12 "
        "months. Which of them actually fired is decided in the findings section. "
        "hours_since_last_login_change is a proxy for the 2-hour rule, not the rule itself."
    )

    st.markdown(f"**Complaints ({len(complaints)})**")
    if complaints.empty:
        st.write("No complaint note is on record for this customer.")
    else:
        st.dataframe(complaints, width="stretch", hide_index=True)
    st.caption(
        "complaint_text is 47 repeated templates across the whole file, so read it as a "
        "weak signal, not as free text a person wrote."
    )


def render_l3(state: dict) -> None:
    """
    L3 Decision, bagian detailnya: tabel findings + sitasi SOP.

    Action, risk level, rationale dan override udah pindah ke `render_headline()` --
    di sini tinggal buktinya per SOP.
    """
    decision = state["decision"]

    st.markdown("**Findings (every SOP evaluated, including the quiet ones)**")
    rows = []
    for finding in decision.findings:
        rows.append(
            {
                "SOP": finding.sop_id,
                "Category": finding.category,
                "Risk level": finding.risk_level,
                "Fired": "yes" if finding.is_actionable else "no",
                "Indicators matched": "; ".join(finding.indicators_matched) or "(none)",
                "SOP citations": "; ".join(finding.citations) or "(none)",
                "Unverified": "; ".join(finding.unverified) or "(none)",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Citations are retrieved from the SOP corpus filtered to the matching document, "
        f"then kept above a similarity of {config.RAG_MIN_SIMILARITY}. On this "
        "4-document corpus that threshold removes nothing -- see the Limitations page."
    )


def render_l4(state: dict) -> None:
    """L4 Reporting: case file SOP-004 dirender apa adanya + status narasi."""
    case = state["case"]

    # Narasinya udah dipajang di headline; di sini cukup lewat section "Narrative
    # Summary" yang emang ada di dalam markdown case file-nya.
    st.markdown(render_case_file_markdown(case))
    st.caption(f"Saved to: {state['case_path']}")


def render_raw_state(as_json: dict) -> None:
    """Jejak audit mentah. Ketutup by default -- ini buat yang mau ngecek, bukan buat dibaca."""
    with st.expander("Raw pipeline state (audit trail)"):
        st.code(json.dumps(as_json, indent=2, default=str), language="json")
