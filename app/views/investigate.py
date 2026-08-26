"""
View layer (Streamlit) -- halaman detail: investigasi satu nasabah, L1 sampai L4.

Struktur halamannya jawaban-dulu-bukti-belakangan: banner keputusan + rationale di atas
(`render_headline`), bukti per layer di expander di bawahnya. Ini kebalikan dari draft
pertama halaman ini, yang nyetak L1->L4 full berurutan dan ngubur keputusannya di tengah.

Dua jalur masuk:
  - langsung: pilih nasabah di sidebar, pencet Investigate;
  - dari worklist Overview: tombol "Open case" naruh id-nya di
    `st.session_state["pending_investigation"]`, halaman ini yang mulung dan langsung
    jalanin (satu klik "Open case" = niat investigasi yang sama sama mencet tombolnya).

ONGKOS: `report_l4` manggil `attach_narrative()`, jadi satu `invoke()` = satu panggilan
OpenAI. Streamlit ngejalanin ulang seluruh script tiap interaksi widget. Dua pengaman:
  1. pipeline cuma jalan di cabang tombol / hand-off, bukan di alur utama script;
  2. hasil di-cache per `customer_id` di `st.session_state` (lihat `resources.investigate`).
Counter di sidebar itu ukuran ongkosnya, bukan hiasan.
"""

from __future__ import annotations

import os
import sys

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import streamlit as st  # noqa: E402

import render  # noqa: E402
import resources  # noqa: E402

st.title("Investigate a Customer")
st.caption(
    "Synthetic dataset generated for this case study. "
    "Flow: L1 Detection -> L2 Investigation -> L3 Decision -> L4 Reporting."
)

# --- hand-off dari worklist Overview ------------------------------------------------
# pop(), bukan get(): sekali dipungut langsung abis, biar kunjungan manual berikutnya
# nggak tiba-tiba jalanin pipeline sendiri.
pending = st.session_state.pop("pending_investigation", None)
if pending is not None:
    # Preset widget SEBELUM widget-nya dibikin -- setelahnya Streamlit nolak.
    st.session_state["manual_mode"] = False
    st.session_state["picked_customer_id"] = pending

# --- sidebar: pilih nasabah ---------------------------------------------------------
with st.sidebar:
    st.header("Customer")

    # Dua jalur masuk. Dropdown buat id yang beneran ada; kolom manual buat nguji jalur
    # `handle_error` di graph, yang nggak pernah kepakai di batch karena semua id di situ
    # emang ada di customers.csv.
    manual_mode = st.checkbox("Enter a customer ID manually", key="manual_mode")
    if manual_mode:
        customer_id = st.text_input(
            "Customer ID", key="manual_customer_id", placeholder="CUST11146"
        ).strip()
        st.caption("An ID that is not in data/customers.csv takes the error path.")
    else:
        customer_id = st.selectbox(
            "Customer ID", resources.get_customer_ids(), key="picked_customer_id"
        )

    run_clicked = st.button("Investigate", type="primary", key="investigate_button")

    # Diisi di akhir script, bukan di sini: kalau diisi sekarang, angkanya ketinggalan
    # satu langkah setiap kali tombolnya baru ditekan.
    run_counter_slot = st.empty()

    st.divider()
    st.caption(
        "One investigation calls the OpenAI narrative once. Repeating the same customer "
        "is served from this session's cache and costs nothing."
    )

# --- jalanin pipeline: cuma pas tombol ditekan atau hand-off dari worklist ----------
if run_clicked or pending is not None:
    if not customer_id:
        st.warning("Enter a customer ID first.")
    else:
        try:
            with st.spinner(f"Running L1 -> L4 for {customer_id}..."):
                resources.investigate(customer_id)
        # Sengaja `Exception` telanjang DI SINI doang, dan cuma di lapisan view: dashboard
        # yang mati sama halaman traceback nggak ngasih apa-apa ke pembacanya. Traceback
        # lengkapnya tetep dicetak ke stderr biar yang ngejalanin app bisa lihat.
        except Exception as exc:  # noqa: BLE001
            st.error(f"Pipeline failed: {resources.format_exception(exc)}")

# --- tampilkan hasil ----------------------------------------------------------------
# Diambil dari cache pakai nasabah yang LAGI dipilih. Ini juga yang bikin panel lama nggak
# nyangkut: ganti nasabah = kunci cache-nya beda = panelnya kosong lagi sampai ditekan.
state = resources.get_result_cache().get(customer_id) if customer_id else None

if state is None:
    st.info("Pick a customer in the sidebar, then press Investigate.")
elif state.get("error"):
    render.render_error(state)
else:
    render.render_headline(state)
    st.divider()

    # Bukti per layer, ketutup by default: headline udah jawab "apa keputusannya",
    # expander jawab "kenapa" buat yang mau ngecek. Detail on demand.
    with st.expander("Detection -- model score (L1)"):
        render.render_l1(state)
    with st.expander("Investigation -- customer profile & patterns (L2)"):
        render.render_l2(state, resources.customer_complaints(state["customer_id"]))
    with st.expander("Decision detail -- findings & SOP citations (L3)"):
        render.render_l3(state)
    with st.expander("Case file -- SOP-004 report & narrative (L4)"):
        render.render_l4(state)

    render.render_raw_state(resources.result_as_json(state))

run_counter_slot.caption(
    f"Pipeline runs this session: {resources.get_run_count()} | "
    f"cached customers: {len(resources.get_result_cache())}"
)
