"""
View layer (Streamlit) -- halaman batch: baca artefak yang UDAH ADA di `outputs/`.

Halaman ini sengaja read-only dan nggak pernah manggil pipeline. Alasannya ongkos, dan
angkanya bukan tebakan: satu `invoke()` itu ~120-160 ms di luar latensi LLM, `build_customer_profile()`
sendiri ~0.07 detik per nasabah, dan `report_l4` nambahin satu panggilan OpenAI per nasabah.
Datanya 1200 nasabah -- jalanin semuanya artinya 1200 panggilan API buat satu klik.

Jadi yang ditampilin di sini hasil run yang udah kesimpen dari notebook, bukan hitungan baru.
Kalau berkasnya belum ada, halamannya bilang notebook mana yang harus dijalanin dulu -- bukan
diam-diam ngitung sendiri, dan bukan ngelempar exception.
"""

from __future__ import annotations

import os
import sys

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import bootstrap  # noqa: F401,E402  -- sys.path repo + .env, harus paling duluan

import streamlit as st  # noqa: E402

import resources  # noqa: E402

st.title("Batch Data (raw)")
st.caption(
    "Synthetic dataset generated for this case study. Not real, customer, or production data."
)
st.caption("The Overview page summarizes pipeline_run.csv; this page holds the raw artifacts.")
st.info(
    "This page only reads artifacts already written to outputs/. It never re-runs the "
    "pipeline: one investigation costs one OpenAI call, and this dataset has 1200 customers."
)

for filename, description, notebook in resources.BATCH_SOURCES:
    st.subheader(filename)
    st.caption(description)

    frame = resources.load_batch_csv(filename)
    if frame is None:
        # Berkas batch di-gitignore, jadi "belum pernah dijalanin" itu keadaan normal buat
        # orang yang baru clone repo -- pesan, bukan error.
        st.warning(
            f"outputs/{filename} not found. Run {notebook} first to generate it, "
            "then reload this page."
        )
        st.divider()
        continue

    st.write(f"{len(frame)} rows.")

    # Cuma ngitung berapa baris per nilai yang UDAH ada di kolomnya. Nggak ada risk level
    # atau action yang dihitung ulang di sini -- itu urusan L3, dan sudah selesai.
    for column in ("action", "risk_level"):
        if column in frame.columns:
            counts = frame[column].value_counts().rename_axis(column).reset_index(
                name="customers"
            )
            st.dataframe(counts, width="content", hide_index=True)

    st.dataframe(frame, width="stretch", hide_index=True)
    st.divider()

st.caption(
    "One customer = one case file. The 25 accounts sharing DEV_SHARED_9F21A therefore "
    "produce 25 near-identical reports, and no field anywhere says they belong to one "
    "event -- see the Limitations page."
)
