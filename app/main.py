"""
View layer (Streamlit) -- entry point + router. Jalanin dari root repo:

    streamlit run app/main.py

File ini nggak punya isi halaman sama sekali; dia cuma nyusun navigasinya. Struktur
halamannya ngikutin prinsip drill-down (overview dulu, detail belakangan):

    Overview      ringkasan batch yang udah ada: KPI, chart, worklist urut risiko
    Investigate   satu nasabah, L1-L4 -- keputusan jadi headline, bukti di expander
    Batch Data    artefak CSV mentah dari outputs/ (buat yang mau angka mentahnya)
    Limitations   batasan terukur dataset dan sistemnya

Kenapa pakai `st.navigation` (MPA v2), bukan folder `pages/`: label halaman di sidebar
bisa dikontrol. Dengan folder `pages/`, halaman utama muncul di sidebar sebagai "main" --
nama file, bukan nama halaman.

Aturan keras `app/` tetep berlaku di semua view: nggak ada `if` yang ngubah `risk_level`
atau `recommended_action`, nggak ada angka ambang yang ditulis langsung, semua dari
`config.py`.
"""

from __future__ import annotations

import os
import sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import bootstrap  # noqa: F401,E402  -- sys.path repo + .env, harus paling duluan

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Fraud Investigation", page_icon="🔎", layout="wide")

# default=True bikin Overview jadi halaman pendaratan -- ringkasan dulu, baru drill down.
_pages = [
    st.Page("views/overview.py", title="Overview", icon="📊", default=True),
    st.Page("views/investigate.py", title="Investigate", icon="🔎"),
    st.Page("views/batch_data.py", title="Batch Data", icon="🗂️"),
    st.Page("views/limitations.py", title="Limitations", icon="⚠️"),
]

st.navigation(_pages).run()
