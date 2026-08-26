"""
View layer (Streamlit) -- semua yang mahal, di-cache di satu tempat.

Modul ini NGGAK punya logika domain. Dia cuma:
  - manggil `pipeline.state.build_resources()` dan `pipeline.graph.build_pipeline()` sekali,
  - nyimpen hasil `graph.invoke()` per `customer_id` di `st.session_state`,
  - baca CSV batch yang udah ada di `outputs/`.

Kenapa caching-nya penting di sini, bukan sekadar optimasi: Streamlit ngejalanin ulang
seluruh script tiap kali ada widget disentuh, dan `report_l4` manggil OpenAI sekali per
`invoke()`. Tanpa cache, satu klik apa pun = satu tagihan API.

`@st.cache_resource`, bukan `@st.cache_data`: koleksi Chroma, RandomForest, OneHotEncoder,
dan graph hasil compile itu objek hidup yang nggak bisa di-pickle. `cache_data` bakal nyoba
nge-pickle hasilnya dan gagal.
"""

from __future__ import annotations

import os
import sys
import traceback

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import bootstrap  # noqa: F401,E402  -- sys.path repo + .env, harus duluan

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
import data_io  # noqa: E402
from investigation import pattern_lookup, sop_retriever  # noqa: E402
from pipeline.graph import build_pipeline  # noqa: E402
from pipeline.state import build_resources, state_to_dict  # noqa: E402


# Key di st.session_state. Konstanta biar tes bisa baca yang sama persis kayak UI.
RESULT_CACHE_KEY = "investigation_results"
RUN_COUNT_KEY = "pipeline_run_count"

# Berkas batch yang UDAH ADA di outputs/. App ini cuma baca -- nggak pernah ngitung ulang
# satu batch pun (1200 nasabah x ~0.07 s buat profil L2 doang, plus satu panggilan OpenAI
# per nasabah).
BATCH_SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "pipeline_run.csv",
        "Pipeline run (L1-L4 through the controller)",
        "notebook 07",
    ),
    (
        "l3_decision_report.csv",
        "L3 decision matrix, two L1 score scenarios",
        "notebook 05",
    ),
    (
        "l4_case_summary.csv",
        "L4 case file summary",
        "notebook 06",
    ),
)


@st.cache_resource(show_spinner="Loading data, SOP index and model...")
def get_pipeline():
    """
    Bangun resources + graph sekali per proses. Balikannya tuple `(resources, graph)`.

    Pengecekan indeks Chroma di sini bukan basa-basi. `build_resources()` bakal nge-upsert
    kalau koleksinya kosong, dan di dalam proses Streamlit pandas selalu keimport duluan --
    kombinasi yang matiin proses tanpa traceback (lihat docstring `app/bootstrap.py`).
    Jadi kalau indeksnya kosong, app-nya berhenti dengan pesan yang bisa dibaca, bukan mati.
    """
    collection = sop_retriever.get_sop_collection()
    if collection.count() == 0:
        st.error(
            "The SOP vector index is empty. Build it once outside Streamlit "
            "(notebooks/03_L2_sop_retriever_rag.ipynb), then reload this page."
        )
        st.stop()

    resources = build_resources()
    return resources, build_pipeline(resources)


@st.cache_data(show_spinner=False)
def get_customer_ids() -> list[str]:
    """
    Daftar id buat dropdown.

    Sengaja baca CSV langsung, BUKAN lewat `get_pipeline()`. Dropdown-nya butuh keisi pas
    halaman pertama kali kebuka, sementara `build_resources()` itu belasan detik (model,
    indeks Chroma, refit encoder). Nempelin dua hal itu bikin app-nya bengong di awal
    padahal pipeline-nya belum tentu dipakai.

    `cache_data` boleh di sini -- list of str bisa di-pickle.
    """
    return sorted(data_io.load_customers()["customer_id"].tolist())


@st.cache_data(show_spinner=False)
def load_batch_csv(filename: str) -> pd.DataFrame | None:
    """
    Baca satu CSV batch dari `outputs/`. `None` kalau filenya belum ada.

    Balikin `None`, jangan ngeraise: berkas batch itu artefak notebook yang di-gitignore,
    jadi "belum pernah dijalanin" itu keadaan wajar buat orang yang baru clone repo ini --
    bukan kesalahan yang pantes bikin halamannya mati.
    """
    path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_result_cache() -> dict:
    """Hasil investigasi per nasabah, sepanjang sesi browser ini."""
    return st.session_state.setdefault(RESULT_CACHE_KEY, {})


def get_run_count() -> int:
    """Berapa kali pipeline beneran jalan di sesi ini. Yang dari cache nggak keitung."""
    return st.session_state.get(RUN_COUNT_KEY, 0)


def investigate(customer_id: str) -> dict:
    """
    Jalanin graph sekali buat satu nasabah, atau balikin hasil yang udah ada di cache.

    Yang disimpen state mentahnya (`decision` masih objek `Decision`), bukan hasil
    `state_to_dict()`. Renderer butuh objeknya -- `RiskFinding.is_actionable` hilang begitu
    di-`asdict()`, dan nyusun ulang aturan "finding mana yang nyala" di lapisan view itu
    persis logika domain yang nggak boleh ada di sini.

    Satu panggilan = satu panggilan OpenAI (`report_l4` -> `attach_narrative`). Jadi
    penambah counter di bawah ini juga yang jadi ukuran ongkos.
    """
    cache = get_result_cache()
    if customer_id in cache:
        return cache[customer_id]

    _resources, graph = get_pipeline()
    out = graph.invoke({"customer_id": customer_id})

    st.session_state[RUN_COUNT_KEY] = get_run_count() + 1
    cache[customer_id] = out
    return out


def result_as_json(state: dict) -> dict:
    """Versi state yang aman buat ditampilin sebagai JSON (jejak audit per langkah)."""
    return state_to_dict(state)


@st.cache_data(show_spinner=False)
def customer_complaints(customer_id: str) -> pd.DataFrame:
    """Komplain satu nasabah, lewat fungsi L2 -- bukan filter tandingan di lapisan view."""
    return pattern_lookup.get_customer_complaints(
        data_io.load_complaint_notes(), customer_id
    )


def format_exception(exc: BaseException) -> str:
    """
    Pesan singkat buat layar, traceback lengkap ke stderr (kelihatan di terminal).

    Traceback nggak ditaruh di halaman: pembaca dashboard nggak bisa berbuat apa-apa sama
    itu, tapi orang yang ngejalanin app-nya butuh. Ditelen diam-diam juga bukan pilihan.
    """
    traceback.print_exc()
    return f"{type(exc).__name__}: {exc}"
