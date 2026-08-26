"""
UAT lapis cepat: `streamlit.testing.v1.AppTest`.

Yang dites di sini: `app/main.py` (router `st.navigation`) beneran dijalanin, widget-nya
beneran diklik, dan hasilnya dibaca dari pohon elemen yang dirender. Struktur app-nya
drill-down: Overview (KPI + worklist) -> Investigate (headline keputusan + bukti di
expander) -> Batch Data / Limitations.

Semua tes di modul ini jalan tanpa `OPENAI_API_KEY` (dikosongin di `conftest.py` sebelum
modul apa pun keimport), kecuali satu tes bertanda `slow` di paling bawah.

Catatan kecepatan: `@st.cache_resource` hidup selama satu proses, jadi resources + graph
cuma dibangun sekali buat seluruh modul ini. Tes pertama yang jalanin investigasi bakal
makan belasan detik; sisanya cepat.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from conftest import (
    APP_SCRIPT,
    AUTO_APPROVE_CUSTOMER,
    CASE_FILE_DIR,
    CLEAN_CUSTOMER,
    PIPELINE_RUN_CSV,
    PROMO_CUSTOMER,
    REPO_ROOT,
    RING_CUSTOMER,
    UNKNOWN_CUSTOMER,
)

# Cukup longgar buat run pertama (build_resources ~10-20 detik di mesin lokal), tapi masih
# ngasih kegagalan yang jelas kalau ada yang nyantol beneran.
TIMEOUT_SECONDS = 300

INVESTIGATE_VIEW = "views/investigate.py"
BATCH_VIEW = "views/batch_data.py"
LIMITATIONS_VIEW = "views/limitations.py"


def fresh_app() -> AppTest:
    """Instance app baru dengan session_state kosong. Halaman default = Overview."""
    app = AppTest.from_file(APP_SCRIPT, default_timeout=TIMEOUT_SECONDS)
    app.run()
    return app


def open_investigate() -> AppTest:
    """App baru langsung di halaman Investigate."""
    app = fresh_app()
    app.switch_page(INVESTIGATE_VIEW)
    app.run()
    return app


def rendered_text(app: AppTest) -> str:
    """
    Gabungin semua teks yang kerender jadi satu string buat dicari isinya.

    Bukan cuma `at.markdown`: pesan penting di app ini kesebar di `st.warning`, `st.info`,
    `st.error`, `st.caption` dan `st.success`. Konten di dalam expander ikut kebaca --
    pohon elemennya flat.
    """
    chunks: list[str] = []
    for elements in (
        app.title,
        app.header,
        app.subheader,
        app.markdown,
        app.caption,
        app.info,
        app.success,
        app.warning,
        app.error,
        app.text,
    ):
        chunks.extend(str(element.value) for element in elements)
    return "\n".join(chunks)


def investigate_manual(app: AppTest, customer_id: str) -> AppTest:
    """Isi id lewat jalur manual di halaman Investigate, terus pencet Investigate."""
    app.checkbox(key="manual_mode").check().run()
    app.text_input(key="manual_customer_id").set_value(customer_id).run()
    app.button(key="investigate_button").click().run()
    return app


def metric_values(app: AppTest) -> dict[str, str]:
    """`{label: value}` dari semua `st.metric`. Label kembar diambil yang pertama (headline)."""
    values: dict[str, str] = {}
    for metric in app.metric:
        values.setdefault(metric.label, str(metric.value))
    return values


def banner_texts(app: AppTest) -> list[str]:
    """Semua alert (error/warning/success) yang isinya banner keputusan."""
    texts = []
    for elements in (app.error, app.warning, app.success):
        for element in elements:
            if "Recommended action:" in str(element.value):
                texts.append(str(element.value))
    return texts


# --- 1a. Overview: halaman pendaratan ----------------------------------------------


def test_overview_opens_with_no_traceback():
    app = fresh_app()

    assert app.exception == [], f"overview raised on first render: {app.exception}"
    assert app.title[0].value == "Agentic Fraud & Anomaly Investigation"
    assert "Synthetic dataset" in rendered_text(app)
    # Worklist + kontrol drill-down-nya ada dari awal.
    assert [s.value for s in app.subheader] == ["Worklist"]
    assert app.button(key="open_case_button").label == "Open case"


def test_overview_kpis_match_the_saved_batch():
    """Angka KPI harus persis isi pipeline_run.csv -- nggak ada yang diketik langsung."""
    frame = pd.read_csv(PIPELINE_RUN_CSV)
    scored = frame[frame["error"].isna()] if "error" in frame.columns else frame
    counts = scored["action"].value_counts()

    metrics = metric_values(fresh_app())
    assert metrics["Customers reviewed"] == str(len(scored))
    assert metrics["Block"] == str(int(counts.get("Block", 0)))
    assert metrics["Escalate"] == str(int(counts.get("Escalate", 0)))
    assert metrics["Auto-Approve"] == str(int(counts.get("Auto-Approve", 0)))


def test_overview_worklist_is_sorted_most_urgent_first():
    """Semua Block duluan, baru Escalate, baru Auto-Approve -- urutan worklist = urutan kerja."""
    frame = pd.read_csv(PIPELINE_RUN_CSV)
    scored = frame[frame["error"].isna()] if "error" in frame.columns else frame
    n_block = int((scored["action"] == "Block").sum())

    app = fresh_app()
    options = app.selectbox(key="worklist_pick").options
    assert len(options) == len(scored)

    actions = scored.set_index("customer_id")["action"]
    assert all(actions[cid] == "Block" for cid in options[:n_block])


def test_overview_open_case_hands_off_to_investigate():
    """Tombol "Open case" = drill down: pindah halaman DAN jalanin investigasi id itu."""
    app = fresh_app()
    app.selectbox(key="worklist_pick").select(RING_CUSTOMER).run()
    app.button(key="open_case_button").click().run()

    assert app.exception == [], f"hand-off raised: {app.exception}"
    assert app.title[0].value == "Investigate a Customer"
    assert any("Block" in t for t in banner_texts(app))
    assert app.session_state["pipeline_run_count"] == 1
    # Hand-off-nya sekali pakai: kunjungan berikutnya nggak boleh auto-jalan lagi.
    # SafeSessionState nggak bisa di-iterate, jadi ngeceknya lewat KeyError.
    with pytest.raises(KeyError):
        app.session_state["pending_investigation"]


def test_overview_without_pipeline_run_csv_asks_for_notebook_07(missing_pipeline_run_csv):
    # Cache dibersihin dulu: `load_batch_csv` di-cache per nama berkas, jadi tanpa ini
    # halamannya bakal ngasih DataFrame basi dari tes sebelumnya dan tesnya lulus palsu.
    import streamlit as st

    st.cache_data.clear()
    assert not os.path.exists(missing_pipeline_run_csv)

    app = fresh_app()
    assert app.exception == [], f"missing-file path raised: {app.exception}"
    warnings = [w.value for w in app.warning]
    assert any("pipeline_run.csv not found" in w for w in warnings), warnings
    assert any("notebook 07" in w for w in warnings), warnings
    # Jalan keluarnya tetep ditawarin, bukan halaman buntu.
    assert app.button[0].label == "Go to Investigate"

    st.cache_data.clear()


# --- 1b. halaman Investigate kebuka -------------------------------------------------


def test_investigate_page_lists_all_customers():
    app = open_investigate()

    assert app.exception == []
    options = app.selectbox(key="picked_customer_id").options
    # 1200 nasabah persis, sesuai data/customers.csv.
    assert len(options) == 1200
    assert RING_CUSTOMER in options
    assert "Pick a customer in the sidebar, then press Investigate." in rendered_text(app)


# --- 2. nasabah ring ---------------------------------------------------------------


def test_ring_customer_headline_first_then_evidence():
    app = investigate_manual(open_investigate(), RING_CUSTOMER)

    assert app.exception == [], f"investigation raised: {app.exception}"

    # Headline: banner keputusan pakai st.error (Block), angka pendukung di metric.
    assert any("Block" in t for t in banner_texts(app))
    metrics = metric_values(app)
    assert metrics["Risk level"] == "HIGH"
    assert metrics["L2 risk level"] == "HIGH"

    # Bukti per layer di expander, urut L1 -> L4.
    labels = [x.label for x in app.expander]
    assert labels == [
        "Detection -- model score (L1)",
        "Investigation -- customer profile & patterns (L2)",
        "Decision detail -- findings & SOP citations (L3)",
        "Case file -- SOP-004 report & narrative (L4)",
        "Raw pipeline state (audit trail)",
    ]

    text = rendered_text(app)
    assert "DEV_SHARED_9F21A" in text
    assert "SOP-002" in text
    # Case file SOP-004 kerender utuh di expander L4.
    assert "## 7. Supporting Data References" in text
    # Rationale-nya di headline, jadi keputusan bisa dibantah tanpa buka expander.
    assert "**Why:**" in text


# --- 3. nasabah tanpa temuan L2 + jalur Auto-Approve -------------------------------


def test_customer_with_no_l2_finding_still_reports_field_4():
    """
    L2 nggak nemu apa-apa -> field 4 SOP-004 WAJIB tetep nulis "checked, nothing found",
    bukan section kosong.

    Action-nya Escalate, bukan Auto-Approve, dan itu bukan bug: skor L1 asli nasabah ini
    0.879 (band HIGH), dan matriks option C maping HIGH x NONE ke Escalate -- lihat
    catatan di `conftest.py`.
    """
    app = open_investigate()
    # Lewat dropdown, bukan kolom manual: jalur ini yang dipakai orang normal.
    app.selectbox(key="picked_customer_id").select(CLEAN_CUSTOMER).run()
    app.button(key="investigate_button").click().run()

    assert app.exception == [], f"investigation raised: {app.exception}"

    assert any("Escalate" in t for t in banner_texts(app))
    metrics = metric_values(app)
    assert metrics["Risk level"] == "NONE"
    assert metrics["L2 risk level"] == "NONE"

    text = rendered_text(app)
    assert f"Customer {CLEAN_CUSTOMER} checked, nothing found." in text
    assert "## 4. Triggering Evidence" in text


def test_auto_approve_customer_renders_the_auto_approve_path():
    """Satu-satunya kotak matriks yang bisa Auto-Approve: L1 LOW x L2 NONE."""
    app = open_investigate()
    app.selectbox(key="picked_customer_id").select(AUTO_APPROVE_CUSTOMER).run()
    app.button(key="investigate_button").click().run()

    assert app.exception == [], f"investigation raised: {app.exception}"

    # Banner Auto-Approve harus lewat st.success, bukan error/warning.
    assert any("Auto-Approve" in str(s.value) for s in app.success)
    metrics = metric_values(app)
    assert metrics["L1 band"] == "LOW"
    assert metrics["Risk level"] == "NONE"

    assert f"Customer {AUTO_APPROVE_CUSTOMER} checked, nothing found." in rendered_text(app)


# --- 4. nasabah promo --------------------------------------------------------------


def test_promo_customer_reports_the_promo_abuse_category():
    app = investigate_manual(open_investigate(), PROMO_CUSTOMER)

    assert app.exception == [], f"investigation raised: {app.exception}"

    text = rendered_text(app)
    assert "Promo Abuse" in text
    assert "SOP-003" in text


# --- 5. id nggak dikenal -----------------------------------------------------------


def test_unknown_customer_shows_a_readable_error(case_files_snapshot):
    app = investigate_manual(open_investigate(), UNKNOWN_CUSTOMER)

    assert app.exception == [], f"error path raised instead of reporting: {app.exception}"
    errors = [e.value for e in app.error]
    assert len(errors) == 1
    assert UNKNOWN_CUSTOMER in errors[0]
    assert "unknown customer_id" in errors[0]

    # Nggak ada headline maupun expander bukti buat jalur ini.
    assert banner_texts(app) == []
    assert [x.label for x in app.expander] == []

    # Dan nggak ada case file baru yang lahir.
    assert set(os.listdir(CASE_FILE_DIR)) == case_files_snapshot


# --- 6. input kosong ---------------------------------------------------------------


@pytest.mark.parametrize("raw_input", ["", "   ", "\t "])
def test_blank_customer_id_warns_instead_of_raising(raw_input, case_files_snapshot):
    app = open_investigate()
    app.checkbox(key="manual_mode").check().run()
    app.text_input(key="manual_customer_id").set_value(raw_input).run()
    app.button(key="investigate_button").click().run()

    assert app.exception == [], f"blank input raised: {app.exception}"
    assert any("Enter a customer ID first." in w.value for w in app.warning)
    assert set(os.listdir(CASE_FILE_DIR)) == case_files_snapshot


# --- 7. klik dua kali = cache ------------------------------------------------------


def test_second_click_on_the_same_customer_is_served_from_cache():
    app = investigate_manual(open_investigate(), RING_CUSTOMER)
    assert app.session_state["pipeline_run_count"] == 1

    app.button(key="investigate_button").click().run()

    # Ini buktinya, dan sengaja counter bukan durasi: satu invoke = satu panggilan OpenAI,
    # jadi yang harus dijamin nggak nambah itu jumlah panggilan, bukan sekadar rasa cepat.
    assert app.session_state["pipeline_run_count"] == 1
    assert len(app.session_state["investigation_results"]) == 1
    # Hasilnya tetep kerender penuh dari cache, bukan hilang.
    assert any("Block" in t for t in banner_texts(app))
    assert "Pipeline runs this session: 1" in rendered_text(app)


# --- 8. ganti nasabah --------------------------------------------------------------


def test_switching_customer_clears_the_previous_panel():
    app = investigate_manual(open_investigate(), RING_CUSTOMER)
    assert any("Block" in t for t in banner_texts(app))

    # Ganti id TANPA mencet Investigate lagi.
    app.text_input(key="manual_customer_id").set_value(CLEAN_CUSTOMER).run()

    assert app.exception == []
    text = rendered_text(app)
    assert "Pick a customer in the sidebar, then press Investigate." in text
    assert banner_texts(app) == []
    assert RING_CUSTOMER not in text
    # Hasil lama tetep di cache -- yang dilarang cuma nampilinnya, bukan nyimpennya.
    assert app.session_state["pipeline_run_count"] == 1


# --- 9. halaman batch + limitations ------------------------------------------------


def test_batch_data_page_opens():
    app = fresh_app()
    app.switch_page(BATCH_VIEW)
    app.run()

    assert app.exception == [], f"batch page raised: {app.exception}"
    assert app.title[0].value == "Batch Data (raw)"

    text = rendered_text(app)
    for filename in ("pipeline_run.csv", "l3_decision_report.csv", "l4_case_summary.csv"):
        assert filename in text


def test_batch_data_page_without_pipeline_run_csv_asks_for_notebook_07(
    missing_pipeline_run_csv,
):
    import streamlit as st

    st.cache_data.clear()
    assert not os.path.exists(missing_pipeline_run_csv)

    app = fresh_app()
    app.switch_page(BATCH_VIEW)
    app.run()

    assert app.exception == [], f"missing-file path raised: {app.exception}"

    warnings = [w.value for w in app.warning]
    assert any("pipeline_run.csv not found" in w for w in warnings), warnings
    assert any("notebook 07" in w for w in warnings), warnings

    # Berkas lain tetep kerender -- satu berkas hilang nggak boleh matiin halamannya.
    assert "l4_case_summary.csv" in rendered_text(app)

    st.cache_data.clear()


def test_limitations_page_opens_with_the_measured_numbers():
    app = fresh_app()
    app.switch_page(LIMITATIONS_VIEW)
    app.run()

    assert app.exception == [], f"limitations page raised: {app.exception}"
    text = rendered_text(app)
    assert "Only 6 of the 12 L3 matrix boxes are reachable" in text
    assert "25 near-identical case files" in text
    assert "traceability, not precision" in text
    assert "DEV_SHARED_9F21A" in text


# --- 10. narasi gagal --------------------------------------------------------------


def test_case_file_is_complete_when_the_narrative_is_not_generated():
    app = investigate_manual(open_investigate(), RING_CUSTOMER)

    assert app.exception == []

    warnings = [w.value for w in app.warning]
    assert any("Narrative not generated." in w for w in warnings), warnings

    case = app.session_state["investigation_results"][RING_CUSTOMER]["case"]
    assert case["narrative_summary"] == "Narrative not generated."
    # Tujuh field SOP-004 tetep utuh walau narasinya gagal.
    for field in (
        "case_id",
        "date_opened",
        "customer_ids",
        "fraud_category",
        "triggering_evidence",
        "risk_level",
        "recommended_action",
        "supporting_data_references",
    ):
        assert field in case, f"SOP-004 field missing without a narrative: {field}"

    text = rendered_text(app)
    assert "## 5. Risk Level" in text
    assert "## 6. Recommended Action" in text


def test_dashboard_never_rewrites_the_decision():
    """Aturan keras `app/`: yang dipajang harus sama persis sama isi `Decision`."""
    app = investigate_manual(open_investigate(), RING_CUSTOMER)

    state = app.session_state["investigation_results"][RING_CUSTOMER]
    decision = state["decision"]
    case = state["case"]
    metrics = metric_values(app)

    assert metrics["Risk level"] == decision.risk_level
    assert any(decision.action in t for t in banner_texts(app))
    assert case["risk_level"] == decision.risk_level
    assert case["recommended_action"] == decision.action


# --- satu tes yang beneran manggil OpenAI ------------------------------------------


@pytest.mark.slow
def test_real_narrative_is_generated_when_a_key_is_available(monkeypatch):
    """
    Satu-satunya tes yang ngeluarin biaya (~$0.0002, satu panggilan gpt-4o-mini).

    Deselect pakai `pytest tests/e2e -m "not slow"`.

    Key-nya dibaca langsung dari `.env` di sini, bukan dari environment: `conftest.py`
    sengaja ngosongin `OPENAI_API_KEY` buat seluruh sesi supaya tes lain nggak jajan.
    """
    from dotenv import dotenv_values

    api_key = dotenv_values(os.path.join(REPO_ROOT, ".env")).get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("no OPENAI_API_KEY in .env, so the real narrative path cannot run")

    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    app = investigate_manual(open_investigate(), CLEAN_CUSTOMER)
    assert app.exception == []

    case = app.session_state["investigation_results"][CLEAN_CUSTOMER]["case"]
    narrative = case["narrative_summary"]

    assert narrative != "Narrative not generated.", (
        "the API call failed; check the key and the network before trusting this run"
    )
    assert len(narrative.split()) > 10
    # Narasinya harus muncul sebagai "Summary:" di headline, bukan cuma di case file.
    assert f"**Summary:** {narrative}" in rendered_text(app)

    # Invarian L4 tetep berlaku: narasi nggak boleh nyentuh dua field ini.
    decision = app.session_state["investigation_results"][CLEAN_CUSTOMER]["decision"]
    assert case["risk_level"] == decision.risk_level
    assert case["recommended_action"] == decision.action
