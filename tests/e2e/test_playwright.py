"""
UAT lapis browser: server Streamlit beneran + Chromium headless (Playwright).

Bedanya sama `test_apptest.py`: di sini app-nya nyala sebagai proses terpisah
(`streamlit run app/main.py`), dan yang diklik itu DOM sungguhan lewat websocket. Yang cuma
bisa ketangkep di lapis ini: navigasi sidebar antar halaman, expander yang beneran harus
diklik biar isinya kelihatan, dan traceback yang nongol di layar.

Struktur app-nya drill-down: Overview (pendaratan) -> Investigate -> Batch Data /
Limitations. Konten expander yang ketutup ADA di DOM tapi nggak visible -- makanya tes
yang mau ngecek isi bukti harus `expand()` dulu, persis kayak user beneran.

Server-nya scope session (satu kali nyala buat semua tes), tapi tiap tes dapet konteks
browser baru dari pytest-playwright -- `st.session_state` mulai kosong lagi tiap tes.

Nggak ada satu pun tes di sini yang manggil OpenAI: fixture `app_env` nyalain server dengan
`OPENAI_API_KEY` kosong.

Prasyarat sekali jalan:
    .venv\\Scripts\\python.exe -m playwright install chromium
"""

from __future__ import annotations

import os
import time

import pytest
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, expect

from conftest import (
    AUTO_APPROVE_CUSTOMER,
    CASE_FILE_DIR,
    CLEAN_CUSTOMER,
    PROMO_CUSTOMER,
    RING_CUSTOMER,
    UNKNOWN_CUSTOMER,
)

# Klik pertama di server yang baru nyala harus bayar `build_resources()` (model, indeks
# Chroma, refit encoder), jadi batasnya digedein. Ini batas atas biar kegagalan tetep
# kelihatan, bukan target waktu.
FIRST_RUN_TIMEOUT_MS = 240_000
UI_TIMEOUT_MS = 30_000

# Potongan label, bukan label utuh: `--` di label expander dirender markdown jadi
# em-dash di DOM, jadi filter pakai teks mentahnya nggak bakal pernah match.
L4_EXPANDER = "SOP-004 report & narrative (L4)"
L2_EXPANDER = "customer profile & patterns (L2)"
L3_EXPANDER = "findings & SOP citations (L3)"
L1_EXPANDER = "model score (L1)"


# --- helper ------------------------------------------------------------------------


def open_overview(page: Page, url: str) -> Page:
    """Buka app-nya; halaman pendaratannya Overview."""
    page.set_default_timeout(UI_TIMEOUT_MS)
    page.goto(url, wait_until="domcontentloaded")
    expect(
        page.get_by_role("heading", name="Agentic Fraud & Anomaly Investigation")
    ).to_be_visible(timeout=FIRST_RUN_TIMEOUT_MS)
    return page


def goto_investigate(page: Page) -> None:
    """Pindah ke halaman Investigate lewat link sidebar, kayak user beneran."""
    page.get_by_test_id("stSidebarNav").get_by_role("link", name="Investigate").click()
    expect(page.get_by_role("heading", name="Investigate a Customer")).to_be_visible()


def assert_no_traceback(page: Page) -> None:
    """
    Nggak boleh ada exception yang kerender di halaman.

    Dicek dua-duanya: elemen exception khas Streamlit, dan kata "Traceback" di teks
    halaman. Yang pertama nangkep exception yang nggak ketangkep, yang kedua nangkep
    traceback yang kelanjur dicetak ke dalam teks biasa.
    """
    assert page.get_by_test_id("stException").count() == 0, "an exception is rendered on the page"
    body = page.locator("body").inner_text()
    assert "Traceback (most recent call last)" not in body


def enable_manual_mode(page: Page) -> None:
    """Centang "Enter a customer ID manually" biar kolom teksnya muncul."""
    page.get_by_test_id("stSidebar").get_by_text("Enter a customer ID manually").click()
    expect(page.get_by_test_id("stTextInput")).to_be_visible()


def type_customer_id(page: Page, customer_id: str) -> None:
    """Isi kolom manual, terus commit-nya pakai Enter (Streamlit baru rerun setelah itu)."""
    field = page.get_by_test_id("stTextInput").locator("input")
    field.fill(customer_id)
    field.press("Enter")


def pick_from_selectbox(page: Page, index: int, value: str) -> None:
    """Pilih nilai di selectbox ke-`index` di halaman (0-based)."""
    box = page.get_by_test_id("stSelectbox").nth(index)
    box.click()
    page.keyboard.type(value)
    page.get_by_role("option", name=value, exact=True).click()
    # Nilainya nempel di `value` combobox-nya, bukan di teks kontainernya.
    expect(box.get_by_role("combobox")).to_have_value(value)


def click_investigate(page: Page) -> None:
    page.get_by_role("button", name="Investigate", exact=True).click()


def decision_banner(page: Page, action: str):
    """Locator banner keputusan: alert yang nyebut action-nya."""
    return (
        page.get_by_test_id("stAlertContainer")
        .filter(has_text="Recommended action:")
        .filter(has_text=action)
        .first
    )


def expand(page: Page, label: str) -> None:
    """Buka satu expander bukti lewat kliknya user, bukan lewat DOM langsung."""
    page.get_by_test_id("stExpander").filter(has_text=label).locator("summary").first.click()


def wait_for_idle(page: Page, timeout: int = FIRST_RUN_TIMEOUT_MS) -> None:
    """
    Tungguin Streamlit selesai rerun.

    Indikatornya status widget di pojok kanan atas. Kalau rerun-nya kelewat cepat (hasil
    dari cache), widget-nya bisa nggak sempat kelihatan sama sekali -- itu bukan kegagalan,
    makanya timeout tahap pertama ditelen.
    """
    status = page.get_by_test_id("stStatusWidget")
    try:
        status.wait_for(state="visible", timeout=3_000)
    except PlaywrightTimeoutError:
        return
    status.wait_for(state="hidden", timeout=timeout)


def investigate_manual(page: Page, customer_id: str) -> None:
    enable_manual_mode(page)
    type_customer_id(page, customer_id)
    click_investigate(page)


def run_counter_text(page: Page) -> str:
    return page.get_by_test_id("stSidebar").get_by_text("Pipeline runs this session:").inner_text()


# --- 1a. Overview: pendaratan = ringkasan ------------------------------------------


def test_overview_opens_with_summary_first(page: Page, streamlit_server: str):
    open_overview(page, streamlit_server)

    # KPI kelihatan TANPA satu klik pun -- ini inti perombakannya.
    for label in ("Customers reviewed", "Block", "Escalate", "Auto-Approve"):
        expect(page.get_by_test_id("stMetric").filter(has_text=label).first).to_be_visible()

    # Dua chart + worklist-nya kerender.
    expect(page.get_by_test_id("stVegaLiteChart").first).to_be_visible()
    expect(page.get_by_role("heading", name="Worklist")).to_be_visible()
    expect(page.get_by_test_id("stDataFrame").first).to_be_visible()
    expect(page.get_by_text("Synthetic dataset generated for this case study")).to_be_visible()

    assert_no_traceback(page)


def test_overview_open_case_drills_down_to_the_investigation(
    page: Page, streamlit_server: str
):
    """Jalur drill-down utuh: worklist -> Open case -> halaman Investigate + hasil."""
    open_overview(page, streamlit_server)

    pick_from_selectbox(page, 0, RING_CUSTOMER)
    page.get_by_role("button", name="Open case").click()

    expect(page.get_by_role("heading", name="Investigate a Customer")).to_be_visible(
        timeout=FIRST_RUN_TIMEOUT_MS
    )
    expect(decision_banner(page, "Block")).to_be_visible(timeout=FIRST_RUN_TIMEOUT_MS)
    assert_no_traceback(page)


# --- 2. nasabah ring: headline dulu, bukti di expander -----------------------------


def test_ring_customer_headline_then_expandable_evidence(page: Page, streamlit_server: str):
    open_overview(page, streamlit_server)
    goto_investigate(page)
    investigate_manual(page, RING_CUSTOMER)

    expect(decision_banner(page, "Block")).to_be_visible(timeout=FIRST_RUN_TIMEOUT_MS)
    metrics = page.get_by_test_id("stMetric")
    expect(metrics.filter(has_text="Risk level").first).to_contain_text("HIGH")

    # Status narasi (di sini: gagal, server tanpa API key) harus kebaca DI HEADLINE,
    # tanpa buka expander apa pun -- itu permintaan eksplisit user (kasus 10 pindah atas).
    expect(page.get_by_text("Narrative not generated.").first).to_be_visible()

    # Keempat expander bukti ada; isinya baru kelihatan setelah diklik.
    for label in (L1_EXPANDER, L2_EXPANDER, L3_EXPANDER, L4_EXPANDER):
        expect(page.get_by_test_id("stExpander").filter(has_text=label).first).to_be_visible()

    expand(page, L2_EXPANDER)
    expect(page.get_by_text("DEV_SHARED_9F21A").first).to_be_visible()

    expand(page, L4_EXPANDER)
    expect(page.get_by_role("heading", name="7. Supporting Data References")).to_be_visible()

    assert_no_traceback(page)


# --- 3. nasabah tanpa temuan L2 + jalur Auto-Approve -------------------------------


def test_customer_without_findings_still_reports_field_4(page: Page, streamlit_server: str):
    """
    Field 4 SOP-004 harus tetep nulis "checked, nothing found", bukan section kosong.

    Action-nya Escalate, bukan Auto-Approve: skor L1 asli nasabah ini 0.879 -- lihat
    catatan di `conftest.py`.
    """
    open_overview(page, streamlit_server)
    goto_investigate(page)
    pick_from_selectbox(page, 0, CLEAN_CUSTOMER)
    click_investigate(page)

    expect(decision_banner(page, "Escalate")).to_be_visible(timeout=FIRST_RUN_TIMEOUT_MS)
    metrics = page.get_by_test_id("stMetric")
    expect(metrics.filter(has_text="Risk level").first).to_contain_text("NONE")

    expand(page, L4_EXPANDER)
    expect(page.get_by_role("heading", name="4. Triggering Evidence")).to_be_visible()
    # Dicari sebagai item daftar, bukan teks bebas: kalimat yang sama juga ada di dalam
    # expander "Raw pipeline state", dan `get_by_text` bakal kena dua-duanya.
    evidence = page.locator(
        "li", has_text=f"Customer {CLEAN_CUSTOMER} checked, nothing found."
    )
    expect(evidence.first).to_be_visible()

    assert_no_traceback(page)


def test_auto_approve_path_renders(page: Page, streamlit_server: str):
    open_overview(page, streamlit_server)
    goto_investigate(page)
    investigate_manual(page, AUTO_APPROVE_CUSTOMER)

    expect(decision_banner(page, "Auto-Approve")).to_be_visible(timeout=FIRST_RUN_TIMEOUT_MS)
    assert_no_traceback(page)


# --- 4. nasabah promo --------------------------------------------------------------


def test_promo_customer_shows_the_promo_abuse_category(page: Page, streamlit_server: str):
    open_overview(page, streamlit_server)
    goto_investigate(page)
    investigate_manual(page, PROMO_CUSTOMER)

    expect(decision_banner(page, "Block")).to_be_visible(timeout=FIRST_RUN_TIMEOUT_MS)

    expand(page, L4_EXPANDER)
    expect(page.get_by_role("heading", name="3. Fraud Category")).to_be_visible()
    expect(page.get_by_text("Promo Abuse").first).to_be_visible()

    assert_no_traceback(page)


# --- 5. id nggak dikenal -----------------------------------------------------------


def test_unknown_customer_shows_a_readable_error(
    page: Page, streamlit_server: str, case_files_snapshot: set
):
    open_overview(page, streamlit_server)
    goto_investigate(page)
    investigate_manual(page, UNKNOWN_CUSTOMER)

    error = page.get_by_test_id("stAlertContainer").filter(has_text="Investigation stopped")
    expect(error).to_be_visible(timeout=FIRST_RUN_TIMEOUT_MS)
    expect(error).to_contain_text(f"unknown customer_id: {UNKNOWN_CUSTOMER}")

    # Nggak ada headline maupun expander bukti buat jalur ini.
    expect(page.get_by_text("Recommended action:")).to_have_count(0)
    expect(page.get_by_test_id("stExpander")).to_have_count(0)

    assert_no_traceback(page)

    # Dan nggak ada case file baru yang lahir di outputs/case_files/.
    assert set(os.listdir(CASE_FILE_DIR)) == case_files_snapshot


# --- 6. input kosong ---------------------------------------------------------------


@pytest.mark.parametrize("raw_input", ["", "   "])
def test_blank_customer_id_does_not_raise(
    page: Page, streamlit_server: str, raw_input: str, case_files_snapshot: set
):
    open_overview(page, streamlit_server)
    goto_investigate(page)
    enable_manual_mode(page)
    if raw_input:
        type_customer_id(page, raw_input)
    click_investigate(page)

    expect(page.get_by_text("Enter a customer ID first.")).to_be_visible(
        timeout=FIRST_RUN_TIMEOUT_MS
    )
    assert_no_traceback(page)
    assert set(os.listdir(CASE_FILE_DIR)) == case_files_snapshot


# --- 7. klik dua kali = cache ------------------------------------------------------


def test_second_click_does_not_run_the_pipeline_again(page: Page, streamlit_server: str):
    """
    Bukti terukurnya counter "Pipeline runs this session", bukan firasat.

    Sengaja counter, bukan durasi: satu `invoke()` = satu panggilan OpenAI, jadi yang harus
    dijamin nggak nambah itu JUMLAH panggilannya. Durasi tetep diukur dan dicetak sebagai
    bukti pendukung, tapi nggak dijadiin syarat lulus -- mesin yang lagi sibuk bikin
    perbandingan waktu jadi sumber tes yang kadang merah tanpa ada yang rusak.
    """
    open_overview(page, streamlit_server)
    goto_investigate(page)
    expect(page.get_by_text("Pipeline runs this session: 0 | cached customers: 0")).to_be_visible()

    investigate_manual(page, RING_CUSTOMER)
    started = time.monotonic()
    expect(page.get_by_text("Pipeline runs this session: 1 | cached customers: 1")).to_be_visible(
        timeout=FIRST_RUN_TIMEOUT_MS
    )
    first_click_seconds = time.monotonic() - started

    started = time.monotonic()
    click_investigate(page)
    wait_for_idle(page)
    second_click_seconds = time.monotonic() - started

    # Counter-nya tetep 1: klik kedua dilayanin dari cache session_state.
    expect(page.get_by_text("Pipeline runs this session: 1 | cached customers: 1")).to_be_visible()
    # Dan tetep 1 setelah app-nya bener-bener diem (bukan kebetulan kefoto sebelum naik).
    page.wait_for_timeout(2_000)
    assert "Pipeline runs this session: 1" in run_counter_text(page)

    # Hasilnya tetep kerender penuh dari cache.
    expect(decision_banner(page, "Block")).to_be_visible()
    assert_no_traceback(page)

    print(
        f"\nfirst click: {first_click_seconds:.2f}s | "
        f"second click (cached): {second_click_seconds:.2f}s"
    )


# --- 8. ganti nasabah --------------------------------------------------------------


def test_switching_customer_clears_the_old_panel(page: Page, streamlit_server: str):
    open_overview(page, streamlit_server)
    goto_investigate(page)
    investigate_manual(page, RING_CUSTOMER)
    expect(decision_banner(page, "Block")).to_be_visible(timeout=FIRST_RUN_TIMEOUT_MS)

    type_customer_id(page, AUTO_APPROVE_CUSTOMER)

    # Panel lama harus hilang, bukan sekadar ketimpa sebagian.
    expect(page.get_by_text("Pick a customer in the sidebar, then press Investigate.")).to_be_visible()
    expect(page.get_by_text("Recommended action:")).to_have_count(0)
    assert RING_CUSTOMER not in page.locator('[data-testid="stMain"]').inner_text()

    assert_no_traceback(page)


# --- 9. halaman batch + limitations ------------------------------------------------


def test_batch_data_page_opens_from_the_sidebar(page: Page, streamlit_server: str):
    open_overview(page, streamlit_server)

    page.get_by_test_id("stSidebarNav").get_by_role("link", name="Batch Data").click()

    expect(page.get_by_role("heading", name="Batch Data (raw)")).to_be_visible(
        timeout=FIRST_RUN_TIMEOUT_MS
    )
    for filename in ("pipeline_run.csv", "l3_decision_report.csv", "l4_case_summary.csv"):
        expect(page.get_by_role("heading", name=filename)).to_be_visible()

    # Halaman ini cuma baca artefak, jangan sampai diam-diam ngitung ulang.
    expect(page.get_by_text("It never re-runs the pipeline")).to_be_visible()
    expect(page.get_by_test_id("stDataFrame").first).to_be_visible()

    assert_no_traceback(page)


def test_limitations_page_opens_from_the_sidebar(page: Page, streamlit_server: str):
    open_overview(page, streamlit_server)

    page.get_by_test_id("stSidebarNav").get_by_role("link", name="Limitations").click()

    expect(page.get_by_role("heading", name="Limitations", exact=True)).to_be_visible(
        timeout=FIRST_RUN_TIMEOUT_MS
    )
    expect(
        page.get_by_text("Only 6 of the 12 L3 matrix boxes are reachable")
    ).to_be_visible()
    expect(page.get_by_text("RAG here buys traceability, not precision")).to_be_visible()

    assert_no_traceback(page)
