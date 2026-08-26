"""
Fixture bersama buat UAT end-to-end dashboard `app/`.

Dua lapis tes pakai fixture di sini:
  - `tests/e2e/test_apptest.py`   -- `streamlit.testing.v1.AppTest`, tanpa browser, cepat.
  - `tests/e2e/test_playwright.py` -- server Streamlit beneran + Chromium headless.

TIGA HAL YANG BIKIN TES INI AMAN DIJALANIN BERULANG-ULANG:

1. **Nggak ada panggilan OpenAI.** `app_env` nyetel `OPENAI_API_KEY` jadi string KOSONG,
   bukan ngehapus key-nya. Bedanya penting: `app/bootstrap.py` manggil `load_dotenv()`, dan
   python-dotenv itu nge-skip key yang UDAH ada di `os.environ` -- apa pun isinya, termasuk
   string kosong. Kalau key-nya cuma dihapus, `.env` bakal ngisinya lagi dan tiap tes jadi
   satu tagihan API. `generate_narrative()` balikin `None` buat key kosong, jadi case
   file-nya tetep jadi lengkap dengan penanda "Narrative not generated." -- itu justru salah
   satu jalur yang emang harus dites.

2. **Port-nya nggak dihardcode.** 8599 dicoba duluan (sesuai kesepakatan), tapi kalau lagi
   kepakai, fixture-nya minta port bebas ke OS. Tes yang gagal gara-gara ada app lain nyala
   itu kabar palsu.

3. **Case file yang lahir waktu tes dibersihin lagi.** `case_files_snapshot` nyatet isi
   `outputs/case_files/` sebelum tes, terus ngehapus berkas yang BARU muncul pas teardown.
   Berkas lama nggak diapa-apain -- `save_case_file()` emang nimpa, dan menimpa artefak yang
   digenerate ulang itu perilaku yang benar, bukan kotoran.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(REPO_ROOT, "src")

for _path in (REPO_ROOT, SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Dimatiin di level modul, BUKAN di fixture, dan itu disengaja. `AppTest` ngejalanin
# `app/main.py` di dalam proses pytest ini, dan `app/bootstrap.py` manggil `load_dotenv()`
# pas pertama kali diimport. python-dotenv nge-skip key yang udah ada di `os.environ`, jadi
# nyetel string kosong DI SINI -- sebelum modul tes mana pun keimport -- yang bikin key
# asli dari `.env` nggak pernah kepasang. Ditaruh di fixture, urutannya udah telat.
# Tes yang emang butuh narasi asli (satu, ditandai `slow`) baca `.env` sendiri.
os.environ["OPENAI_API_KEY"] = ""

APP_SCRIPT = os.path.join(REPO_ROOT, "app", "main.py")
CASE_FILE_DIR = os.path.join(REPO_ROOT, "outputs", "case_files")
PIPELINE_RUN_CSV = os.path.join(REPO_ROOT, "outputs", "pipeline_run.csv")

PREFERRED_PORT = 8599

# Nasabah yang dipakai berulang di tes. Ketiganya dari `outputs/pipeline_run.csv` yang udah
# ada, jadi hasil yang diharapin bukan tebakan.
RING_CUSTOMER = "CUST11146"      # anggota ring DEV_SHARED_9F21A -> HIGH / Block
CLEAN_CUSTOMER = "CUST10000"     # L2 nggak nemu apa-apa (risk NONE), tapi L1-nya 0.879
PROMO_CUSTOMER = "CUST10061"     # penebus promo -> kategori Promo Abuse, HIGH / Block
AUTO_APPROVE_CUSTOMER = "CUST10001"  # L1 0.119 + L2 NONE -> satu-satunya jalur Auto-Approve
UNKNOWN_CUSTOMER = "CUST99999"   # nggak ada di customers.csv -> jalur handle_error

# CATATAN yang harganya satu tes merah, jangan dihapus: CLAUDE.md nyebut CUST10000 sebagai
# "clean -> NONE/Auto-Approve". Itu bener buat notebook 06, yang jalan pakai skor SKENARIO
# L1 0.05. Pipeline beneran ngasih dia 0.879 -- band L1 HIGH -- dan matriks option C bilang
# HIGH x NONE = Escalate. Jadi `risk_level`-nya emang NONE dan field 4 tetep nulis "checked,
# nothing found", tapi ACTION-nya Escalate, bukan Auto-Approve. Nasabah yang beneran
# Auto-Approve di skor asli ada di AUTO_APPROVE_CUSTOMER; angkanya dari
# `outputs/pipeline_run.csv` (Block 39 / Auto-Approve 16 / Escalate 4).


def pytest_configure(config: pytest.Config) -> None:
    """Daftarin marker `slow` biar `-m "not slow"` bisa dipakai tanpa warning."""
    config.addinivalue_line(
        "markers",
        "slow: hits the real OpenAI API (one call, ~$0.0002). Deselect with -m 'not slow'.",
    )


def _free_port(preferred: int = PREFERRED_PORT) -> int:
    """Balikin `preferred` kalau nganggur, kalau nggak minta port bebas ke OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", preferred))
        except OSError:
            pass
        else:
            return preferred

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as spare:
        spare.bind(("127.0.0.1", 0))
        return spare.getsockname()[1]


def _wait_until_serving(url: str, process: subprocess.Popen, timeout: float) -> None:
    """
    Polling ke endpoint `/healthz` Streamlit sampai dia jawab.

    Polling, bukan `sleep(n)` asal: waktu boot pertama itu nggak bisa ditebak (di sini
    pandas, scikit-learn, dan chromadb keimport duluan), jadi sleep tetap bakal kependekan
    di mesin lambat dan kepanjangan di mesin cepat. Prosesnya juga diperiksa tiap putaran --
    server yang mati waktu boot harus ketauan sekarang, bukan lewat timeout 60 detik.
    """
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(
                f"streamlit exited with code {process.returncode} before serving:\n{output}"
            )
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last_error = exc
        time.sleep(0.25)

    raise TimeoutError(f"streamlit did not serve {url} within {timeout}s ({last_error})")


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def app_env() -> dict:
    """Environment buat app: sama kayak sekarang, tapi OpenAI-nya dimatiin."""
    env = dict(os.environ)
    # Kosong, bukan dihapus -- lihat catatan 1 di docstring modul.
    env["OPENAI_API_KEY"] = ""
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


@pytest.fixture(scope="session")
def streamlit_server(app_env: dict) -> str:
    """
    Nyalain `streamlit run app/main.py` sebagai subprocess, yield URL-nya, matiin pas kelar.

    Scope session: satu server buat semua tes browser. Tiap tes tetep dapet konteks browser
    baru dari pytest-playwright, jadi `st.session_state` (termasuk cache hasil investigasi)
    mulai dari kosong lagi setiap tes -- nggak ada state yang nyampur antar tes.
    """
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            APP_SCRIPT,
            "--server.headless=true",
            f"--server.port={port}",
            "--server.address=127.0.0.1",
            "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false",
        ],
        cwd=REPO_ROOT,
        env=app_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        _wait_until_serving(f"{url}/healthz", process, timeout=180)
        yield url
    finally:
        if process.poll() is None:
            if sys.platform == "win32":
                # terminate() cuma ngenain prosesnya sendiri; Streamlit bisa ninggalin anak
                # proses yang tetep megang portnya. /T ngebunuh sepohon.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture(autouse=True)
def case_files_snapshot():
    """
    Nyatet isi `outputs/case_files/` sebelum tes, hapus yang baru lahir pas teardown.

    Autouse: tes mana pun yang mencet Investigate bakal nulis case file, jadi pembersihannya
    nggak boleh nunggu diminta. Yield-nya set nama berkas SEBELUM tes, biar tes jalur error
    bisa mastiin nggak ada berkas baru sama sekali.
    """
    os.makedirs(CASE_FILE_DIR, exist_ok=True)
    before = set(os.listdir(CASE_FILE_DIR))

    yield before

    for name in set(os.listdir(CASE_FILE_DIR)) - before:
        try:
            os.remove(os.path.join(CASE_FILE_DIR, name))
        except OSError:
            # Kegagalan hapus nggak boleh bikin tes yang lulus jadi merah -- yang dites
            # bukan filesystem-nya.
            pass


@pytest.fixture
def missing_pipeline_run_csv():
    """
    Sembunyiin `outputs/pipeline_run.csv` selama satu tes, balikin lagi setelahnya.

    Di-rename, bukan dihapus: berkas itu hasil run notebook 07 yang mahal diulang, dan tes
    yang ngerusak artefak orang bukan tes yang boleh dijalanin dua kali.
    """
    backup = PIPELINE_RUN_CSV + ".uat-backup"
    existed = os.path.exists(PIPELINE_RUN_CSV)
    if existed:
        os.replace(PIPELINE_RUN_CSV, backup)
    try:
        yield PIPELINE_RUN_CSV
    finally:
        if existed:
            os.replace(backup, PIPELINE_RUN_CSV)
