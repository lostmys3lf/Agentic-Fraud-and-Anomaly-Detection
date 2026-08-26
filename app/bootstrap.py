"""
View layer (Streamlit) -- penyiapan sys.path dan urutan import.

WAJIB diimport paling awal oleh setiap entry script di `app/` (main.py dan tiap file di
`app/pages/`), sebelum `streamlit` atau modul `src/` mana pun.

Dua tugasnya:

1. Naruh repo root dan `src/` di `sys.path`. `streamlit run app/main.py` bikin `sys.path[0]`
   nunjuk ke folder `app/`, jadi `import config` dan `import pipeline.state` nggak ketemu
   tanpa langkah ini. Path-nya diturunin dari `__file__`, bukan dari cwd -- app-nya harus
   tetep jalan dari mana pun dipanggil.

2. Baca `.env` supaya `OPENAI_API_KEY` sampai ke `reporting/narrative.py`. Key-nya nggak
   pernah dibaca ulang atau ditampilin di UI -- cuma nempel di environment proses ini.
   `override=False` (default python-dotenv) sengaja dibiarkan: kalau environment yang
   manggil udah nyetel `OPENAI_API_KEY` (termasuk nyetel jadi string kosong), nilai itu
   yang menang. Itu yang bikin tes E2E bisa matiin panggilan OpenAI tanpa nyentuh `.env`.

CATATAN LINGKUNGAN, dan ini beda dari notebook -- jangan disamain:
Aturan "import chromadb sebelum pandas" (lihat CLAUDE.md) NGGAK BISA dipenuhi di dalam
proses Streamlit. `streamlit run` ngimport streamlit duluan, dan streamlit narik pandas
sebelum baris pertama script kita jalan. Konsekuensinya cuma satu dan sempit: `upsert()` ke
koleksi Chroma bisa matiin proses (exit 139, tanpa traceback). Makanya `app/resources.py`
NGGAK PERNAH nge-upsert -- dia ngecek koleksinya udah keisi dan nyuruh user bikin indeksnya
dari notebook 03 kalau kosong. Query (`collection.query()`) aman, dan itu satu-satunya yang
dipakai app ini.
"""

from __future__ import annotations

import os
import sys

# Tetep diimport paling atas walau di proses Streamlit pandas udah keduluan. Alasannya
# dokumentasi: modul ini yang jadi penanda urutan import buat pembaca berikutnya.
import chromadb  # noqa: F401,E402

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for _path in (PROJECT_ROOT, SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
