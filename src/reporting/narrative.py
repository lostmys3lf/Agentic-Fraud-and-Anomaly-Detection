"""
L4 Reporting -- ringkasan naratif case file, ditulis LLM (OpenAI).

Tambahan opsional buat `case_file.py`. Input: dict case file yang **udah jadi** dari
`build_case_file()`. Output: satu paragraf bahasa Inggris yang masuk ke field baru
`narrative_summary`, di LUAR 7 field SOP-004.

Kenapa inputnya case file jadi, bukan data mentah: biar LLM-nya nggak punya ruang buat
ngarang. Semua angka, indikator dan sitasi udah dihitung L1-L3 dan dikunci di dict itu;
tugas modelnya cuma nyusun kalimat dari bahan yang udah ada.

**Aturan L4 nggak berubah**: `risk_level` dan `recommended_action` tetep salinan dari
`Decision`. Modul ini dilarang nulis ulang dua field itu, dan `attach_narrative()` ngecek
itu pakai `assert`. Kalau LLM-nya bilang lain, yang salah LLM-nya, bukan L3.

Degradasi: kalau `OPENAI_API_KEY` nggak ada atau API-nya error/timeout, `generate_narrative()`
balikin `None` dan case file tetep jadi -- cuma `narrative_summary`-nya diisi kalimat penanda.
Laporan yang gagal ditulis gara-gara layanan pihak ketiga mati itu regresi, bukan fitur.

Catatan environment (biar nggak abis satu run buat nemuin):
- `openai==3.3.1` dan `python-dotenv==1.2.2` ada di `requirements.txt`, dua-duanya udah
  diverifikasi nggak nggeser satu pin pun yang ada.
- `.env` harus berbentuk `OPENAI_API_KEY=sk-...`, satu baris, tanpa spasi di sekitar `=`,
  kalau nggak `python-dotenv` nggak bisa baca.
- Key dibaca dari env var, NGGAK PERNAH ditulis di kode atau di notebook.
"""

from __future__ import annotations

import json
import os

import openai
from openai import OpenAI

import config


def build_narrative_prompt(case: dict) -> str:
    """
    Susun prompt buat LLM dari satu dict case file.

    Fungsi murni, nggak manggil API apa pun -- jadi bisa dites tanpa key dan tanpa internet,
    dan prompt-nya bisa di-review kayak kode biasa.
    """
    # Cuma field yang kepake buat nulis paragraf. supporting_data_references dibuang
    # (list id mentah, model gampang ngutip id acak dari situ) dan limitations juga
    # dibuang -- dia udah jadi section sendiri di markdown, kalau ikut masuk prompt
    # bakal ditulis ulang di dalam narasi.
    payload = {
        "case_id": case["case_id"],
        "customer_ids": case["customer_ids"],
        "fraud_category": case["fraud_category"],
        "triggering_evidence": case["triggering_evidence"],
        "risk_level": case["risk_level"],
        "recommended_action": case["recommended_action"],
        "rationale": case["decision_trace"]["rationale"],
    }

    instructions = f"""You are a fraud analyst writing the summary paragraph of an internal case file.
Write one paragraph of at most {config.NARRATIVE_MAX_WORDS} words describing what was found and why the case
was rated at this risk level.

Rules:
- Use ONLY the facts given in the input. Never add a number, date, ID, or fact that
  is not present there.
- Do not change, question, or re-derive the risk level or the recommended action.
  State them as given.
- Do not give advice, next steps, or opinions beyond the recommended action.
- Plain factual prose. No bullet points, no headings, no markdown.
- The data is synthetic; do not describe it as real customer activity."""

    # Bagian data diserialisasi, bukan ditempel f-string panjang: kalau outputnya aneh,
    # gampang keliatan field mana yang salah masuk.
    return instructions + "\n\nCase data:\n" + json.dumps(payload, indent=2)


def generate_narrative(
    case: dict,
    model: str = config.OPENAI_MODEL,
    timeout_seconds: float = config.OPENAI_TIMEOUT_SECONDS,
) -> str | None:
    """
    Panggil OpenAI sekali, balikin teks narasinya. `None` kalau nggak bisa atau gagal.

    Balikin `None`, jangan ngeraise: pemanggilnya (`attach_narrative`) harus tetep bisa
    bikin case file tanpa narasi. Satu API timeout nggak boleh matiin satu batch.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI()
    prompt = build_narrative_prompt(case)

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=500,
            timeout=timeout_seconds,
        )
        return resp.output_text.strip()
    # Sengaja error jaringan/API doang, bukan `Exception` telanjang -- bug di prompt
    # sendiri (KeyError, dsb) harus tetep kelihatan, jangan ketelen diam-diam.
    except (openai.APIError, openai.APITimeoutError,
            openai.APIConnectionError, openai.AuthenticationError) as e:
        print(f"narrative failed for {case.get('case_id')}: {e}")
        return None


def attach_narrative(case: dict, model: str = config.OPENAI_MODEL) -> dict:
    """
    Balikin salinan `case` plus key `narrative_summary`.

    Dipanggil setelah `build_case_file()` dan sebelum `save_case_file()`, jadi narasinya
    ikut kesimpan di `.json` maupun `.md`.
    """
    out = dict(case)

    text = generate_narrative(case, model=model)
    # Penanda eksplisit, bukan string kosong: di laporan, kosong nggak bisa dibedain
    # sama narasi yang gagal digenerate.
    out["narrative_summary"] = text if text else "Narrative not generated."

    # Invarian L4: layer ini nggak mutusin apa-apa, cuma nambah prosa.
    assert out["risk_level"] == case["risk_level"], "narrative must not change risk_level"
    assert out["recommended_action"] == case["recommended_action"], \
        "narrative must not change recommended_action"

    return out
