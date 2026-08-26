"""
L4 Reporting -- satu Decision jadi satu case file format SOP-004.

Layer terakhir. Input: `Decision` dari L3 (`decision.decide.decide()`) dan `profile` dict
dari L2 (`investigation.pattern_lookup.build_customer_profile()`). Output: satu dict yang
aman buat `json.dumps()`, versi markdown-nya, dan file di `config.CASE_FILE_DIR`.

**Modul ini nggak mutusin apa-apa.** Risk level dan action udah final di L3. Kalau di sini
ada `if` yang ngubah salah satunya, itu bug, bukan fitur. Tugas L4 cuma tiga: nyalin,
nyusun sesuai template, nulis ke disk. Alasannya audit -- laporan yang diem-diem beda dari
keputusan yang dieksekusi bikin seluruh jejaknya nggak kepakai.

Kenapa `build_case_file()` butuh DUA argumen, bukan cuma Decision: field 7 SOP-004 minta
transaction/event/complaint ids, dan `Decision` nggak bawa satupun. Daftar id itu adanya di
profil L2 (`transaction_ids`, `swap_event_ids`, `complaint_ids`). Decision bawa bukti yang
*dipakai aturan*, profil bawa data mentah yang *dirujuk laporan* -- dua hal beda.

Grain kasus: **satu nasabah = satu case** (diputusin 2026-08-18). Konsekuensi yang harus
ditulis apa adanya di evaluasi, bukan disembunyiin: tiap akun dalam satu ring shared-device
dapet case file sendiri-sendiri yang isinya mirip, karena data ini nggak punya kunci yang
nyambungin mereka jadi satu kasus (lihat "Known data gaps" di CLAUDE.md).

Semua yang masuk case file harus lolos `json.dumps()`. Nggak ada numpy scalar, nggak ada
DataFrame, nggak ada objek datetime -- tanggal disimpen sebagai string.

Catatan urutan import, load-bearing di Windows: modul ini nge-import `decision.decide`
buat type hint `Decision`, dan itu narik `chromadb` lewat `sop_retriever`. Notebook yang
manggil modul ini harus tetep naruh `import chromadb` di atas `import pandas`.
"""

import json
import os
from datetime import datetime

import config
from decision import policy_rules
from decision.decide import Decision


# --- vocabulary ---------------------------------------------------------------
# Urutan dan nama key laporan. Ini kontrak antara build_case_file(),
# render_case_file_markdown() dan save_case_file() -- ketiganya baca daftar yang sama,
# jadi nambah field cukup di satu tempat.
#
# Field 1 SOP-004 ("Case ID and date opened") dipecah jadi dua key: satu id, satu tanggal.
# Digabung jadi satu string, tanggalnya nggak bisa dipakai buat sortir atau filter.
# "limitations" BUKAN bagian dari 7 field SOP-004 -- lihat collect_limitations().
CASE_FILE_FIELDS: tuple[str, ...] = (
    "case_id",                    # SOP-004 field 1a
    "date_opened",                # SOP-004 field 1b
    "customer_ids",               # SOP-004 field 2
    "fraud_category",             # SOP-004 field 3
    "triggering_evidence",        # SOP-004 field 4
    "risk_level",                 # SOP-004 field 5
    "recommended_action",         # SOP-004 field 6
    "supporting_data_references",  # SOP-004 field 7
    "limitations",                # tambahan repo ini, di luar SOP-004
)

# Limitasi yang sama di semua laporan, nggak tergantung isi Decision -- makanya konstanta,
# bukan diitung ulang per kasus. Dua-duanya dari temuan L2/L3 di CLAUDE.md.
FIXED_LIMITATIONS: tuple[str, ...] = (
    ("SOP citations confirm the retrieved document is the right one, but not that the "
     "cited clause is the exact clause that applies."),
    ("No SOP in this dataset has a borderline case, so these rules are verified against "
     "their own specification only and remain untested on hard cases."),
)

# Field 3 SOP-004 nulis kategorinya "SIM Swap / Topup Fraud / Promo Abuse", sementara
# policy_rules pakai kosakata sendiri (CATEGORY_*). Laporan ngikut SOP, bukan ngikut nama
# konstanta di kode -- makanya dipetakan di sini. Kategori yang nggak kedaftar bakal
# KeyError, bukan diloloskan apa adanya: nama konstanta yang bocor ke laporan itu bug yang
# nggak keliatan sampai ada yang baca laporannya.
#
# shared_device -> "Topup Fraud" karena aturannya emang dari SOP-002 (Topup Fraud
# Detection); "shared device" itu indikatornya, bukan nama kategorinya.
_CATEGORY_LABELS: dict[str, str] = {
    policy_rules.CATEGORY_SIM_SWAP: "SIM Swap",
    policy_rules.CATEGORY_SHARED_DEVICE: "Topup Fraud",
    policy_rules.CATEGORY_PROMO_ABUSE: "Promo Abuse",
}

# Judul tiap bagian markdown, nomornya ikut SOP-004 (bukan selera). Field 1 nggak ada di
# sini karena dia gabungan dua key (case_id + date_opened) dan dicetak manual.
_SOP_SECTIONS: tuple[tuple[str, str], ...] = (
    ("2. Customer ID(s)", "customer_ids"),
    ("3. Fraud Category", "fraud_category"),
    ("4. Triggering Evidence", "triggering_evidence"),
    ("5. Risk Level", "risk_level"),
    ("6. Recommended Action", "recommended_action"),
    ("7. Supporting Data References", "supporting_data_references"),
)


def case_id_for(decision: Decision, opened_at: datetime) -> str:
    """
    Bikin id kasus buat satu Decision.

    Datanya nggak punya `case_id` sama sekali (CLAUDE.md, "Known data gaps"), jadi id ini
    kita yang nyiptain. Dua syarat: dua orang yang jalanin ulang repo ini harus dapet id
    yang sama buat kasus yang sama, dan id-nya harus bisa dibalik jadi "ini nasabah siapa,
    kapan dibuka".

    Tanggalnya dipotong sampai hari doang: dua kasus nasabah yang sama di hari yang sama
    itu satu kasus, bukan dua. Formatnya juga aman dipakai jadi nama file -- nggak ada
    spasi, slash, atau titik dua, karena `save_case_file()` bikin nama file dari sini.
    """
    customer_id = decision.customer_id
    opened_at_str = opened_at.strftime("%Y%m%d")
    return f"Case-{customer_id}-{opened_at_str}"


def summarize_evidence(decision: Decision) -> list[str]:
    """
    Field 4 SOP-004: ringkasan bukti pemicu, lengkap sama rujukan pasal SOP-nya.

    Ini field yang bikin laporannya bisa dibantah orang. Satu baris ringkasan doang nggak
    cukup -- pembaca harus bisa lihat indikator apa yang nyala DAN dari dokumen mana
    aturannya, tanpa buka kode.

    Dua keputusan yang ada di sini:
      - Finding yang diam nggak dibuang diam-diam. Kalau nggak ada satupun yang nyala,
        laporannya tetep nulis "checked, nothing found" -- biar pembaca bisa bedain
        "udah dicek, aman" dari "nggak pernah dicek".
      - Sitasi kosong ditulis apa adanya, bukan dikarang. Yang haram cuma satu: nyebut
        nomor pasal yang nggak dibalikin retriever.

    String yang dibalikin fungsi ini masuk laporan, jadi isinya bahasa Inggris.
    """
    finding_nyala = []
    for f in decision.findings:
        if f.is_actionable:
            finding_nyala.append(f)

    summarized = []
    if not finding_nyala:
        summarized.append(f"Customer {decision.customer_id} checked, nothing found.")
    else:
        for x in finding_nyala:
            # Tuple -> satu string. Dipisah rapi biar kebaca kalau indikatornya lebih dari satu.
            indicators = "; ".join(x.indicators_matched)
            citations = "; ".join(x.citations)
            if x.citations:
                suatu_finding = f"[{x.sop_id} | {x.risk_level}] {indicators} (Ref: {citations})"
            else:
                suatu_finding = f"[{x.sop_id} | {x.risk_level}] {indicators} (Ref: there are no citations)"
            summarized.append(suatu_finding)
    return summarized


def collect_supporting_references(profile: dict) -> dict:
    """
    Field 7 SOP-004: id data mentah yang jadi rujukan (transaksi, event swap, komplain).

    Sumbernya profil L2, bukan Decision -- alasannya ada di docstring modul.

    Semua id dimasukin tanpa dipotong: di data ini satu nasabah paling banyak belasan
    transaksi, jadi laporannya masih kebaca. Jumlahnya tetep ditulis di key `n_*` biar
    pembaca nggak usah ngitung sendiri, dan biar ketauan kalau suatu saat daftarnya
    dipotong tapi jumlahnya nggak ikut berubah.
    """
    return {
        "transaction_ids": profile["transaction_ids"],
        "n_transaction_ids": len(profile["transaction_ids"]),
        "swap_event_ids": profile["swap_event_ids"],
        "n_swap_event_ids": len(profile["swap_event_ids"]),
        "complaint_ids": profile["complaint_ids"],
        "n_complaint_ids": len(profile["complaint_ids"]),
    }


def collect_limitations(decision: Decision) -> list[str]:
    """
    Bagian "Limitations". Ini DI LUAR 7 field SOP-004, sengaja.

    Kenapa ada: tiga aturan SOP di repo ini cuma bisa dicek separo, dan datanya nggak
    punya kasus abu-abu. Laporan yang diam soal itu kelihatan lebih yakin daripada
    buktinya -- dan itu persis kesalahan yang paling mahal di laporan fraud.

    Isinya dua macam:
      (a) Per kasus -- gabungan field `unverified` dari semua `RiskFinding` di Decision ini.
      (b) Tetap -- `FIXED_LIMITATIONS`, sama di semua laporan.

    Semua finding diloop, nggak difilter `is_actionable`, karena finding yang diam emang
    nggak pernah ngisi `unverified` (lihat `policy_rules.py`) -- filternya bakal jadi
    baris yang nggak ngefek apa-apa.

    Duplikatnya dibuang pakai `dict.fromkeys()`, bukan `set()`: `set` urutannya nggak
    dijamin, dan laporan yang isinya sama tapi urutannya beda tiap run susah di-diff.
    """
    unverified_data = []
    for f in decision.findings:
        if f.unverified:
            unverified_data.extend(f.unverified)

    clean_unverified_data = list(dict.fromkeys(unverified_data))
    return clean_unverified_data + list(FIXED_LIMITATIONS)


def build_case_file(
    decision: Decision,
    profile: dict,
    opened_at: datetime | None = None,
) -> dict:
    """
    Rakit satu case file lengkap: 7 field SOP-004 + limitations, siap di-`json.dumps()`.

    Fungsi ini yang manggil keempat fungsi di atas. Dia nggak ngitung ulang risk level
    atau action -- dua-duanya disalin apa adanya dari `decision`.

    Tiga keputusan yang ada di sini:
      - Dua argumen dicek nasabahnya cocok, kalau beda langsung `raise`. Nasabah beda
        antara Decision dan profil = laporan yang buktinya punya orang lain, dan itu
        nggak bakal ketauan dari hasilnya karena bentuk laporannya tetep bener.
      - `opened_at` opsional, default waktu sekarang. Dibikin parameter (bukan `now()`
        langsung di dalam) supaya waktunya bisa dikunci pas dites -- kalau nggak, dua
        panggilan yang isinya sama nggak akan pernah identik.
      - `decision_trace` ditaruh di key sendiri DI LUAR 7 field SOP-004. SOP-004 nggak
        minta, tapi tanpa itu pembaca nggak bisa lihat "Block" ini datang dari L1 yang
        mana dan L2 bilang apa. Dia nggak boleh nyelundup ke dalam field SOP.
    """
    if decision.customer_id != profile["customer_id"]:
        raise ValueError("Decision and profile must refer to the same customer.")
    if opened_at is None:
        opened_at = datetime.now()

    # Field 3: kategori dari finding yang nyala doang, pakai kosakata SOP-004. Bisa lebih
    # dari satu nyala bareng, jadi semuanya ditulis -- milih satu bakal ngilangin bukti
    # yang udah kadung ada di field 4.
    categories = [_CATEGORY_LABELS[f.category] for f in decision.findings if f.is_actionable]

    return {
        "case_id": case_id_for(decision, opened_at),
        "date_opened": opened_at.strftime("%Y-%m-%d"),
        # SOP-004 nulis "Customer ID(s)". Grain per-nasabah itu keputusan repo ini, bukan
        # bentuk final laporan fraud -- makanya tetep list walau isinya satu.
        "customer_ids": [decision.customer_id],
        "fraud_category": categories,
        "triggering_evidence": summarize_evidence(decision),
        "risk_level": decision.risk_level,
        "recommended_action": decision.action,
        "supporting_data_references": collect_supporting_references(profile),
        "limitations": collect_limitations(decision),
        "decision_trace": {
            "confidence_score": decision.confidence_score,
            "l1_source": decision.l1_source,
            "l1_band": decision.l1_band,
            "l2_risk_level": decision.l2_risk_level,
            "override_applied": decision.override_applied,
            "rationale": decision.rationale,
        },
    }


def _format_value(value) -> str:
    """dict/list -> teks markdown. Repr Python (['a', 'b']) nggak boleh bocor ke laporan."""
    if isinstance(value, list):
        if not value:
            return "(none)"
        return "\n".join(f"- {item}" for item in value)
    if isinstance(value, dict):
        lines = []
        for key, sub in value.items():
            if isinstance(sub, list):
                sub_text = ", ".join(str(x) for x in sub) if sub else "(none)"
            else:
                sub_text = str(sub)
            lines.append(f"- {key}: {sub_text}")
        return "\n".join(lines)
    if value is None:
        return "(not available)"
    return str(value)


def render_case_file_markdown(case: dict) -> str:
    """
    Case file dict -> teks markdown SOP-004 buat dibaca manusia.

    Dipisah dari `build_case_file()` karena pembacanya beda: JSON buat mesin (diff,
    agregasi, tes), markdown buat orang yang harus mutusin. Digabung, salah satunya pasti
    jadi korban.

    Fungsi ini balikin `str`, nggak `print()`. Yang nge-print nggak bisa ditulis ke file
    dan nggak bisa dites.

    Teks laporannya bahasa Inggris.
    """
    parts = [f"# Fraud Case File: {case['case_id']}", ""]

    parts.append("## 1. Case ID and Date")
    parts.append(f"- Case ID: {case['case_id']}")
    parts.append(f"- Date opened: {case['date_opened']}")
    parts.append("")

    for title, key in _SOP_SECTIONS:
        parts.append(f"## {title}")
        parts.append(_format_value(case[key]))
        parts.append("")

    # Dipisah garis biar pembaca langsung lihat dua bagian ini di luar 7 field SOP-004.
    parts.append("---")
    parts.append("")
    # Case file lama (dibikin sebelum layer narasi ada) nggak punya key ini, jadi
    # section-nya cuma muncul kalau ada -- renderer-nya nggak boleh mati gara-gara itu.
    if "narrative_summary" in case:
        parts.append("## Narrative Summary (not part of SOP-004)")
        parts.append(_format_value(case["narrative_summary"]))
        parts.append("")
    parts.append("## Limitations (not part of SOP-004)")
    parts.append(_format_value(case["limitations"]))
    parts.append("")
    parts.append("## Decision Trace (not part of SOP-004)")
    parts.append(_format_value(case["decision_trace"]))

    return "\n".join(parts)


def save_case_file(case: dict, output_dir: str = config.CASE_FILE_DIR) -> str:
    """
    Tulis case file (JSON + markdown) ke disk, balikin path file JSON-nya.

    Dua format sekaligus: JSON buat diproses ulang, markdown buat dibaca. Nama filenya
    dari `case_id`, makanya `case_id_for()` dilarang ngandung spasi atau slash.

    `encoding='utf-8'` ditulis eksplisit karena default Windows itu cp1252 dan dia bakal
    error di karakter non-ASCII pertama.

    File yang udah ada sengaja ditimpa: case file itu artefak yang digenerate ulang, bukan
    dokumen yang diedit tangan. Kalau nggak ditimpa, isinya jadi basi tanpa ada yang tau.
    """
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{case['case_id']}.json")
    md_path = os.path.join(output_dir, f"{case['case_id']}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_case_file_markdown(case))

    # Path-nya dibalikin, bukan cuma di-print: pemanggil butuh itu buat langkah berikutnya.
    return json_path
