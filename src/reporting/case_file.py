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

from datetime import datetime

import config
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


def case_id_for(decision: Decision, opened_at: datetime) -> str:
    """
    Bikin id kasus buat satu Decision.

    Datanya nggak punya `case_id` sama sekali (CLAUDE.md, "Known data gaps"), jadi id ini
    kita yang nyiptain. Dua syarat: dua orang yang jalanin ulang repo ini harus dapet id
    yang sama buat kasus yang sama, dan id-nya harus bisa dibalik jadi "ini nasabah siapa,
    kapan dibuka".

    TODO:
      1. Tentuin dulu Decision mana yang mau dipakai buat nyoba fungsi ini.
         Yang menarik: satu nasabah yang kena override shared-device, satu yang cuma promo,
         satu yang bersih. Tiga kasus, bukan satu.
      2. Ambil `customer_id` dari `decision`.
      3. Ubah `opened_at` jadi bagian tanggal aja, tanpa jam.
         Dua kasus nasabah yang sama di hari yang sama itu satu kasus, bukan dua.
      4. Gabungin jadi satu string dengan prefiks tetap.
         Lu putusin: formatnya. Yang penting kebaca sama manusia dan aman dipakai jadi nama
         file -- jangan ada spasi, slash, atau titik dua di dalamnya.
      5. Balikin `str`, bukan f-string yang nyimpen objek datetime.

    Hint: `strftime()`, f-string
    """
    raise NotImplementedError


def summarize_evidence(decision: Decision) -> list[str]:
    """
    Field 4 SOP-004: ringkasan bukti pemicu, lengkap sama rujukan pasal SOP-nya.

    Ini field yang bikin laporannya bisa dibantah orang. Satu baris ringkasan doang nggak
    cukup -- pembaca harus bisa lihat indikator apa yang nyala DAN dari dokumen mana
    aturannya, tanpa buka kode.

    TODO:
      1. Pakai Decision yang sama kayak di `case_id_for()`.
         Cetak dulu `decision.findings` mentah-mentah biar kelihatan isinya apa aja.
      2. Loop semua finding, ambil yang nyala doang.
         `RiskFinding` punya properti yang udah jawab "ini nyala apa nggak" -- pakai itu,
         jangan bandingin risk level manual.
      3. Buat tiap finding yang nyala, susun satu string yang mengandung: `sop_id`,
         `risk_level`, isi `indicators_matched`, dan `citations`.
      4. Urus kasus `citations` kosong secara eksplisit.
         Sitasi kosong itu informasi, bukan error. Yang haram cuma satu: nulis nomor pasal
         yang nggak dibalikin retriever.
      5. Finding yang diam jangan dibuang diam-diam.
         Lu putusin: mau ditulis "checked, nothing found" atau cukup disebut di rationale.
         Bedanya: pembaca laporan bisa bedain "udah dicek, aman" dari "nggak pernah dicek"
         atau nggak.

    String yang dibalikin fungsi ini masuk laporan, jadi isinya bahasa Inggris.

    Hint: `for`, `.is_actionable`, `", ".join()`, f-string
    """
    raise NotImplementedError


def collect_supporting_references(profile: dict) -> dict:
    """
    Field 7 SOP-004: id data mentah yang jadi rujukan (transaksi, event swap, komplain).

    Sumbernya profil L2, bukan Decision -- alasannya ada di docstring modul.

    TODO:
      1. Pakai profil dari nasabah yang sama kayak Decision di langkah sebelumnya.
         Cetak `profile.keys()` dulu, cari key mana yang isinya daftar id.
      2. Ambil ketiga daftar id itu dari profil.
      3. Lu putusin: semua id dimasukin, atau dipotong sampai sejumlah tertentu?
         Satu nasabah bisa punya puluhan transaksi sementara yang jadi bukti cuma sebagian.
         Kalau dipotong, laporan wajib nyebut jumlah aslinya -- daftar terpotong yang
         nggak bilang dia terpotong itu bohong.
      4. Balikin dict, bukan tiga variabel kepisah.
         Satu objek gampang dipindah ke JSON dan gampang ditambah key-nya nanti.
      5. Cek tipenya: `json.dumps()` harus lolos. `.tolist()` udah dipanggil di L2, jadi
         harusnya `list` biasa -- pastiin, jangan diasumsiin.

    Hint: `dict`, `len()`, slicing, `json.dumps()`
    """
    raise NotImplementedError


def collect_limitations(decision: Decision) -> list[str]:
    """
    Bagian "Limitations". Ini DI LUAR 7 field SOP-004, sengaja.

    Kenapa ada: tiga aturan SOP di repo ini cuma bisa dicek separo, dan datanya nggak
    punya kasus abu-abu. Laporan yang diam soal itu kelihatan lebih yakin daripada
    buktinya -- dan itu persis kesalahan yang paling mahal di laporan fraud.

    Isinya dua macam, jangan dicampur jadi satu daftar tanpa keterangan:

      (a) Per kasus. Gabungan field `unverified` dari semua `RiskFinding` di Decision ini.
          Datanya udah ada, tinggal dikumpulin -- L3 udah nyatet tiap potongan aturan yang
          nggak bisa dia cek.
      (b) Tetap, sama di semua laporan. Ambil dari CLAUDE.md bagian status L3, dua hal:
          sitasi SOP cuma mastiin dokumennya bener belum tentu pasalnya tepat, dan ketiga
          SOP nggak punya kasus abu-abu di dataset ini jadi aturannya belum teruji di
          kasus susah.

    TODO:
      1. Pakai Decision yang sama. Cetak `finding.unverified` dari tiap finding dulu, lihat
         bentuknya.
      2. Kumpulin `unverified` dari semua finding jadi satu daftar.
         Lu putusin: cuma dari finding yang nyala, atau dari semua finding? Cek dulu
         `policy_rules.py` -- finding yang diam ngisi `unverified` nggak.
      3. Buang duplikat, tapi jangan sampai urutannya jadi acak tiap kali dijalanin.
         Laporan yang isinya sama tapi urutannya beda tiap run susah di-diff.
      4. Tulis dua kalimat limitasi tetap sebagai konstanta modul.
         Tetap artinya nggak diitung ulang per kasus -- kalau dia tergantung isi Decision,
         berarti dia masuk kelompok (a).
      5. Gabung (a) dan (b) jadi satu `list[str]` yang dibalikin.
         Lu putusin: pembaca perlu tau mana yang per-kasus dan mana yang tetap nggak, dan
         gimana cara nunjukinnya tanpa nambah struktur baru.

    String di sini masuk laporan, jadi bahasa Inggris.

    Hint: `for`, `list`, `dict.fromkeys()`, konstanta modul
    """
    raise NotImplementedError


def build_case_file(
    decision: Decision,
    profile: dict,
    opened_at: datetime | None = None,
) -> dict:
    """
    Rakit satu case file lengkap: 7 field SOP-004 + limitations, siap di-`json.dumps()`.

    Fungsi ini yang manggil keempat fungsi di atas. Dia nggak boleh ngitung ulang risk
    level atau action -- dua-duanya disalin apa adanya dari `decision`.

    TODO:
      1. Pakai pasangan (decision, profile) dari nasabah yang sama.
         Nasabah beda antara dua argumen = laporan yang buktinya bukan punya dia. Lu
         putusin: mau dicek dan di-`raise`, atau dipercaya aja? Pikirin siapa yang manggil
         fungsi ini nanti.
      2. Isi `opened_at` kalau `None`.
         Lu putusin: pakai waktu sekarang, atau wajib dikasih pemanggil? Konsekuensinya:
         waktu-sekarang bikin laporan yang sama nggak pernah identik dua kali.
      3. Bangun dict-nya dengan key persis `CASE_FILE_FIELDS`, satu per satu:
         - field 1: `case_id_for()` dan tanggalnya sebagai string.
         - field 2: `customer_ids` bentuknya daftar walau isinya satu.
           SOP-004 nulis "Customer ID(s)", dan grain per-nasabah itu keputusan repo ini,
           bukan bentuk final laporan fraud.
         - field 3: kategori dari finding yang nyala. Dua hal harus lu urus di sini:
           kosakata `policy_rules.CATEGORY_*` beda sama tulisan SOP-004, dan bisa lebih
           dari satu kategori nyala barengan. Lu putusin: dipetakan gimana, dan kalau lebih
           dari satu yang nyala, semuanya ditulis atau dipilih satu.
         - field 4: `summarize_evidence()`.
         - field 5 dan 6: disalin dari `decision`. Cek `Decision` bisa ngeluarin level yang
           nggak ada di kosakata SOP-004 nggak -- kalau iya, itu keputusan lu mau diapain.
         - field 7: `collect_supporting_references()`.
         - limitations: `collect_limitations()`.
      4. Lu putusin: `decision.rationale`, `l1_band`, `l2_risk_level`, `confidence_score`
         masuk laporan nggak? SOP-004 nggak minta, tapi tanpa itu pembaca nggak bisa lihat
         keputusannya datang dari mana. Kalau dimasukin, taruh di key sendiri di luar 7
         field -- jangan nyelundup ke dalam field SOP.
      5. Verifikasi hasilnya lewat `json.dumps()` sebelum lanjut.
         Ini satu-satunya cara tau ada numpy scalar atau datetime yang nyempil.

    Hint: `dict`, `datetime.now()`, `.isoformat()`, `json.dumps()`
    """
    raise NotImplementedError


def render_case_file_markdown(case: dict) -> str:
    """
    Case file dict -> teks markdown yang dibaca manusia.

    Dipisah dari `build_case_file()` karena dua pembaca beda: JSON buat mesin (diff,
    agregasi, tes), markdown buat orang yang harus mutusin. Digabung, salah satunya pasti
    jadi korban.

    TODO:
      1. Pakai dict hasil `build_case_file()` dari langkah sebelumnya.
      2. Tulis judul laporan yang mengandung `case_id`.
      3. Loop `CASE_FILE_FIELDS`, cetak tiap field sebagai satu bagian bernomor.
         Nomor dan urutannya ngikutin SOP-004 -- itu template resminya, bukan selera.
      4. Urus value yang bentuknya `list` dan `dict` biar nggak kecetak sebagai repr Python.
         `['a', 'b']` di laporan itu tanda kode yang bocor ke output.
      5. Pisahin bagian limitations dari 7 field SOP-004 secara visual.
         Dia bukan field SOP -- pembaca harus bisa lihat itu tanpa dijelasin.
      6. Balikin satu `str`, jangan `print()` di dalam fungsi.
         Fungsi yang nge-print nggak bisa ditulis ke file dan nggak bisa dites.

    Teks laporan bahasa Inggris.

    Hint: `"\\n".join()`, `enumerate()`, `isinstance()`, f-string
    """
    raise NotImplementedError


def save_case_file(case: dict, output_dir: str = config.CASE_FILE_DIR) -> str:
    """
    Tulis case file ke disk, balikin path file yang ditulis.

    TODO:
      1. Pakai case dict yang udah jadi.
      2. Bikin `output_dir` kalau belum ada.
      3. Susun nama file dari `case_id`.
         Makanya `case_id_for()` dilarang ngandung spasi dan slash.
      4. Tulis JSON-nya. `encoding='utf-8'` ditulis eksplisit -- default Windows itu
         cp1252 dan dia bakal error di karakter non-ASCII pertama.
      5. Tulis versi markdown-nya juga lewat `render_case_file_markdown()`.
      6. Lu putusin: file yang udah ada ditimpa atau di-`raise`? Case file itu artefak yang
         digenerate ulang, bukan dokumen yang diedit tangan -- tapi keputusan itu tetep
         harus disengaja.
      7. Balikin path-nya, jangan cuma `print()`.
         Pemanggil butuh path itu buat langkah berikutnya.

    Hint: `os.makedirs()`, `os.path.join()`, `open(..., encoding='utf-8')`, `json.dump()`
    """
    raise NotImplementedError
