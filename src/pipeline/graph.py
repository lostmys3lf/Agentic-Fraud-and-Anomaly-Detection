"""
Graph L1 -> L2 -> L3 -> L4 (controller layer).

Modul ini nggak mutusin apa pun soal fraud. Tiap node cuma manggil fungsi yang udah ada di
layer-nya masing-masing, terus naruh hasilnya di state. Kalau ada `if` di sini yang ngubah
risk level atau action, itu bug -- sama persis kayak aturan di `reporting/case_file.py`.

Grain-nya satu nasabah = satu kali `invoke()`, ngikutin grain case file yang udah dikunci di
L4. Batch = loop biasa di sisi pemanggil.

CATATAN LINGKUNGAN (bukan bagian yang diturunin, ini fakta yang harganya satu run gagal):
`sop_retriever` narik `chromadb`, dan di Windows `chromadb` HARUS keimport sebelum `pandas`
atau proses mati tanpa traceback (exit 139) waktu `upsert()`. Makanya import chromadb ada di
paling atas di bawah ini. `langgraph` sendiri netral -- diuji di dua urutan, dua-duanya aman.

Diturunin di `notebooks/07_pipeline_orchestration.ipynb`, dimigrasi ke sini 2026-08-25.
"""

from __future__ import annotations

import chromadb  # noqa: F401  -- harus di atas pandas, lihat catatan di docstring modul

from functools import partial
from typing import Literal

from langgraph.graph import END, StateGraph

from decision.decide import decide
from detection import model as detection_model
from features import transaction_features
from investigation.pattern_lookup import build_customer_profile
from pipeline.state import PipelineResources, PipelineState
from reporting import case_file, narrative


def score_l1(state: PipelineState, resources: PipelineResources) -> dict:
    """L1: skor tiap transaksi nasabah, ambil yang tertinggi."""
    # Nasabah yang nggak ada di customers.csv dicegat di sini, sebelum L2, soalnya
    # build_customer_profile() bakal ngeraise KeyError buat id kayak gitu.
    if state["customer_id"] not in set(resources.customers["customer_id"]):
        return {"error": f"unknown customer_id: {state['customer_id']}"}

    cust_tx = resources.transactions[
        resources.transactions["customer_id"] == state["customer_id"]
    ]

    # Nol transaksi bukan berarti bersih -- skornya None, dan L3 bacanya BORDERLINE.
    # l1_source tetep "transaction": modelnya emang itu, cuma nggak ada baris buat diskor.
    if cust_tx.empty:
        return {"confidence_score": None, "l1_source": "transaction"}

    feats = transaction_features.build_legit_features(cust_tx, resources.customers)
    x = transaction_features.transform_features(
        feats, resources.encoder, resources.feature_columns
    )
    scores = detection_model.predict_score(resources.transaction_model, x)

    # max, bukan rata-rata: satu transaksi 0.90 di antara 99 transaksi 0.01 rata-ratanya
    # 0.019 -- di bawah L1_TX_SCORE_LOW, alias Auto-Approve buat fraud beneran.
    # float() karena .max() balikin numpy.float64 yang nggak bisa di-json.dumps().
    return {"confidence_score": float(scores.max()), "l1_source": "transaction"}


def investigate_l2(state: PipelineState, resources: PipelineResources) -> dict:
    """L2: rakit profil nasabah (device sharing, promo, komplain, riwayat swap)."""
    # Sengaja tanpa try/except. Nasabah nggak dikenal udah dicegat router setelah L1;
    # kalau di sini masih KeyError, router-nya yang bocor dan itu harus kelihatan.
    profile = build_customer_profile(
        state["customer_id"],
        resources.customers,
        resources.transactions,
        resources.sim_swap_events,
        resources.complaint_notes,
    )
    return {"profile": profile}


def decide_l3(state: PipelineState, resources: PipelineResources) -> dict:
    """L3: gabungin skor L1 + profil L2 + korpus SOP jadi satu `Decision`."""
    # Node ini ngoper doang. Matriks 3x4, override shared-device dan penanganan skor None
    # semuanya udah tinggal di decision/.
    return {
        "decision": decide(
            confidence_score=state["confidence_score"],
            profile=state["profile"],
            collection=resources.sop_collection,
            l1_source=state["l1_source"],
        )
    }


def report_l4(state: PipelineState, resources: PipelineResources) -> dict:
    """L4: rakit case file SOP-004, tulis .json + .md ke config.CASE_FILE_DIR."""
    # Dua argumen, bukan satu: field 7 SOP-004 minta id transaksi/event/komplain, dan
    # daftar id itu cuma ada di profil L2, nggak dibawa Decision.
    case = case_file.build_case_file(state["decision"], state["profile"])
    # Satu baris, di antara build dan save: narasinya ikut kesimpan ke .json dan .md.
    # Gagal manggil API nggak fatal -- attach_narrative() tetep balikin case file utuh.
    case = narrative.attach_narrative(case)
    path = case_file.save_case_file(case)
    return {"case": case, "case_path": path}


def handle_error(state: PipelineState, resources: PipelineResources) -> dict:
    """Ujung buat nasabah yang nggak bisa diproses. Nggak nulis case file apa pun."""
    # Nggak ngitung ulang apa-apa: pesannya udah disiapin score_l1. Node ini yang bikin
    # kegagalannya kelihatan di state akhir, dan tempat naruh logging kalau nanti perlu.
    message = state.get("error") or f"unknown customer_id: {state['customer_id']}"
    return {"error": message}


def route_after_l1(state: PipelineState) -> Literal["investigate_l2", "handle_error"]:
    """Satu-satunya percabangan di graph ini. Cuma baca state, balikin nama node."""
    # confidence_score None BUKAN error: itu nasabah nol transaksi, dan L3 udah maping-in
    # ke BORDERLINE lalu Escalate. Kalau dibelokin ke handle_error, kasus yang mestinya
    # Escalate malah berhenti tanpa laporan.
    if state.get("error"):
        return "handle_error"
    return "investigate_l2"


def build_pipeline(resources: PipelineResources):
    """Rakit dan compile graph-nya. Balikannya punya `.invoke()` dan `.stream()`."""
    g = StateGraph(PipelineState)

    # partial() nempelin `resources` ke tiap node, jadi LangGraph tinggal ngasih `state`.
    # Ini yang bikin resources nggak usah ikut masuk ke state.
    g.add_node("score_l1", partial(score_l1, resources=resources))
    g.add_node("investigate_l2", partial(investigate_l2, resources=resources))
    g.add_node("decide_l3", partial(decide_l3, resources=resources))
    g.add_node("report_l4", partial(report_l4, resources=resources))
    g.add_node("handle_error", partial(handle_error, resources=resources))

    g.set_entry_point("score_l1")

    # Nama di kanan harus sama persis sama nama yang didaftarin di add_node().
    g.add_conditional_edges(
        "score_l1",
        route_after_l1,
        {"investigate_l2": "investigate_l2", "handle_error": "handle_error"},
    )
    g.add_edge("investigate_l2", "decide_l3")
    g.add_edge("decide_l3", "report_l4")

    # Dua ujung ke END. Node yang nggak nyambung ke END bikin graph nggak berhenti.
    g.add_edge("report_l4", END)
    g.add_edge("handle_error", END)

    return g.compile()
