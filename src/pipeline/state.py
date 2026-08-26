"""
Bentuk data yang ngalir lewat graph (controller layer, di atas L1-L4).

Dua benda yang beda tujuan, jangan dicampur:

  PipelineState      berubah tiap node, isinya satu nasabah, ringan.
  PipelineResources  dibangun SEKALI di luar graph, sama buat semua nasabah, berat
                     (4 DataFrame + koleksi Chroma + model .pkl). Nggak pernah masuk state.

Kenapa dipisah: kalau DataFrame ikut nempel di state, tiap node ngopi ulang seluruh dataset
dan jejak auditnya jadi nggak bisa diserialisasi.

Satu-satunya nilai di state yang nggak langsung JSON-able itu `decision` (dataclass
`Decision`). Sengaja disimpan sebagai objek, bukan dict, karena `build_case_file()` di L4
minta objeknya. Yang ngurus serialisasi `state_to_dict()` di bawah -- dipanggil pas mau
nulis jejaknya, bukan pas node lagi jalan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypedDict

import pandas as pd

import config
import data_io
from decision.decide import Decision
from detection import model as detection_model
from features import transaction_features
from investigation import sop_retriever


class PipelineState(TypedDict, total=False):
    """Isi state per nasabah. `total=False` karena field diisi bertahap tiap node."""

    customer_id: str                  # input, diisi pemanggil sebelum graph jalan
    confidence_score: float | None    # L1: skor max, None kalau nasabah nol transaksi
    l1_source: str                    # L1: model mana yang ngeluarin skor itu
    profile: dict                     # L2: hasil build_customer_profile()
    decision: Decision                # L3: hasil decide()
    case: dict                        # L4: case file SOP-004
    case_path: str                    # L4: path file .json (file .md-nya sebelahan)
    error: str | None                 # jalur gagal, None kalau lancar


@dataclass
class PipelineResources:
    """Barang yang di-load sekali, dipakai semua nasabah. Nggak pernah masuk state."""

    customers: pd.DataFrame
    transactions: pd.DataFrame
    sim_swap_events: pd.DataFrame
    complaint_notes: pd.DataFrame
    sop_collection: Any          # koleksi Chroma
    transaction_model: Any       # RandomForest hasil notebook 02
    encoder: Any                 # OneHotEncoder yang di-fit di data train
    feature_columns: list[str]   # urutan kolom persis kayak waktu latih


def build_resources() -> PipelineResources:
    """
    Load semuanya sekali. Panggil ini di luar graph, terus oper hasilnya ke build_pipeline().

    Kenapa encoder-nya di-fit ulang di sini, bukan dibaca dari file: `save_model()` cuma
    nyimpen model, encoder-nya nggak ikut ke-pickle. Padahal model cuma ngerti kolom hasil
    encoder yang di-fit di data train. Jadi split waktunya diulang persis (`TIME_SPLIT_DATE`)
    supaya kolomnya sama. Kalau split-nya beda, kolomnya beda, dan prediksinya ngaco tanpa
    ngeluarin error.
    """
    customers = data_io.load_customers()
    transactions = data_io.load_transactions()
    sim_swap_events = data_io.load_sim_swap_events()
    complaint_notes = data_io.load_complaint_notes()

    collection = sop_retriever.get_sop_collection()
    if collection.count() == 0:
        chunks = sop_retriever.chunk_documents(sop_retriever.load_sop_documents())
        collection = sop_retriever.build_sop_index(chunks, rebuild=True)

    transaction_model = detection_model.load_model(config.TRANSACTION_MODEL_PATH)

    features = transaction_features.build_legit_features(transactions, customers)
    train, test = data_io.time_based_split(features)
    x_train, _x_test, encoder = transaction_features.encode_features(train, test)

    return PipelineResources(
        customers=customers,
        transactions=transactions,
        sim_swap_events=sim_swap_events,
        complaint_notes=complaint_notes,
        sop_collection=collection,
        transaction_model=transaction_model,
        encoder=encoder,
        feature_columns=list(x_train.columns),
    )


def state_to_dict(state: PipelineState) -> dict:
    """Salinan state yang aman buat `json.dumps()`. Dataclass `Decision` -> dict biasa."""
    out = dict(state)
    if "decision" in out:
        out["decision"] = asdict(out["decision"])
    return out
