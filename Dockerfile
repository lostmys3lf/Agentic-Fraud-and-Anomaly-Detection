# Container buat view layer (app/, Streamlit). Target deploy: Google Cloud Run.
#
# Prinsip yang dipegang di sini:
#   1. Image ini SELF-CONTAINED buat runtime -- data, SOP, model, dan indeks Chroma semuanya
#      ikut masuk. Nggak ada satu pun yang ditarik dari network pas container start.
#   2. Rahasia NGGAK PERNAH masuk image. OPENAI_API_KEY diinjeksi pas runtime.
#   3. Image ini cuma bisa di-build dari folder lokal, BUKAN dari hasil git clone repo ini --
#      outputs/chroma_sop_index/ dan outputs/*.pkl sengaja nggak di-push ke GitHub.

# Samain sama Python lokal (3.11.9) supaya pin di requirements-app.txt beneran valid.
# -slim, bukan -alpine: alpine pakai musl libc, dan wheel numpy/scikit-learn/onnxruntime
# dibangun buat glibc. Di alpine dia bakal compile dari source berjam-jam, atau gagal.
FROM python:3.11-slim

# PYTHONUNBUFFERED wajib di Cloud Run: tanpa itu print() nyangkut di buffer dan log lu
# baru muncul pas container mati -- persis pas lu paling butuh log-nya.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# User biasa, bukan root. Dibikin di awal supaya COPY bisa langsung --chown, jadi nggak perlu
# `chown -R` belakangan (itu bikin layer duplikat sebesar isi folder yang di-chown).
RUN useradd --create-home --uid 1000 appuser

# /srv, bukan /app. Soalnya folder kita sendiri namanya app/, jadi WORKDIR /app bakal
# ngasilin /app/app/main.py -- jalan sih, tapi bikin bingung tiap kali baca path di log.
WORKDIR /srv

# --- Layer 1: dependency ----------------------------------------------------------------
# requirements di-copy SENDIRIAN dulu, sebelum kode. Docker nge-cache tiap baris, dan cache
# hangus begitu file yang di-COPY berubah. Kalau kode ikut di-copy di sini, tiap kali lu ubah
# satu baris di render.py, pip install ulang dari nol (~5 menit). Dipisah begini, pip cuma
# jalan lagi kalau requirements-app.txt-nya sendiri yang berubah.
COPY requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

# --- Layer 2: kode + data ----------------------------------------------------------------
# config.py naruh PROJECT_ROOT = dirname(__file__), dan app/bootstrap.py naruh PROJECT_ROOT =
# parent dari app/. Dua-duanya cuma cocok kalau struktur folder di dalam container PERSIS
# sama kayak di repo. Makanya ini di-copy satu-satu, bukan `COPY . .` -- biar keliatan apa
# yang beneran dibutuhin, dan biar nggak ada file nyasar yang kebawa diam-diam.
COPY --chown=appuser:appuser config.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app/ ./app/

# data/ BUKAN cuma bahan training. src/pipeline/state.py:71-74 baca keempat CSV ini tiap kali
# pipeline jalan, buat L2 pattern lookup. Tanpa ini app-nya mati di request pertama.
COPY --chown=appuser:appuser data/ ./data/

# Korpus RAG L2. Dibutuhin walau indeksnya udah jadi, soalnya isi teks chunk-nya dibaca
# balik pas bikin sitasi SOP di laporan L4.
COPY --chown=appuser:appuser fraud_policy_docs/ ./fraud_policy_docs/

# Indeks Chroma yang udah jadi. Ini yang gua bilang "alasannya nanti pas ngomongin cold start":
# app/resources.py SENGAJA nggak pernah upsert (aturan chromadb-sebelum-pandas nggak bisa
# dipenuhi di dalam proses Streamlit), dia cuma ngecek count() dan berhenti kalau kosong.
# Jadi indeksnya HARUS udah ada di image. Nggak bisa dibangun pas container start.
COPY --chown=appuser:appuser outputs/chroma_sop_index/ ./outputs/chroma_sop_index/

# Model L1 + hasil batch yang dibaca halaman batch_data. Halaman itu sengaja nggak pernah
# recompute: 1200 customer = 1200 panggilan OpenAI buat satu klik.
COPY --chown=appuser:appuser outputs/transaction_fraud_model.pkl ./outputs/
COPY --chown=appuser:appuser outputs/sim_swap_fraud_model.pkl ./outputs/
COPY --chown=appuser:appuser outputs/pipeline_run.csv ./outputs/
COPY --chown=appuser:appuser outputs/l3_decision_report.csv ./outputs/
COPY --chown=appuser:appuser outputs/l4_case_summary.csv ./outputs/

# L4 save_case_file() nulis ke sini. Dia udah makedirs(exist_ok=True) sendiri, tapi foldernya
# dibikin di sini supaya kepemilikannya jelas -- kalau nggak, appuser nulis ke folder yang
# induknya punya root. Isinya EPHEMERAL di Cloud Run: filesystem container itu tmpfs (RAM),
# hilang tiap instance mati. Kalau case file perlu awet, tujuannya GCS, bukan folder ini.
RUN mkdir -p outputs/case_files && chown appuser:appuser outputs/case_files

USER appuser

# --- Layer 3: pemanasan model embedding --------------------------------------------------
# DefaultEmbeddingFunction()-nya Chroma itu all-MiniLM-L6-v2 versi ONNX, ~80 MB, dan dia
# di-download PAS DIPAKAI PERTAMA KALI ke ~/.cache/chroma -- bukan pas pip install.
# Tanpa baris ini, request pertama ke Cloud Run bakal nunggu download 80 MB, dan itu gagal
# total kalau instance-nya kena batasan network keluar. Dipanggil di sini biar cache-nya
# ikut ke-bake jadi layer image. Ditaruh SESUDAH `USER appuser` supaya HOME-nya
# /home/appuser, jadi cache-nya kebaca sama proses yang nanti beneran jalan.
RUN python -c "from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()(['warmup'])"

# Cloud Run nyuntik $PORT sendiri; default ini cuma biar `docker run` lokal enak.
ENV PORT=8080
EXPOSE 8080

# Shell form (bukan JSON array) DISENGAJA -- ${PORT} butuh shell buat di-expand. Kalau
# ditulis CMD ["streamlit", ..., "--server.port=${PORT}"], streamlit nerima string
# "${PORT}" mentah dan langsung mati.
#   --server.address=0.0.0.0 : wajib. Default streamlit cuma listen di localhost, dan dari
#                              luar container itu artinya nggak ada yang jawab.
#   --server.headless=true   : jangan coba buka browser, di container nggak ada browser.
CMD streamlit run app/main.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
