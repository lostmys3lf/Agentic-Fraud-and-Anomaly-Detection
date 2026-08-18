"""
Single source of truth for paths and thresholds used across all layers (L1-L4).
No module should hardcode a file path or a SOP numeric threshold -- import it from here.
"""

import os
import pathlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
POLICY_DOCS_DIR = os.path.join(PROJECT_ROOT, "fraud_policy_docs")

CUSTOMERS_CSV = os.path.join(DATA_DIR, "customers.csv")
TRANSACTIONS_CSV = os.path.join(DATA_DIR, "transactions.csv")
SIM_SWAP_EVENTS_CSV = os.path.join(DATA_DIR, "sim_swap_events.csv")
COMPLAINT_NOTES_CSV = os.path.join(DATA_DIR, "complaint_notes.csv")

# Event tables are temporal -- always split by time, never randomly (CLAUDE.md Modeling bar).
# Train: 2026-04-01 to 2026-05-31. Test: 2026-06-01 to 2026-06-30.
TIME_SPLIT_DATE = "2026-06-01"

# --- SOP-001 SIM Swap ---
SIM_SWAP_DISTANCE_KM_THRESHOLD = 100
SIM_SWAP_LOGIN_CHANGE_HOURS_THRESHOLD = 2  # proxy via hours_since_last_login_change, see CLAUDE.md gap notes
SIM_SWAP_DEVICE_CHANGES_12MO_THRESHOLD = 2
SIM_SWAP_VERIFY_HOLD_HOURS = 1   # 1 indicator -> verify and hold
SIM_SWAP_ESCALATE_MINUTES = 30   # 2+ indicators -> HIGH, block and escalate within this window

# --- SOP-002 Topup / Voucher ---
SHARED_DEVICE_ACCOUNTS_MEDIUM = 2   # same device/instrument across 2 accounts/24h = MEDIUM
SHARED_DEVICE_ACCOUNTS_HIGH = 3     # 3+ accounts/24h = HIGH
NIGHT_HOUR_START = 1                # 01:00 (inclusive)
NIGHT_HOUR_END = 5                  # 05:00 (exclusive)
ROUND_AMOUNT_DIVISOR = 50_000
SHARED_DEVICE_WINDOW_HOURS = 24     # the "within a 24-hour window" in SOP-002 section 2

# --- SOP-003 Promo Bundle ---
PROMO_REDEMPTIONS_LOW = 2       # 1-2 per 90d = LOW
PROMO_REDEMPTIONS_MEDIUM = 3    # 3 = MEDIUM
PROMO_REDEMPTIONS_HIGH = 4      # 4+ with cancel-recreate = HIGH
PROMO_BLOCK_DAYS = 180
PROMO_WINDOW_DAYS = 90          # the "within a 90-day period" the counts above are measured over

# --- L2 Investigation: RAG over fraud_policy_docs ---
# Chroma runs in-process and writes one folder -- no server, no account, no network
# after the embedding model is cached, so a student can re-run this offline.
CHROMA_PERSIST_DIR = os.path.join(OUTPUT_DIR, "chroma_sop_index")
CHROMA_COLLECTION_SOP = "fraud_sop"

# Chroma's DefaultEmbeddingFunction: all-MiniLM-L6-v2 exported to ONNX, 384 dimensions.
# Same model sentence-transformers would give us, but it runs on onnxruntime instead of
# torch (~80 MB download, cached in ~/.cache/chroma, vs a ~2 GB torch install).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# The corpus is 4 short documents (~13 sections total), so asking for many hits just
# returns the whole corpus and makes the retrieval comparison meaningless.
RAG_TOP_K = 3

# NOT an SOP number -- a provisional cutoff for "this hit is too weak to cite".
# It has to be tuned against real queries in notebook 03 before anything relies on it.
RAG_MIN_SIMILARITY = 0.20

# --- L3 Decision: bands for the L1 confidence score ---
# NOT SOP numbers, and NOT two points on one scale. LOW and HIGH answer different
# questions, so they get tuned against different metrics:
#
#   LOW   "below this we are willing to Auto-Approve" -- a recall guarantee. Pick it from
#         the operating point where nothing fraudulent in the test window scored beneath
#         it. Setting it too high closes a real case with nobody looking.
#   HIGH  "above this a human should look" -- a precision budget, sized by how many cases
#         a reviewer can actually work through. In the option-C matrix an L1 HIGH never
#         blocks anything on its own (Block always needs an L2 HIGH carrying a cited SOP
#         clause), so this is an Escalate-grade cut -- not the near-perfect precision you
#         would demand before auto-blocking an account.
#
# One pair per model, not one pair for the repo. A probability threshold describes one
# model's calibration, not a policy: 0.70 from the transaction RandomForest says nothing
# about a swap model trained at a different grain with a different base rate (2.69% vs
# 10.0%). Same reason MLFLOW_EXPERIMENT_* is split further down -- if the runs must not be
# mixed, the thresholds cannot be either.
#
# PROVISIONAL, and the only guessed numbers left in this file: replace them with the
# operating point read off the June split in notebook 02 (the tuned RandomForest sits at
# precision 0.262 / recall 1.00, PR-AUC 0.788) and write the (precision, recall) pair you
# picked next to each one. Note the leak while doing it -- promo rows are trivially
# separable, so a threshold tuned on this data is partly tuned on the generator.
L1_TX_SCORE_LOW = 0.30
L1_TX_SCORE_HIGH = 0.70

# No SIM-swap model exists yet -- notebook 02 Bagian 2 is still an unrun template. None on
# purpose: score_band() has to fail loudly rather than quietly read a swap score against
# the transaction model's calibration.
L1_SWAP_SCORE_LOW = None
L1_SWAP_SCORE_HIGH = None

# Which pair a score gets read against. The key is the model that produced the score, so
# the caller has to say where its number came from -- see decide(..., l1_source=...).
L1_SCORE_BANDS = {
    "transaction": (L1_TX_SCORE_LOW, L1_TX_SCORE_HIGH),
    "sim_swap": (L1_SWAP_SCORE_LOW, L1_SWAP_SCORE_HIGH),
}

# --- L4 Reporting: SOP-004 case files ---
# One folder per run of the reporting layer, under outputs/ so .gitignore already covers
# it. Case files are generated artifacts, never edited by hand -- regenerate instead.
CASE_FILE_DIR = os.path.join(OUTPUT_DIR, "case_files")

# --- MLflow experiment tracking (L1) ---
# SQLite backend, not the older file-based one: MLflow 3.x put the filesystem tracking
# store in maintenance mode (it raises unless MLFLOW_ALLOW_FILE_STORE=true), and the
# model registry only works on a database backend anyway. Still entirely local -- one
# .db file, no server process -- so the notebook stays runnable on Colab and offline.
# Browse the runs locally with:
#   mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db
MLFLOW_DB_PATH = os.path.join(OUTPUT_DIR, "mlflow.db")
MLFLOW_ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "mlartifacts")

# MLflow wants URIs, not Windows paths. as_posix() gives forward slashes and as_uri()
# percent-encodes -- both matter here because this project's path contains spaces and
# an "&", which would otherwise be read as URI syntax.
MLFLOW_TRACKING_URI = "sqlite:///" + pathlib.Path(MLFLOW_DB_PATH).as_posix()
MLFLOW_ARTIFACT_URI = pathlib.Path(MLFLOW_ARTIFACTS_DIR).as_uri()

# One experiment per detection aspect -- transaction and SIM-swap models have
# different grain and different metrics, so mixing their runs would make the
# comparison table meaningless.
MLFLOW_EXPERIMENT_SIM_SWAP = "sim_swap_detection"
MLFLOW_EXPERIMENT_TRANSACTION = "transaction_detection"
