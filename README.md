# Agentic Fraud & Anomaly Detection

Case study: agentic fraud and anomaly investigation for a telco, built on a **synthetic**
labeled dataset. Personal learning / DS-AI mentoring project — not production software.

## Flow

`Detect -> Investigate -> Decide -> Report`

- **L1 Detection** — ML / anomaly scoring producing a confidence score.
- **L2 Investigation** — RAG over `fraud_policy_docs/` (SOP-001 to SOP-004) plus historical pattern lookup.
- **L3 Decision** — weighs L1 + L2 findings against policy thresholds, emits Auto-Approve / Escalate / Block.
- **L4 Reporting** — compiles a structured, auditable case file in SOP-004 format.

Fraud types in scope: reseller rings, SIM-swap account takeover, promo abuse.

## Data

All data under `data/` is **synthetic**, generated for this case study — never real, customer,
or production data. Event tables span 2026-04-01 to 2026-06-30, joined on `customer_id`
across 1,200 customers.

| file | rows | label |
|---|---|---|
| `customers.csv` | 1,200 | — |
| `transactions.csv` | 7,634 | `is_fraud_label`, 2.69% positive |
| `sim_swap_events.csv` | 200 | `is_fraud_label`, 10.0% positive |
| `complaint_notes.csv` | 280 | `related_to_fraud_case` (proxy only), 21.4% positive |

Known data gaps (no case-level label, no `case_id`, several SOP indicators not directly
computable) and a documented label-leakage pattern are tracked in `CLAUDE.md`.

## Repo structure

```
config.py                     paths + all SOP thresholds
data/                          read-only source CSVs
fraud_policy_docs/             L2 RAG corpus (SOP-001..004)
src/
  data_io.py                   generic load/parse
  features/                    L1, case-specific feature builders
  detection/                   L1 baseline rules, model, evaluation
  investigation/                L2 RAG retriever + pattern lookup
  decision/                     L3 policy thresholds -> decision
  reporting/                    L4 SOP-004 case file builder
notebooks/
  01_data_understanding.ipynb
  02_detection_baseline_and_model.ipynb
  03_agent_pipeline_end_to_end.ipynb
outputs/                       notebook artifacts (gitignored)
```

## Status

- **L1 (Detection)** — baseline + model logic prototyped in `02_detection_baseline_and_model.ipynb`
  for the transaction fraud task (time-based split, Random Forest, MLflow-tracked). SIM-swap
  section and migration into `src/` still pending.
- **L2 (Investigation)** — `src/investigation/` scaffolded; RAG pipeline (Chroma + embeddings)
  design proven in a prior session, being re-derived step by step in
  `03_agent_pipeline_end_to_end.ipynb`.
- **L3 (Decision) / L4 (Reporting)** — not yet built.

## Setup

Python 3.11.9, virtualenv in `.venv`.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Experiment tracking (MLflow) and the RAG vector store (Chroma) both run locally, no external
services — see `CLAUDE.md` for tracking URI, persist paths, and Windows-specific version pins.
