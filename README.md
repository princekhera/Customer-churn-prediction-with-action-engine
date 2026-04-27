# 🎯 Marketing AI Platform

Three integrated systems for data-driven marketing:

| System | What it does |
|--------|-------------|
| **Churn Engine** | Predict which customers will churn + why + what to do |
| **Next-Best-Action Engine** | Decide the optimal action per customer in real time |
| **GenAI Marketing Copilot** | Ask natural language questions about your marketing data |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data & train all models
python train_all.py

# 3. Launch Streamlit dashboard
streamlit run dashboard/app.py

# 4. (Optional) Launch FastAPI backend
python api/main.py
# API docs at: http://localhost:8000/docs
```

---

## 📁 Project Structure

```
marketing_ai/
├── data/
│   ├── generate_data.py         ← Synthetic data generator
│   ├── churn_data.csv           ← 5,000 customers with churn labels
│   ├── nba_data.csv             ← 5,000 customers with action labels
│   ├── weekly_analytics.csv     ← 52 weeks of marketing KPIs
│   ├── segment_performance.csv  ← Segment-level metrics
│   └── schema.json              ← Metadata for GenAI copilot
│
├── churn_engine/
│   └── churn_model.py           ← GradientBoosting + Action Engine + SHAP explanations
│
├── nba_engine/
│   └── nba_model.py             ← Multi-class NBA + Contextual Bandit simulation
│
├── genai_copilot/
│   └── copilot.py               ← RAG pipeline + rule engine + OpenAI integration
│
├── api/
│   └── main.py                  ← FastAPI REST API
│
├── dashboard/
│   └── app.py                   ← Streamlit dashboard (all 3 systems)
│
├── models/                      ← Saved .pkl model files (auto-created)
├── train_all.py                 ← One-shot training script
├── requirements.txt
└── .env.example                 ← Set OPENAI_API_KEY for LLM copilot
```

---

## System 1: Churn Engine

**Model:** GradientBoostingClassifier (XGBoost-equivalent)
- **ROC-AUC: 0.956** | Accuracy: 90%
- Inputs: 12 behavioral + transactional features
- Output: churn probability + risk tier + recommended action + top 6 feature drivers

**Action Engine:**
| Risk | Trigger | Action |
|------|---------|--------|
| HIGH (>70%) + Enterprise/Pro | → | Personal call |
| HIGH (>70%) + declining spend | → | Discount offer |
| HIGH (>70%) otherwise | → | Personalized email |
| MEDIUM (40-70%) + low usage | → | Product tutorial |
| LOW (<40%) | → | No action |

**Feature Importance (top 3):**
1. `last_login_days_ago` — 39.3%
2. `login_count_30d` — 25.8%
3. `feature_usage_score` — 11.5%

---

## System 2: Next-Best-Action Engine

**Model:** Multi-class GradientBoosting
- **Accuracy: 99.8%** | Macro F1: 0.97
- 16 input features (behavioral + engagement + campaign history)
- 4 action classes: `send_email`, `push_notification`, `upsell_offer`, `do_nothing`

**Contextual Bandit:**
- Epsilon-greedy (ε=0.15) for online exploration/exploitation
- Simulates customer response feedback
- Learns best global action from reward signals
- Q-values converge after ~200 rounds

---

## System 3: GenAI Marketing Copilot

**RAG Pipeline:**
1. Intent detection from natural language query
2. Retrieval of relevant data context (weekly KPIs, segments, trends)
3. Answer generation (rule engine offline, OpenAI GPT if API key set)

**Supported query types:**
- `performance_drop` → why metrics dropped, root cause analysis
- `high_value_segment` → LTV, spend, engagement by segment
- `targeting` → segment recommendations for campaigns
- `retention` → churn alerts and win-back strategies
- `revenue` → revenue trends, best/worst periods
- `channel` → channel performance attribution

**Auto-Insight Engine:** Generates weekly marketing brief automatically.

---

## API Reference

```bash
# Health check
GET /health

# Churn prediction (single)
POST /churn/predict
Body: { "customer_id": "C001", "login_count_30d": 5, "last_login_days_ago": 45, ... }

# NBA recommendation (single)
POST /nba/recommend
Body: { "customer_id": "C001", "email_opens_30d": 4, ... }

# GenAI Copilot
POST /copilot/ask
Body: { "question": "Why did conversion drop last week?" }

# Weekly Brief
GET /copilot/brief

# Full customer profile (churn + NBA combined)
POST /full-profile
Body: { customer features }
```

---

## Adding OpenAI for LLM Copilot

```bash
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-your-key-here
```

The copilot automatically uses GPT-4o-mini when the key is present, 
and falls back to the rule-based engine without it.

---

## Replacing Synthetic Data with Real Data

1. Replace CSVs in `data/` with your production data
2. Ensure column names match those in `churn_engine/churn_model.py` → `FEATURES`
3. Re-run `python train_all.py`

For the copilot, update `data/schema.json` to describe your tables.
