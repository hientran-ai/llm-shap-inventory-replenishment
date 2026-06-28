# LLM-Augmented SHAP for Explainable Inventory Replenishment

An end-to-end Explainable Agentic AI framework for intelligent inventory replenishment that unifies **demand forecasting**, **autonomous decision-making**, and **explainable AI (XAI)**. Using the M5 Forecasting dataset, we train LightGBM and XGBoost models for demand prediction, feed them into an Inventory Agent that computes safety stock and reorder points, and translate SHAP attributions into human-readable business insights via **Llama 3**.

---

## 🏗️ System Architecture

```
M5 Dataset — 30,490 products × 1,947 days (Store CA_1: 3,049 SKUs)
        │
        ▼
 Feature Engineering
 ├── Lag features      : lag_7, lag_14, lag_28
 ├── Rolling stats     : 7/28-day mean & std
 ├── Calendar features : weekday, month, SNAP days, holiday flags
 └── Pricing features  : sell_price, price_norm, markdown %
        │
        ▼
 ┌──────────────────┐     ┌─────────────────┐
 │   LightGBM ✅    │     │    XGBoost      │
 │  RMSE: 2.1026   │     │  RMSE: 2.1175   │
 │  MAE:  1.0915   │     │  MAE:  1.0977   │
 │  R²:   0.6335   │     │  R²:   0.6283   │
 └────────┬─────────┘     └────────┬────────┘
          └──────────┬─────────────┘
                     ▼
           Demand Forecast
         (28-day horizon, 85,372 records)
                     │
                     ▼
          Inventory Agent
          ├── Safety Stock (SS)
          ├── Reorder Point (ROP)
          └── Order Quantity (OQ)
                     │
                     ▼
     SHAP Explainability Layer (TreeExplainer)
     ├── Global: mean |SHAP| feature importance
     └── Local: per-product waterfall decomposition
                     │
                     ▼
     LLM Narrative Layer (Llama 3)
     ├── Overall Assessment
     ├── Root-Cause Analysis
     └── Operational Recommendations
```

---

## 📊 Model Performance

| Metric | LightGBM | XGBoost |
|--------|----------|---------|
| **RMSE** | **2.1026** | 2.1175 |
| **MAE**  | **1.0915** | 1.0977 |
| **R²**   | **0.6335** | 0.6283 |

- Training set: **5,747,365** samples (Store CA_1, prior to 2016-03-28)
- Validation set: **85,372** product-day records (28-day out-of-sample horizon)
- **LightGBM** selected as the core forecasting engine based on superior accuracy and lower memory footprint

---

## 📁 Repository Structure

```
├── Data_feature.ipynb                # Feature engineering pipeline
├── Forecast_model(LIGHTGBM).ipynb    # LightGBM training & evaluation
├── XGBOOST.ipynb                     # XGBoost training & evaluation
├── SHAP_Pipeline_Forecast.ipynb      # Global & local SHAP explainability
├── Inventory_agent(m5+sup).ipynb     # Inventory decision agent (SS, ROP, OQ)
├── Experiments_Evaluation.ipynb      # Model comparison & metrics
├── lgb_model.txt                     # Saved LightGBM model
└── xgb_model.json                    # Saved XGBoost model
```

---

## ⚙️ Feature Engineering

| Feature Group | Features | Count |
|---|---|---|
| Temporal Lag | lag_7, lag_14, lag_28 | 5 |
| Rolling Window Statistics | 7/28-day mean, std, min, max | 8 |
| Calendar & Event | weekday, month, SNAP days, holiday flags | 5 |
| Pricing & Promotional | sell_price, price_norm, markdown % | 2 |

---

## 🤖 Inventory Agent Logic

| Component | Formula |
|---|---|
| **Safety Stock (SS)** | `z × σ_demand × √(Lead Time)` — z = 1.65 (95% service level) |
| **Reorder Point (ROP)** | `avg_daily_demand × Lead Time + SS` |
| **Order Quantity (OQ)** | `total_forecast_demand + SS − current_inventory` |
| **Decision** | `REORDER` if `current_inventory ≤ ROP`, else `ENOUGH` |

---

## 🔍 SHAP Explainability

Global feature importance (mean |SHAP value|) across the validation set:

| Feature | Mean \|SHAP\| | Role |
|---|---|---|
| `sell_price` | 0.59 | Demand suppressor at high prices |
| `rmean_28` | 0.27 | Long-term demand trend |
| `lag_7` | 0.27 | Short-term sales momentum |
| `price_norm` | 0.25 | Price elasticity signal |

SHAP values are computed using **TreeExplainer** (exact, polynomial-time) and structured into:
- **Global explanation**: bar chart + beeswarm summary plot across 500 validation instances
- **Local explanation**: waterfall plot per product–date instance, top-8 contributors

---

## 🤖 MARL Agents (Simulation Layer)

Three MARL paradigms evaluated for autonomous multi-product replenishment:

| Agent | Type | Coordination |
|---|---|---|
| **MADDPG** | Continuous control | Centralized critic (CTDE) |
| **MADQN** | Discrete Q-learning | Decentralized (independent) |
| **QMIX** | Discrete + monotonicity constraint | Centralized mixing network |

---

## 💬 LLM Narrative Layer (Llama 3)

SHAP attributions are converted into a structured 3-section manager report:

1. **Overall Assessment** — demand outlook vs. baseline
2. **Root-Cause Analysis** — top SHAP drivers explained in business terms
3. **Operational Recommendations** — concrete restocking actions

> *Example output for FOODS_3_421_CA_1 (18 Apr 2016, predicted demand: 0.17 units):*
> "Expected demand is projected well below historical baselines due to unfavorable local pricing and weak short-term momentum. Immediate warehouse stock expansion is unjustified; promotional price readjustments or strategic inventory re-allocation should instead be deployed to stimulate latent product velocity."

---

## 📦 Dataset

🔗 [M5 Forecasting — Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

Download and place in your working directory:
```
sales_train_evaluation.csv   # 30,490 × 1,947 daily unit sales
calendar.csv                 # 1,969 × 14 time-based metadata
sell_prices.csv              # 6,841,121 × 4 weekly item prices
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/hientran-ai/llm-shap-inventory-replenishment.git
cd llm-shap-inventory-replenishment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
#    Data_feature.ipynb
#    → Forecast_model(LIGHTGBM).ipynb
#    → SHAP_Pipeline_Forecast.ipynb
#    → Inventory_agent(m5+sup).ipynb
```

---

## 🛠️ Tech Stack

`Python` `LightGBM` `XGBoost` `SHAP` `Llama 3` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Jupyter`

---

## 👥 Authors

- **Tran Thi Thu Hien** — [hientran-ai](https://github.com/hientran-ai)

---

## 📜 Citation

```bibtex
@inproceedings{hien2026llmshap,
  title     = {LLM-Augmented SHAP for Explainable Inventory Replenishment},
  author    = {Tran Thi Thu Hien et al.},
  booktitle = {Proceedings of the 2nd International Conference on Computer Sciences,
               Engineering, and Technology Innovation (ICoCSETI 2026)},
  year      = {2026},
  publisher = {IEEE}
}
```
