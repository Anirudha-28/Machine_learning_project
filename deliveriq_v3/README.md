# DeliverIQ — AI-Powered Delivery Prediction & Business Intelligence Platform

A Brazilian e-commerce logistics platform that predicts delivery times with a trained ML model, answers customer questions through a tool-calling AI agent, and gives admins a business-intelligence assistant and dashboard — all running on local infrastructure with no per-query API cost.

---

## Problem Statement

Existing problems:

- Customers receive generic delivery estimates that ignore real operating conditions.
- Support staff repeatedly answer the same delivery and policy questions manually.
- Order status and delivery reasoning are not self-served.
- Business owners have no fast way to see revenue, top categories, or risk states.

### Our Solution

DeliverIQ builds an intelligent delivery-logistics platform that:

- Predicts delivery days using a trained machine learning model.
- Answers customer questions through a tool-calling AI agent.
- Provides administrators with a business-intelligence assistant and dashboard.
- Runs AI workloads locally using Ollama, eliminating per-query API costs.

---

## Key Features

- 🚚 **ML-based delivery prediction**
- 🤖 **Customer AI assistant**
- 🧠 **Tool-calling AI agents**
- 🔎 **Retrieval-Augmented Generation (RAG)**
- 📊 **Business Intelligence dashboard**
- 👨‍💼 **Admin AI assistant**
- 🔐 **Role-based authentication**
- 🗄️ **MySQL prediction and order logging**
- 🏠 **Fully local LLM inference using Ollama**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Programming | Python |
| Backend | Flask |
| Data / ML | Pandas, NumPy, CatBoost, LightGBM |
| LLM Runtime | Ollama |
| LLM | Qwen2.5:3B |
| Embeddings | Nomic-Embed-Text |
| RAG | Custom RAG + cosine similarity |
| Database | MySQL |
| Frontend | HTML, CSS, JavaScript |
| Authentication | Flask Sessions |
| Version Control | Git, GitHub |

---

## Dataset

- **Source:** Real Brazilian Olist e-commerce fields combined with operational logistics fields.
- **Records:** 200,600
- **Raw columns:** 42
- **Engineered features:** 93
- **Prediction target:** `delivery_days`

### Operational Data

The dataset includes:

- Orders
- Customers
- Sellers
- Products
- Payments
- Courier partner
- Vehicle type
- Warehouse location and capacity
- Warehouse processing hours
- Traffic index
- Weather condition
- Rainfall
- Road condition
- Fuel price index
- Seller dispatch time
- Estimated dispatch time
- Holiday and festival indicators

---

## Feature Engineering Pipeline

The prediction pipeline includes:

- Haversine distance between seller and customer coordinates.
- Target encoding for product categories.
- One-hot encoding across categorical operational fields.
- Shipping-mode-specific delivery estimation formulas.
- Operational feature sampling from historical scenarios.
- Automatic feature alignment using the trained model's feature schema.

The final inference pipeline rebuilds the **93-column training schema** before prediction.

---

## Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| **CatBoost** | **1.2549** | **1.5706** | **0.8384** |
| LightGBM | 1.3257 | 1.6645 | 0.8183 |
| Linear Regression | 1.4062 | 1.7834 | 0.7914 |

### Why CatBoost?

CatBoost was selected as the best-performing model because it:

- Handles categorical features effectively.
- Captures non-linear relationships.
- Reduces overfitting through ordered boosting.
- Handles missing values robustly.
- Provides a reliable feature schema for inference.

### Operational Feature Importance

Removing operational features such as:

- Traffic
- Rainfall
- Fuel price
- Warehouse processing
- Seller dispatch time
- Rural route
- Courier breakdown

reduced model performance to:

**MAE:** 1.447  
**RMSE:** 1.822  
**R²:** 0.782

This demonstrates that operational features provide meaningful predictive value.

### Top Drivers

The major drivers of delivery time include:

1. Distance
2. Weather
3. Rainfall
4. Traffic index
5. Warehouse processing
6. Courier and vehicle type
7. Road condition

---

## System Architecture

```text
                    ┌──────────────────────┐
                    │      Frontend        │
                    │ HTML / CSS / JS      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Flask Backend     │
                    │       app.py         │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
       ┌────────────┐   ┌────────────┐   ┌────────────┐
       │ ML Model   │   │ AI Agents  │   │    RAG     │
       │ CatBoost   │   │ ReAct Loop │   │ Knowledge  │
       │ 93 Features│  │ Customer   │   │ Base       │
       └─────┬──────┘   │ Admin      │   └─────┬──────┘
             │          └─────┬──────┘         │
             │                │                │
             ▼                ▼                ▼
       ┌────────────────────────────────────────────┐
       │              Ollama Runtime                │
       │              Qwen2.5:3B                    │
       └─────────────────────┬──────────────────────┘
                             │
                             ▼
                       ┌───────────┐
                       │   MySQL   │
                       └───────────┘
