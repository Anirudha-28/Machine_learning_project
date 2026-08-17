DeliverIQ — AI-Powered Delivery Prediction & Business Intelligence Platform
A Brazilian e-commerce logistics platform that predicts delivery times with a trained ML model, answers customer questions through a tool-calling AI agent, and gives admins a business-intelligence assistant and dashboard — all running on local infrastructure with no per-query API cost.

---

Problem Statement
Existing problems:

- Customers receive generic delivery estimates that ignore real operating conditions

- Support staff repeatedly answer the same delivery and policy questions manually

- Order status and delivery reasoning are not self-served — every query needs a human

- Business owners have no fast way to see revenue, top categories, or risk states

**Our solution** builds a delivery-logistics platform that:

- Predicts delivery days with a trained ML model using real operational data

- Answers customer questions through a tool-calling AI agent

- Gives admins a business-intelligence assistant + dashboard

- Runs entirely on local infrastructure (Ollama) — no per-query API cost

---

Tech Stack
| Layer | Technology |

|---|---|

| Backend | Python, Flask |

| Data / ML | Pandas, NumPy, CatBoost, LightGBM |

| LLM Runtime | Ollama (Qwen2.5:3B) |

| Embeddings | Nomic-Embed-Text |

| Database | MySQL |

| Frontend | HTML, CSS, JavaScript |

---

Dataset
- **Source:** Real Brazilian Olist e-commerce fields (orders, customers, sellers, products, payments) combined with operational fields — courier partner & vehicle type, warehouse location/capacity/processing hours, traffic index, weather condition, rainfall, road condition, fuel price index, seller/estimated dispatch time, holiday/festival flags.

- **Records:** 200,600

- **Raw columns:** 42

- **Engineered features:** 93

- **Prediction target:** delivery\_days

---

Feature Engineering Pipeline
- Haversine distance between seller and customer coordinates

- Target encoding — product category → mean historical delivery\_days

- One-hot encoding across 10 categorical fields (priority, shipping mode, weather, road condition, warehouse, courier partner, vehicle type, state, etc.)

- Calibrated formula per shipping mode: estimated\_delivery\_days = slope × distance\_km + intercept

- Operational fields (traffic, weather, courier, warehouse) sampled from 40 real historical scenarios per prediction, so the model explores its full decision tree instead of one fixed leaf

- app.py reorders columns to model.feature\_names\_ at inference time as a safety net against schema drift

---

Model Performance
| Model | MAE | RMSE | R² Score |

|---|---|---|---|

| **CatBoost (best)** | **1.2549** | **1.5706** | **0.8384** |

| LightGBM | 1.3257 | 1.6645 | 0.8183 |

| Linear Regression | 1.4062 | 1.7834 | 0.7914 |

Why CatBoost:

- Handles categorical features natively — no manual target/one-hot leakage risk

- Captures non-linear relationships between distance, traffic, weather, and delivery time

- Ordered boosting reduces overfitting versus standard gradient boosting

- Robust to missing values in operational fields

- Exposes feature\_names\_, enabling safe column alignment at inference

Operational feature impact: Removing operational fields (traffic, rainfall, fuel price, warehouse processing, seller dispatch, rural route, courier breakdown) drops CatBoost to MAE 1.447 / RMSE 1.822 / R² 0.782 — confirming these features drive real predictive lift, not noise.

Top drivers of delivery time: distance (Haversine), weather & rainfall, traffic index, warehouse processing, courier & vehicle type, road condition.

---

Architecture
1. Prediction Flow (Flask + ML)

1. User logs in (session-based auth; role = customer or admin)

2. User selects category, state, city, shipping mode, and priority on the product page

3. FeatureBuilder rebuilds the 93-column training schema (distance, encodings, operational sampling)

4. Columns are reordered to model.feature\_names\_ as a safety net

5. The CatBoost model predicts delivery\_days, returning mean/min/max

6. The prediction is logged to MySQL (predictions table in schema.sql)

7. The result is shown on the UI / dashboard

2. Customer AI Assistant — Tool-Calling Agent
A manual ReAct loop (Thought → Action → Observation) in agent.py — max 3 steps, zero extra dependencies, works with any Ollama model.

Tool registry:

- predict\_delivery — runs the ML model for a new delivery estimate

- get\_order\_by\_id — looks up a specific past order from MySQL

- get\_my\_recent\_orders — fetches the customer's most recent orders

- search\_knowledge — retrieves policy/FAQ answers via RAG

The tool's observation is fed back to Ollama (Qwen2.5:3B), which either calls another tool or returns a final\_answer — a warm, 2–5 sentence reply grounded only in real tool output, never invented.

3. Retrieval-Augmented Generation (RAG)
- *Knowledge base:** courier policy, shipping policy, FAQs, delivery rules (plain .txt files in knowledge\_base/)

- Chunking: 120-word overlapping chunks, 20-word overlap

- Embedding: Ollama — nomic-embed-text

- Storage: local embedding store (vector\_store.pkl) — no external vector DB

- Retrieval: cosine similarity search, top-k most relevant chunks

- Generation: Qwen2.5 LLM (Ollama) answers using retrieved context — fully local, no internet

- Reliability fallback: if the vector store isn't built yet or Ollama is unreachable, the app automatically falls back to keyword search over the same knowledge-base files — the chatbot never goes silent

4. Admin Intelligence — Dashboard & Agent
Admin dashboard(from dataset + MySQL):

- Revenue summary — total revenue, avg order value, freight cost %

- Top categories by revenue & average delivery days

- Top couriers, top cities, top sellers

- State-wise revenue & delivery breakdown

- Shipping-mode performance (economy/standard/express/same-day)

- Fraud/risk states — highest average delivery days (bottleneck signal)

Admin agent flow (admin\_agent.py):

- Reachable only behind @admin\_required

- ReAct loop (max 4 steps), separate tool registry — never exposed to customers

- admin\_tools.py pulls live numbers from CSV/MySQL — revenue, categories, states, shipping, risk

- Qwen2.5 LLM produces a professional, number-grounded final answer

### System Overview


Frontend — HTML / CSS / JS  (index, product, dashboard, chatbot, admin\_chatbot)

&#x20;       │

Flask Application (app.py) — routes, sessions, role-based access

&#x20;       │

&#x20;  ┌────┴────┬─────────────┬──────────────┐

ML Prediction  AI Agents      RAG          FeatureBuilder

CatBoost(.pkl) agent.py       knowledge\_base/\*.txt

93 features    admin\_agent.py nomic-embed-text

&#x20;              ReAct loop     vector\_store.pkl

&#x20;       │

MySQL (predictions log)  ·  Ollama Runtime — Qwen2.5:3B (fully local, no API cost)

---

## Authentication & Role-Based Access

- Session-based Flask login — @login\_required and @admin\_required decorators guard routes

- Two roles: **customer** and **admin**, each redirected to a different home page after login

- Every prediction is logged to MySQL (predictions table) with state, city, category, and predicted days

| Role | Access |

|---|---|

| Customer | Own predictions & order history, delivery prediction tool, customer chatbot (policy + order Q&A) |

| Admin | Full revenue & courier/seller analytics, state & shipping-mode performance, admin chatbot for business questions |

---

## Project Structure


homehaven\_v3/

├── app.py                      # Flask app — routes, auth, sessions

├── agent.py                    # Customer AI agent (ReAct loop)

├── admin\_agent.py               # Admin AI agent (ReAct loop)

├── tools.py                     # Customer-facing tools (prediction, orders, RAG)

├── admin\_tools.py                # Admin-facing analytics tools

├── feature\_engineering.py       # FeatureBuilder pipeline (93 engineered features)

├── rag.py                       # RAG: chunking, embeddings, vector store

├── llm.py                       # Ollama LLM client

├── test\_model.py                 # Local model sanity check script

├── schema.sql                    # MySQL schema (predictions table)

├── catboost\_baseline\_model.pkl   # Trained CatBoost model

├── logistics\_data.csv            # Training / analytics dataset

├── requirements.txt

├── .env.example                   # Environment variable template

├── knowledge\_base/                # RAG source docs (policies, FAQs)

├── static/                        # CSS \& JS

└── templates/                     # Jinja2 HTML templates

---

## Setup & Installation

### Prerequisites

- Python 3.9+

- MySQL Server

- [Ollama](https://ollama.com) installed and running locally

1. Clone the repository

git clone https://github.com/maneprathamesh/DBDA---Machine-Learning-Project.git

cd DBDA---Machine-Learning-Project/homehaven\_v3
2. Install Python dependencies

pip install -r requirements.txt
3. Pull the required Ollama models

ollama pull qwen2.5:3b

ollama pull nomic-embed-text
4. Set up the database

mysql -u root -p < schema.sql
5. Configure environment variables

Copy .env.example to .env and fill in real values:

cp .env.example .env

SECRET\_KEY=<a random secret string>

ADMIN\_PASSWORD=<your admin password>

CUSTOMER\_PASSWORD=<your customer password>

DB\_HOST=localhost

DB\_USER=root

DB\_PASSWORD=<your MySQL password>

DB\_NAME=delivery\_prediction\_db

6. Run the app

bash

python app.py

The app runs at http://localhost:5000.

---

Results
- Best model: **CatBoost** — R² 0.8384, MAE 1.25 days, RMSE 1.57 days

- Operational features confirmed to drive real accuracy gains (R² drops to 0.782 without them)

- Working tool-calling agents for both customers and admins

- Local RAG chatbot answering policy/delivery questions — no internet dependency

- Role-based dashboard separating customer and admin capabilities

- Every prediction and chat interaction is grounded in real tool output — no hallucinated numbers

Future Scope
- Live Weather & Google Maps API in place of historical sampling

- Real-time courier GPS tracking

- Swap the manual ReAct loop for native tool-calling (qwen2.5:7b / llama3.1)

- Multi-language chatbot support

- Cloud deployment (AWS / Azure) with authentication hardening
