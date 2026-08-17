"""
tools.py
--------
Every capability of the app (ML prediction, database lookup, knowledge-base
search) lives here as a plain Python function with a clear name, docstring,
and typed-ish signature. These are the "tools" the agent chooses between.

Nothing in here talks to Flask or to the LLM - that separation is what
lets the same functions be called directly by app.py's routes AND by
agent.py's tool-calling loop.
"""

import os
import pickle

import numpy as np

from feature_engineering import FeatureBuilder, CATEGORICAL_COLUMNS
from rag import search as _rag_search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "catboost_baseline_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "Synthetic_Logistics_with_Real_Olist_Fields.csv")

DB_CONFIG = dict(
    host=os.environ.get("DB_HOST", "localhost"),
    user=os.environ.get("DB_USER", "root"),
    password=os.environ.get("DB_PASSWORD", "prerana"),
    database=os.environ.get("DB_NAME", "delivery_prediction_db"),
)

# NOTE: no more hardcoded per-state centroids. The dataset has dozens of
# distinct real cities with their own lat/lng per state - fb.resolve_
# customer_location() (feature_engineering.py) looks up an exact city if
# given, or samples a real city that actually exists in that state.

# ── Load model + feature builder once, at import time ──
print("[tools] Loading CatBoost model...")
with open(MODEL_PATH, "rb") as f:
    _model = pickle.load(f)
MODEL_FEATURE_ORDER = list(_model.feature_names_)
print(f"[tools] Model expects {len(MODEL_FEATURE_ORDER)} features.")

print("[tools] Building feature lookups from CSV...")
fb = FeatureBuilder(CSV_PATH)
print(f"[tools] Catalog: {len(fb.catalog)} product categories.")

ORDER_PRIORITIES = CATEGORICAL_COLUMNS["order_priority"]
SHIPPING_MODES = CATEGORICAL_COLUMNS["shipping_mode"]
KNOWN_CATEGORIES = fb.catalog["product_category_name_english"].tolist()
KNOWN_STATES = CATEGORICAL_COLUMNS["customer_state"]


def get_cities_for_state(customer_state):
    """
    Returns the sorted list of real city names that exist for this state
    in the dataset (fb.customer_geo) - used to populate a dropdown so the
    user always picks a city we actually have exact lat/lng for, instead
    of typing free text that may not match.
    """
    subset = fb.customer_geo[fb.customer_geo["customer_state"] == customer_state]
    return sorted(subset["customer_city"].dropna().unique().tolist())


# ═══════════════════════════════════════════════════════════════════════
# TOOL 1: ML prediction
# ═══════════════════════════════════════════════════════════════════════
def predict_delivery(category, customer_state, customer_city="",
                      shipping_mode="standard", order_priority="medium",
                      log_to_db=True):
    """
    Predicts delivery time (in days) for a product category shipped to a
    Brazilian state, using the CatBoost model.

    Args:
        category: a product_category_name_english value, e.g. "housewares"
        customer_state: 2-letter Brazilian state code, e.g. "SP", "RJ"
        customer_city: optional city name, for logging only
        shipping_mode: one of economy/standard/express/same_day
        order_priority: one of low/medium/high/urgent
        log_to_db: if True, writes the result to the predictions table

    Returns: dict with predicted_days_mean/min/max and prediction_id.
    Raises ValueError if category or state isn't recognized.
    """
    if category not in KNOWN_CATEGORIES:
        raise ValueError(
            f"Unknown category '{category}'. Known categories: "
            f"{', '.join(KNOWN_CATEGORIES[:15])}..."
        )
    if customer_state not in KNOWN_STATES:
        raise ValueError(
            f"Unknown state '{customer_state}'. Must be a Brazilian state "
            f"code, one of: {', '.join(KNOWN_STATES)}"
        )
    shipping_mode = shipping_mode if shipping_mode in SHIPPING_MODES else "standard"
    order_priority = order_priority if order_priority in ORDER_PRIORITIES else "medium"

    product = fb.get_product(category)
    lat, lng, resolved_city = fb.resolve_customer_location(customer_state, customer_city)

    feature_df = fb.build_feature_row(
        product=product,
        customer_state=customer_state,
        customer_city=resolved_city,
        customer_lat=lat,
        customer_lng=lng,
        order_priority=order_priority,
        shipping_mode=shipping_mode,
        n_samples=40,
    )
    missing = [c for c in MODEL_FEATURE_ORDER if c not in feature_df.columns]
    for c in missing:
        feature_df[c] = 0
    feature_df = feature_df[MODEL_FEATURE_ORDER]

    preds = np.clip(_model.predict(feature_df), 1, 30)

    # Raw min/max across 40 simulated scenarios (different couriers/weather/
    # seller distance/etc) is honestly labeled "Best Case"/"Worst Case" in the
    # web UI - that's fine there. But a chatbot saying "between 2.9 and 15.2
    # days" without that framing reads as an absurdly wide plain estimate.
    # So: keep min/max as-is for the UI, and ALSO expose a tighter 20th-80th
    # percentile "typical" range that the chatbot should prefer in prose.
    result = {
        "category": category,
        "customer_state": customer_state,
        "customer_city": resolved_city,
        "shipping_mode": shipping_mode,
        "order_priority": order_priority,
        "predicted_days_mean": round(float(np.mean(preds)), 1),
        "predicted_days_min": round(float(np.min(preds)), 1),
        "predicted_days_max": round(float(np.max(preds)), 1),
        "predicted_days_typical_low": round(float(np.percentile(preds, 20)), 1),
        "predicted_days_typical_high": round(float(np.percentile(preds, 80)), 1),
        "n_samples": len(preds),
    }

    result["prediction_id"] = (
        _log_prediction(customer_state, resolved_city, category,
                         result["predicted_days_mean"],
                         result["predicted_days_min"],
                         result["predicted_days_max"])
        if log_to_db else None
    )
    return result


def _log_prediction(customer_state, customer_city, category,
                     predicted_days_mean, predicted_days_min, predicted_days_max):
    try:
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO predictions
               (customer_state, customer_city, product_category,
                predicted_days_mean, predicted_days_min, predicted_days_max)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (customer_state, customer_city, category,
             predicted_days_mean, predicted_days_min, predicted_days_max),
        )
        conn.commit()
        prediction_id = cur.lastrowid
        cur.close()
        conn.close()
        return prediction_id
    except Exception as e:
        print(f"[DB logging skipped] {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# TOOL 2: database lookup
# ═══════════════════════════════════════════════════════════════════════
def get_predictions(limit=1, customer_state=None, prediction_id=None):
    """
    Reads past predictions from MySQL.

    Args:
        limit: how many rows to return (most recent first). Ignored if
               prediction_id is given.
        customer_state: optional 2-letter state filter, e.g. "SP"
        prediction_id: if given, fetch that exact prediction by id instead

    Returns: list of dicts (empty list if none found / DB unavailable).
    """
    try:
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)

        if prediction_id is not None:
            cur.execute("SELECT * FROM predictions WHERE prediction_id = %s",
                        (prediction_id,))
            rows = cur.fetchall()
        elif customer_state:
            cur.execute(
                """SELECT * FROM predictions WHERE customer_state = %s
                   ORDER BY prediction_id DESC LIMIT %s""",
                (customer_state, int(limit)),
            )
            rows = cur.fetchall()
        else:
            cur.execute(
                "SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT %s",
                (int(limit),),
            )
            rows = cur.fetchall()

        cur.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB lookup failed] {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# TOOL 2a: fetch one specific order by ID (customer-facing)
# ═══════════════════════════════════════════════════════════════════════
def get_order_by_id(order_id):
    """
    Fetches a single customer order from the predictions table by its ID.
    Returns a customer-safe dict — no courier, seller, or warehouse details.
    """
    try:
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor(dictionary=True)
        cur.execute(
            """SELECT prediction_id, customer_state, customer_city,
                      product_category, predicted_days_mean,
                      predicted_days_min, predicted_days_max,
                      prediction_time
               FROM predictions
               WHERE prediction_id = %s""",
            (int(order_id),)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return {
                "found": False,
                "message": f"No order found with ID #{order_id}. "
                           f"Please check the order number and try again."
            }
        return {
            "found":                True,
            "order_id":             row["prediction_id"],
            "product_category":     row["product_category"],
            "customer_state":       row["customer_state"],
            "customer_city":        row["customer_city"],
            "predicted_days_mean":  row["predicted_days_mean"],
            "predicted_days_min":   row["predicted_days_min"],
            "predicted_days_max":   row["predicted_days_max"],
            "order_placed_at":      str(row["prediction_time"]),
        }
    except Exception as e:
        print(f"[DB order lookup failed] {e}")
        return {"found": False, "message": f"Could not reach the database: {e}"}


# ═══════════════════════════════════════════════════════════════════════
# TOOL 2b: fetch recent orders (customer-facing)
# ═══════════════════════════════════════════════════════════════════════
def get_my_recent_orders(limit=3):
    """
    Returns the most recent N predictions from the DB.
    Customer-safe — no courier/seller/warehouse details.
    """
    try:
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor(dictionary=True)
        cur.execute(
            """SELECT prediction_id, customer_state, customer_city,
                      product_category, predicted_days_mean, prediction_time
               FROM predictions
               ORDER BY prediction_id DESC
               LIMIT %s""",
            (int(limit),)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if not rows:
            return {"found": False, "message": "No orders found in the database yet."}
        return {
            "found":  True,
            "count":  len(rows),
            "orders": [
                {
                    "order_id":            r["prediction_id"],
                    "product_category":    r["product_category"],
                    "customer_state":      r["customer_state"],
                    "customer_city":       r["customer_city"],
                    "predicted_days_mean": r["predicted_days_mean"],
                    "order_placed_at":     str(r["prediction_time"]),
                }
                for r in rows
            ]
        }
    except Exception as e:
        print(f"[DB recent orders failed] {e}")
        return {"found": False, "message": f"Could not reach the database: {e}"}


# ═══════════════════════════════════════════════════════════════════════
# TOOL 3: knowledge-base / RAG search
# ═══════════════════════════════════════════════════════════════════════
def search_knowledge(query, top_k=3):
    """
    Searches the local knowledge_base/ (via vector_store.pkl) for text
    relevant to `query` - use this for "why"/"how"/general logistics
    questions that aren't about a specific order.

    Returns: a string of the most relevant chunks (empty string if the
    vector index hasn't been built yet or the Gemini embedding call fails).
    """
    try:
        return _rag_search(query, top_k=top_k)
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"[RAG search failed] {e}")
        return ""
