"""
admin_tools.py
--------------
Admin-only tools: revenue stats, top categories, state breakdown,
recent predictions summary. These are NEVER exposed to the customer agent.
"""
import os
from tools import DB_CONFIG
import pandas as pd

# Load CSV once for offline stats (no DB needed for CSV-based metrics)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Synthetic_Logistics_with_Real_Olist_Fields.csv")
_df = pd.read_csv(CSV_PATH)


def get_revenue_summary():
    """Total revenue, avg order value, freight stats from dataset."""
    return {
        "total_revenue":        round(float(_df["payment_value"].sum()), 2),
        "avg_order_value":      round(float(_df["payment_value"].mean()), 2),
        "total_freight_cost":   round(float(_df["freight_value"].sum()), 2),
        "avg_freight_cost":     round(float(_df["freight_value"].mean()), 2),
        "total_orders":         int(len(_df)),
        "freight_revenue_pct":  round(float(_df["freight_value"].sum() / _df["payment_value"].sum() * 100), 1),
    }


def get_top_categories(limit=10):
    """Top product categories by revenue and order volume."""
    grp = _df.groupby("product_category_name_english").agg(
        total_revenue=("payment_value", "sum"),
        total_orders=("payment_value", "count"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_freight=("freight_value", "mean"),
    ).round(2).sort_values("total_revenue", ascending=False).head(limit)
    return grp.reset_index().to_dict(orient="records")


def get_state_breakdown():
    """Revenue and delivery performance per Brazilian state."""
    grp = _df.groupby("customer_state").agg(
        total_orders=("payment_value", "count"),
        total_revenue=("payment_value", "sum"),
        avg_delivery_days=("delivery_days", "mean"),
    ).round(2).sort_values("total_revenue", ascending=False)
    return grp.reset_index().to_dict(orient="records")


def get_shipping_mode_stats():
    """Revenue and avg delivery days split by shipping mode."""
    grp = _df.groupby("shipping_mode").agg(
        total_orders=("payment_value", "count"),
        total_revenue=("payment_value", "sum"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_freight=("freight_value", "mean"),
    ).round(2)
    return grp.reset_index().to_dict(orient="records")


def get_recent_db_predictions(limit=10):
    """Last N predictions stored in MySQL by any user."""
    try:
        import mysql.connector
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT %s",
            (int(limit),)
        )
        rows = cur.fetchall()
        cur.close(); conn.close()
        return rows
    except Exception as e:
        return {"error": str(e)}


def get_fraud_risk_states():
    """States with highest avg delivery days — potential logistics issues."""
    grp = _df.groupby("customer_state")["delivery_days"].mean().round(2)
    top = grp.sort_values(ascending=False).head(5)
    return [{"state": k, "avg_delivery_days": v} for k, v in top.items()]


def get_monthly_revenue_trend():
    """Revenue and order volume by order_month, for the dashboard growth chart."""
    grp = _df.groupby("order_month").agg(
        total_revenue=("payment_value", "sum"),
        total_orders=("payment_value", "count"),
    ).round(2).sort_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return [
        {
            "month": month_names[int(m) - 1],
            "total_revenue": row["total_revenue"],
            "total_orders": int(row["total_orders"]),
        }
        for m, row in grp.iterrows()
    ]


def _run_query(sql, params=()):
    """Runs a read query against the live orders table in MySQL."""
    import mysql.connector
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_top_couriers(limit=3):
    """Top courier partners by order volume, with avg delivery performance — live from MySQL orders."""
    try:
        return _run_query(
            """SELECT courier_partner,
                      COUNT(*) AS total_orders,
                      ROUND(AVG(delivery_days), 2) AS avg_delivery_days,
                      ROUND(SUM(payment_value), 2) AS total_revenue,
                      SUM(courier_vehicle_breakdown_flag) AS breakdown_count
               FROM orders
               GROUP BY courier_partner
               ORDER BY total_orders DESC
               LIMIT %s""",
            (int(limit),),
        )
    except Exception as e:
        return {"error": str(e)}


def get_top_cities(limit=10):
    """Top customer cities by order volume — live from MySQL orders."""
    try:
        return _run_query(
            """SELECT customer_city, customer_state,
                      COUNT(*) AS total_orders,
                      ROUND(SUM(payment_value), 2) AS total_revenue
               FROM orders
               GROUP BY customer_city, customer_state
               ORDER BY total_orders DESC
               LIMIT %s""",
            (int(limit),),
        )
    except Exception as e:
        return {"error": str(e)}


def get_top_sellers(limit=10):
    """
    Top sellers by revenue — live from MySQL orders.
    Note: this dataset has no individual seller_id, so seller_city + seller_state
    is used as the seller grouping key (the finest seller-level detail available).
    """
    try:
        return _run_query(
            """SELECT seller_city, seller_state,
                      COUNT(*) AS total_orders,
                      ROUND(SUM(payment_value), 2) AS total_revenue,
                      ROUND(AVG(seller_avg_dispatch_time), 2) AS avg_dispatch_time
               FROM orders
               GROUP BY seller_city, seller_state
               ORDER BY total_revenue DESC
               LIMIT %s""",
            (int(limit),),
        )
    except Exception as e:
        return {"error": str(e)}
