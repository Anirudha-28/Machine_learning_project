import os
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import tools
from agent import run_agent
from admin_agent import run_admin_agent
from admin_tools import (
    get_revenue_summary, get_top_categories,
    get_top_couriers, get_top_cities, get_top_sellers,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "deliveriq-secret-2025")

# ── Simple user store (replace with DB later) ──
USERS = {
    "admin":    {"password": "admin123",    "role": "admin"},
    "customer": {"password": "customer123", "role": "customer"},
}

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = USERS.get(username)
        if user and user["password"] == password:
            session["user"] = username
            session["role"] = user["role"]
            if user["role"] == "admin":
                return redirect(url_for("admin_chatbot"))
            return redirect(url_for("home"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ════════════════════════════════════
# MAIN ROUTES
# ════════════════════════════════════
@app.route("/")
@login_required
def home():
    products = tools.fb.list_products()
    return render_template("index.html", products=products, role=session.get("role"))

@app.route("/product/<category_slug>")
@login_required
def product_detail(category_slug):
    product = tools.fb.get_product(category_slug)
    return render_template(
        "product.html",
        product=product,
        states=tools.KNOWN_STATES,
        priorities=tools.ORDER_PRIORITIES,
        shipping_modes=tools.SHIPPING_MODES,
        selected_city="",
        role=session.get("role"),
    )

@app.route("/cities/<customer_state>")
@login_required
def cities(customer_state):
    return jsonify(tools.get_cities_for_state(customer_state))

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json() if request.is_json else request.form
    category_slug  = data["category"]
    customer_state = data["customer_state"]
    customer_city  = data.get("customer_city", "")
    order_priority = data.get("order_priority", "medium")
    shipping_mode  = data.get("shipping_mode", "standard")

    result = tools.predict_delivery(
        category=category_slug,
        customer_state=customer_state,
        customer_city=customer_city,
        shipping_mode=shipping_mode,
        order_priority=order_priority,
    )

    if request.is_json:
        return jsonify(result)

    product = tools.fb.get_product(category_slug)
    return render_template(
        "product.html",
        product=product,
        states=tools.KNOWN_STATES,
        priorities=tools.ORDER_PRIORITIES,
        shipping_modes=tools.SHIPPING_MODES,
        result=result,
        selected_state=customer_state,
        selected_city=result.get("customer_city", customer_city),
        selected_priority=order_priority,
        selected_shipping=shipping_mode,
        role=session.get("role"),
    )

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", role=session.get("role"))

# ════════════════════════════════════
# CUSTOMER CHATBOT
# ════════════════════════════════════
@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html", role=session.get("role"))

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    payload  = request.get_json() or {}
    message  = payload.get("message", "")
    history  = (payload.get("history") or [])[-8:]
    try:
        result = run_agent(message, history=history)
        reply  = result["reply"]
    except RuntimeError as e:
        reply  = str(e)
        result = {"reply": reply, "trace": []}
    except Exception as e:
        import traceback; traceback.print_exc()
        reply  = "Sorry, I'm having trouble reaching the assistant right now."
        result = {"reply": reply, "trace": []}
    return jsonify({"reply": reply, "trace": result.get("trace", [])})

# ════════════════════════════════════
# ADMIN CHATBOT
# ════════════════════════════════════
@app.route("/admin/chatbot")
@admin_required
def admin_chatbot():
    return render_template("admin_chatbot.html", role="admin")


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    revenue = get_revenue_summary()
    top_categories = get_top_categories(5)

    top_couriers = get_top_couriers(3)
    top_cities = get_top_cities(10)
    top_sellers = get_top_sellers(10)

    db_error = None
    for result in (top_couriers, top_cities, top_sellers):
        if isinstance(result, dict) and "error" in result:
            db_error = result["error"]

    if db_error:
        top_couriers, top_cities, top_sellers = [], [], []

    return render_template(
        "admin_dashboard.html",
        revenue=revenue,
        top_categories=top_categories,
        top_couriers=top_couriers,
        top_cities=top_cities,
        top_sellers=top_sellers,
        db_error=db_error,
    )

@app.route("/admin/chat", methods=["POST"])
@admin_required
def admin_chat():
    payload  = request.get_json() or {}
    message  = payload.get("message", "")
    history  = (payload.get("history") or [])[-8:]
    try:
        result = run_admin_agent(message, history=history)
        reply  = result["reply"]
    except RuntimeError as e:
        reply  = str(e)
        result = {"reply": reply, "trace": []}
    except Exception as e:
        import traceback; traceback.print_exc()
        reply  = "Sorry, admin assistant error."
        result = {"reply": reply, "trace": []}
    return jsonify({"reply": reply, "trace": result.get("trace", [])})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
