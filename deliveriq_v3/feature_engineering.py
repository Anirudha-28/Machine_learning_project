"""
feature_engineering.py  — FIXED VERSION
========================================
Bugs fixed:
1. shipping_mode now taken from USER input — not randomly sampled
2. express_delivery derived from USER's shipping_mode — not sampled
3. estimated_delivery_days COMPUTED from distance + shipping_mode formula
   (was randomly sampled before — it's a 0.69-corr proxy for the target,
    so random sampling was the #1 reason model always predicted 6+ days)
4. build_feature_row() accepts shipping_mode as explicit parameter
"""

import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Categorical columns — alphabetically sorted (drop_first removes index 0)
# ---------------------------------------------------------------------------
CATEGORICAL_COLUMNS = {
    "order_priority":       ["high", "low", "medium", "urgent"],
    "shipping_mode":        ["economy", "express", "same_day", "standard"],
    "weather_condition":    ["clear", "heavy_rain", "light_rain", "storm"],
    "road_condition":       ["average", "damaged", "excellent", "good", "poor"],
    "warehouse_capacity":   ["High", "Low", "Medium"],
    "warehouse_location":   ["WH_Central", "WH_East", "WH_North", "WH_Premium", "WH_South", "WH_West"],
    "courier_vehicle_type": ["bike", "large_truck", "motorcycle", "small_van"],
    "courier_partner":      ["EcoDeliver", "FastShip", "LogiPrime", "QuickMove", "ReliExpress", "RuralPost"],
    "seller_state":         None,   # filled at load time
    "customer_state":       None,   # filled at load time
}

BRAZIL_HOLIDAYS_MONTH_DAY = {
    (1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 25),
}

# Calibrated from real data: estimated_delivery_days = slope*dist_km + intercept
_EST_DELIVERY_FORMULA = {
    "economy":  (0.002466, 2.0024),
    "express":  (0.002460, 1.9640),
    "same_day": (0.002543, 1.9806),
    "standard": (0.002521, 1.9100),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class FeatureBuilder:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # Target encoding maps
        self.cat_name_enc = self.df.groupby("product_category_name")["delivery_days"].mean().to_dict()
        self.cat_name_eng_enc = self.df.groupby("product_category_name_english")["delivery_days"].mean().to_dict()
        self.global_mean_days = self.df["delivery_days"].mean()

        # Fill real state lists
        CATEGORICAL_COLUMNS["seller_state"]   = sorted(self.df["seller_state"].dropna().unique().tolist())
        CATEGORICAL_COLUMNS["customer_state"] = sorted(self.df["customer_state"].dropna().unique().tolist())

        # Product catalog
        self.catalog = (
            self.df.sort_values("order_id")
            .groupby("product_category_name_english", as_index=False)
            .first()
        )

        # Operational columns still sampled from history (genuine variability)
        # NOTE: shipping_mode and estimated_delivery_days REMOVED from here
        self._op_cols = [
            "courier_partner", "courier_vehicle_type", "courier_vehicle_breakdown_flag",
            "warehouse_location", "warehouse_capacity", "warehouse_processing_hours",
            "traffic_index", "road_condition", "rainfall_mm", "weather_condition",
            "fuel_price_index", "seller_avg_dispatch_time", "estimated_dispatch_time",
            "rural_route_flag",
        ]

        # ── Real city-level geography (FIX 4) ──
        # The dataset has a distinct city + lat/lng for every customer_state
        # and seller_state (dozens of cities per state, e.g. city_BA_0,
        # city_BA_4, ... each with its own coordinates - not one point per
        # state). Using a single hardcoded state centroid, or always the
        # same "first row" seller for a whole category, throws that away
        # and makes distance (and therefore the prediction) far less
        # realistic. Instead:
        #   - customer_geo: every real (state, city) -> (lat, lng) pair,
        #     so we can look up an exact city or sample a real one.
        #   - seller_pool: every real seller (state, city, lat, lng) that
        #     has actually sold in a given category, so distance and
        #     seller_state vary sample-to-sample the same way courier/
        #     weather/road already do, instead of being frozen to one
        #     seller for the whole category.
        self.customer_geo = (
            self.df[["customer_state", "customer_city", "customer_lat", "customer_lng"]]
            .dropna()
            .drop_duplicates(subset=["customer_state", "customer_city"])
            .reset_index(drop=True)
        )
        self.seller_pool_by_category = {
            cat: g[["seller_state", "seller_city", "seller_lat", "seller_lng"]]
            .dropna()
            .reset_index(drop=True)
            for cat, g in self.df.groupby("product_category_name_english")
        }

    def resolve_customer_location(self, customer_state, customer_city=None, rng=None):
        """
        Looks up real coordinates for a customer.

        If customer_city is given and matches a real city in that state
        (case-insensitive), use its exact lat/lng. Otherwise, sample a
        real city that actually exists in that state from the dataset
        (never a single fixed state centroid) and return its name too, so
        callers can show/log which city was used.

        Returns (lat, lng, resolved_city_name). Falls back to Brasília's
        coordinates only if the state has no rows at all.
        """
        subset = self.customer_geo[self.customer_geo["customer_state"] == customer_state]
        if subset.empty:
            return -15.83, -47.86, customer_city or ""

        if customer_city:
            match = subset[subset["customer_city"].str.lower() == customer_city.strip().lower()]
            if not match.empty:
                row = match.iloc[0]
                return float(row["customer_lat"]), float(row["customer_lng"]), row["customer_city"]

        rng = rng or np.random.default_rng()
        row = subset.iloc[int(rng.integers(0, len(subset)))]
        return float(row["customer_lat"]), float(row["customer_lng"]), row["customer_city"]

    def get_product(self, category_slug):
        row = self.catalog[self.catalog["product_category_name_english"] == category_slug]
        if row.empty:
            row = self.catalog.iloc[[0]]
        return row.iloc[0].to_dict()

    def list_products(self):
        return self.catalog[[
            "product_category_name_english", "product_category_name",
            "product_weight_g", "freight_value", "payment_value",
            "seller_state", "seller_city",
        ]].to_dict(orient="records")

    def _one_hot(self, col, value, row):
        for cat in CATEGORICAL_COLUMNS[col]:
            if cat == CATEGORICAL_COLUMNS[col][0]:
                continue  # dropped reference category
            row[f"{col}_{cat}"] = 1 if value == cat else 0

    def build_feature_row(
        self,
        product,
        customer_state,
        customer_city,
        customer_lat,
        customer_lng,
        order_priority,
        shipping_mode,          # ← NOW explicitly from the user
        n_samples=40,
        random_state=None,
    ):
        rng = np.random.default_rng(random_state)
        sample_idx  = rng.integers(0, len(self.df), size=n_samples)
        op_samples  = self.df.iloc[sample_idx][self._op_cols].reset_index(drop=True)

        # FIX 4: sample a REAL seller (state/city/lat/lng) per row from
        # everyone who has actually sold this category, instead of using
        # one fixed "first row" seller for every sample. Distance and
        # same_state/seller_state now vary sample-to-sample too, which is
        # more honest about real seller variety within a category.
        seller_pool = self.seller_pool_by_category.get(product["product_category_name_english"])
        if seller_pool is None or seller_pool.empty:
            seller_pool = pd.DataFrame([{
                "seller_state": product["seller_state"],
                "seller_city": product.get("seller_city", ""),
                "seller_lat": product["seller_lat"],
                "seller_lng": product["seller_lng"],
            }])
        seller_idx = rng.integers(0, len(seller_pool), size=n_samples)
        seller_samples = seller_pool.iloc[seller_idx].reset_index(drop=True)

        now  = pd.Timestamp.now()
        rows = []

        # FIX 3 formula needs a representative distance for the est.
        # delivery proxy; use the median across this request's sampled
        # sellers rather than a single arbitrary one.
        _dists_for_formula = [
            haversine_km(customer_lat, customer_lng, float(r.seller_lat), float(r.seller_lng))
            for r in seller_samples.itertuples()
        ]
        slope, intercept = _EST_DELIVERY_FORMULA.get(shipping_mode, (0.002521, 1.9100))
        computed_est_delivery = slope * float(np.median(_dists_for_formula)) + intercept

        for i in range(n_samples):
            op  = op_samples.iloc[i]
            sel = seller_samples.iloc[i]
            row = {}

            # Calendar features
            row["order_month"]          = now.month
            row["order_weekday"]        = now.weekday()
            row["order_hour"]           = now.hour
            row["purchase_dayofweek"]   = now.weekday()
            row["is_weekend_purchase"]  = int(now.weekday() >= 5)
            row["is_year_end_peak"]     = int(now.month in (11, 12))
            row["holiday_flag"]         = int((now.month, now.day) in BRAZIL_HOLIDAYS_MONTH_DAY)
            row["festival_flag"]        = int(now.month in (6, 12))

            # Product fields
            weight_g      = float(product["product_weight_g"])
            length_cm     = float(product.get("product_length_cm", 20))
            width_cm      = float(product.get("product_width_cm",  15))
            height_cm     = float(product.get("product_height_cm", 10))
            volume_cm3    = length_cm * width_cm * height_cm
            freight_value = float(product["freight_value"])
            payment_value = float(product["payment_value"])

            row["product_weight_g_log"]    = math.log1p(weight_g)
            row["product_volume_cm3_log"]  = math.log1p(volume_cm3)
            row["freight_value_log"]       = math.log1p(freight_value)
            row["payment_value_log"]       = math.log1p(payment_value)
            row["freight_payment_ratio"]   = freight_value / payment_value if payment_value else 0

            row["product_category_name_target_enc"] = self.cat_name_enc.get(
                product["product_category_name"], self.global_mean_days)
            row["product_category_name_english_target_enc"] = self.cat_name_eng_enc.get(
                product["product_category_name_english"], self.global_mean_days)

            # Geography — real seller location for THIS sample (FIX 4),
            # not a single fixed seller frozen for the whole category
            seller_lat_i   = float(sel["seller_lat"])
            seller_lng_i   = float(sel["seller_lng"])
            seller_state_i = sel["seller_state"]
            dist_km_i      = haversine_km(customer_lat, customer_lng, seller_lat_i, seller_lng_i)

            row["distance_km_log"] = math.log1p(dist_km_i)
            row["same_state"]      = int(customer_state == seller_state_i)
            self._one_hot("seller_state",   seller_state_i, row)
            self._one_hot("customer_state", customer_state, row)

            # FIX 1: use USER's order_priority
            self._one_hot("order_priority", order_priority, row)

            # FIX 2: use USER's shipping_mode — not sampled
            self._one_hot("shipping_mode", shipping_mode, row)
            row["express_delivery"] = 1 if shipping_mode == "express" else 0

            # FIX 3: use computed estimated_delivery_days — not sampled
            row["estimated_delivery_days"] = computed_est_delivery

            # Operational fields — still sampled (genuine day-to-day variability)
            row["rural_route_flag"]                = int(op["rural_route_flag"])
            row["courier_vehicle_breakdown_flag"]  = int(op["courier_vehicle_breakdown_flag"])
            row["seller_avg_dispatch_time"]        = float(op["seller_avg_dispatch_time"])
            row["warehouse_processing_hours"]      = float(op["warehouse_processing_hours"])
            row["traffic_index"]                   = float(op["traffic_index"])
            row["rainfall_mm"]                     = float(op["rainfall_mm"])
            row["fuel_price_index"]                = float(op["fuel_price_index"])
            row["estimated_dispatch_time"]         = float(op["estimated_dispatch_time"])

            self._one_hot("weather_condition",    op["weather_condition"],    row)
            self._one_hot("road_condition",       op["road_condition"],       row)
            self._one_hot("warehouse_capacity",   op["warehouse_capacity"],   row)
            self._one_hot("warehouse_location",   op["warehouse_location"],   row)
            self._one_hot("courier_vehicle_type", op["courier_vehicle_type"], row)
            self._one_hot("courier_partner",      op["courier_partner"],      row)

            rows.append(row)

        return pd.DataFrame(rows)
