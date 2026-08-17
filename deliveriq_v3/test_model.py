"""
test_model.py
Run this locally (after `pip install -r requirements.txt`) to confirm the
model gives varied, realistic predictions across different products/states
BEFORE you rely on the Flask app or chatbot. Calls the exact same
tools.predict_delivery() function app.py and agent.py use, so a pass here
means the whole pipeline is wired correctly.
"""
import random

from tools import predict_delivery, KNOWN_CATEGORIES, KNOWN_STATES

random.seed(1)
SHIPPING_MODES = ["economy", "standard", "express", "same_day"]
PRIORITIES = ["low", "medium", "high", "urgent"]

categories = random.sample(KNOWN_CATEGORIES, min(10, len(KNOWN_CATEGORIES)))

print(f"{'Category':<30} {'State':<5} {'Ship':<9} {'Mean':>6} {'Min':>6} {'Max':>6}")
print("-" * 68)

all_means = []
for cat in categories:
    state = random.choice(KNOWN_STATES)
    shipping_mode = random.choice(SHIPPING_MODES)
    priority = random.choice(PRIORITIES)

    result = predict_delivery(
        category=cat,
        customer_state=state,
        shipping_mode=shipping_mode,
        order_priority=priority,
        log_to_db=False,  # skip DB writes for this sanity check
    )
    all_means.append(result["predicted_days_mean"])
    print(f"{cat[:30]:<30} {state:<5} {shipping_mode:<9} "
          f"{result['predicted_days_mean']:>6.1f} "
          f"{result['predicted_days_min']:>6.1f} "
          f"{result['predicted_days_max']:>6.1f}")

print("-" * 68)
spread = max(all_means) - min(all_means)
print(f"Min mean: {min(all_means):.1f}  Max mean: {max(all_means):.1f}  Spread: {spread:.1f} days")
if spread < 2:
    print("\n⚠️  Predictions are still clustered - something in the pipeline is still off.")
else:
    print("\n✅ Predictions show realistic variation across products/states/shipping modes.")
