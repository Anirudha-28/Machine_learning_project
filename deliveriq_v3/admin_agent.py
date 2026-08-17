"""
admin_agent.py
--------------
Admin-only agent with access to business intelligence tools.
Customers never hit this — it's behind @admin_required in app.py.

Same Gemini native-function-calling approach as agent.py — see that file's
docstring for why this replaced the manual JSON ReAct loop.
"""
import json

import admin_tools
from llm import generate

MAX_STEPS = 3

TOOL_SPECS = {
    "get_revenue_summary": {
        "fn": admin_tools.get_revenue_summary,
        "description": "Returns total revenue, avg order value, total freight cost, total orders from the full dataset. Use for any revenue/financial question.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "get_top_categories": {
        "fn": admin_tools.get_top_categories,
        "description": "Returns top product categories ranked by revenue with order count and avg delivery days.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Number of categories to return (default 10)."}},
            "required": [],
        },
    },
    "get_state_breakdown": {
        "fn": admin_tools.get_state_breakdown,
        "description": "Returns revenue and delivery stats per Brazilian state. Use for geographic/regional questions.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "get_shipping_mode_stats": {
        "fn": admin_tools.get_shipping_mode_stats,
        "description": "Returns revenue, order count, avg delivery days split by shipping mode (economy/standard/express/same_day).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "get_recent_db_predictions": {
        "fn": admin_tools.get_recent_db_predictions,
        "description": "Fetches the most recent prediction logs from MySQL. Use when admin asks about recent user activity or prediction history.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "How many rows (default 10)."}},
            "required": [],
        },
    },
    "get_fraud_risk_states": {
        "fn": admin_tools.get_fraud_risk_states,
        "description": "Returns states with highest avg delivery days — potential logistics bottlenecks or risk areas.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

SYSTEM_PROMPT = (
    "You are the DeliverIQ Admin Intelligence Assistant. You have access to "
    "business data tools that give you real numbers from the dataset and database. "
    "You assist the admin (business owner/manager) with revenue, logistics performance, "
    "top products, state performance, and user activity insights. Always call a tool "
    "to get real numbers — never invent statistics.\n\n"
    "Final answer style: professional but friendly, use real numbers from tool output, "
    "2-5 sentences, no bullet points, no markdown. Lead with the key insight."
)

FUNCTION_DECLARATIONS = [
    {"name": name, "description": spec["description"], "parameters": spec["parameters"]}
    for name, spec in TOOL_SPECS.items()
]


def _run_tool(action, action_input):
    spec = TOOL_SPECS.get(action)
    if not spec:
        return json.dumps({"error": f"no such tool '{action}'"})
    try:
        return json.dumps(spec["fn"](**(action_input or {})), default=str)
    except Exception as e:
        return json.dumps({"error": f"{action}: {e}"})


def run_admin_agent(user_message, history=None, max_steps=MAX_STEPS):
    contents = [
        {"role": ("model" if h.get("role") == "assistant" else "user"), "parts": [{"text": h["content"]}]}
        for h in (history or [])
    ]
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    trace = []
    system_instruction = {"parts": [{"text": SYSTEM_PROMPT}]}

    for _step in range(max_steps):
        content = generate(
            contents,
            system_instruction=system_instruction,
            tools=FUNCTION_DECLARATIONS,
            temperature=0.1,
        )
        parts = content.get("parts", [])
        fn_calls = [p["functionCall"] for p in parts if "functionCall" in p]

        if not fn_calls:
            reply = "".join(p.get("text", "") for p in parts).strip()
            reply = reply or "Sorry, I couldn't come up with an answer for that."
            trace.append({"final_answer": reply})
            return {"reply": reply, "trace": trace}

        contents.append({"role": "model", "parts": parts})
        response_parts = []
        for call in fn_calls:
            name = call.get("name")
            args = call.get("args") or {}
            observation = _run_tool(name, args)
            trace.append({"action": name, "action_input": args, "observation": observation})
            response_parts.append({"functionResponse": {"name": name, "response": {"result": observation}}})
        contents.append({"role": "function", "parts": response_parts})

    contents.append({
        "role": "user",
        "parts": [{"text": "Please give your final answer now, in plain text, based on the tool results above."}],
    })
    content = generate(contents, system_instruction=system_instruction, temperature=0.1)
    reply = "".join(p.get("text", "") for p in content.get("parts", [])).strip()
    reply = reply or "Sorry, I'm having trouble finishing that request."
    trace.append({"forced_final_answer": reply})
    return {"reply": reply, "trace": trace}
