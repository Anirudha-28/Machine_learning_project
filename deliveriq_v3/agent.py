"""
agent.py
--------
A real tool-calling agent: the LLM sees the user's message plus a list of
available tools, decides for itself which tool (if any) to call, sees the
tool's real output, and either calls another tool or gives a final answer.

This version uses Gemini's NATIVE function calling instead of a manual
"respond with a JSON object" ReAct loop. Two speed wins over the old
Ollama + hand-rolled-JSON version:

1. No more regex-parsing the model's raw text for a JSON blob, and no more
   burning an entire extra round trip whenever a small local model wrote
   slightly-invalid JSON ("respond with ONLY the JSON object" retries).
   Gemini returns a structured functionCall part directly.
2. Gemini's hosted flash-lite model responds in a fraction of the time a
   local 3B model does on a laptop CPU/GPU, and there's no cold-start /
   "model still loading into RAM" delay on the first request.
"""

import json

import tools
from llm import generate

MAX_STEPS = 3  # tool-call round trips before we force a final answer

# ── Tool registry: name -> callable + OpenAPI-style parameter schema ──
TOOL_SPECS = {
    "predict_delivery": {
        "fn": tools.predict_delivery,
        "description": (
            "Predicts delivery time for a product category shipped to a "
            "Brazilian customer state. Use this whenever the user asks to "
            "predict/estimate delivery time for a product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Product category (English name from the catalog).",
                    "enum": tools.KNOWN_CATEGORIES,
                },
                "customer_state": {
                    "type": "string",
                    "description": "2-letter Brazilian state code.",
                    "enum": tools.KNOWN_STATES,
                },
                "customer_city": {
                    "type": "string",
                    "description": "Optional city name.",
                },
                "shipping_mode": {
                    "type": "string",
                    "enum": tools.SHIPPING_MODES,
                    "description": "Defaults to 'standard' if not given.",
                },
                "order_priority": {
                    "type": "string",
                    "enum": tools.ORDER_PRIORITIES,
                    "description": "Defaults to 'medium' if not given.",
                },
            },
            "required": ["category", "customer_state"],
        },
    },
    "get_order_by_id": {
        "fn": tools.get_order_by_id,
        "description": (
            "Fetches a specific customer order from the database by its order ID. "
            "ALWAYS use this when the user mentions an order number or reference like "
            "'order #12', 'reference #5', 'my order 12', 'order number 7', etc. "
            "Extract the number from their message and pass it as order_id. "
            "Never say you cannot look up an order — always call this tool first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer", "description": "The order/prediction ID number."},
            },
            "required": ["order_id"],
        },
    },
    "get_my_recent_orders": {
        "fn": tools.get_my_recent_orders,
        "description": (
            "Fetches the most recent orders from the database. Use when the user "
            "asks about 'my last order', 'my recent orders', 'my previous predictions', "
            "or 'show my orders' — NOT when they give a specific order number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "How many recent orders to return (default 3)."},
            },
            "required": [],
        },
    },
    "search_knowledge": {
        "fn": tools.search_knowledge,
        "description": (
            "Searches the logistics knowledge base for explanations - WHY "
            "deliveries take a certain time, how shipping modes/weather/"
            "warehouse factors affect delivery, general policy questions. "
            "Not for specific order lookups or new predictions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or topic to search for."},
            },
            "required": ["query"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are the customer assistant for DeliverIQ, a Brazilian delivery-logistics app. "
    "You have access to tools and MUST call them to get real data — NEVER invent, "
    "guess, or fabricate any order details, delivery times, or policies.\n\n"

    "STRICT RULES — violating these is a critical failure:\n"
    "1. ORDER LOOKUP: If the user mentions ANY order number, reference number, or "
    "   order ID (e.g. '#12', 'order 12', 'reference #5', 'my order number 7'), "
    "   you MUST call get_order_by_id with that number immediately. "
    "   Extract just the integer from their message. Never say you cannot look it up.\n"
    "2. NO HALLUCINATION: Never invent order details, delivery times, policies, "
    "   or facts. Only report what a tool actually returned.\n"
    "3. POLICY QUESTIONS: For questions about express policy, shipping rules, "
    "   or delivery explanations, ALWAYS call search_knowledge first. "
    "   Do not answer policy questions from memory.\n"
    "4. WHAT CUSTOMERS CAN SEE: Customers can see their own order details "
    "   (product category, state, city, predicted days, order time). "
    "   They CANNOT see revenue, seller info, courier details, warehouse details, "
    "   or other customers' orders. If asked for admin-level info, politely decline.\n\n"

    "All state codes (SP, RJ, MG, BA, etc.) are Brazilian states.\n\n"
    "Final answer style: warm and friendly, 2-5 sentences, no bullet points, no markdown. "
    "When showing an order: include order ID, product category, customer location, "
    "predicted delivery days, and when it was made. "
    "When showing a delivery prediction: lead with predicted_days_mean and the "
    "typical low/high range. Never show min/max unless asked specifically."
)

FUNCTION_DECLARATIONS = [
    {"name": name, "description": spec["description"], "parameters": spec["parameters"]}
    for name, spec in TOOL_SPECS.items()
]


def _run_tool(action: str, action_input: dict) -> str:
    spec = TOOL_SPECS.get(action)
    if not spec:
        return json.dumps({"error": f"no such tool '{action}'", "available_tools": list(TOOL_SPECS.keys())})
    try:
        result = spec["fn"](**(action_input or {}))
        return json.dumps(result, default=str)
    except TypeError as e:
        return json.dumps({"error": f"bad arguments for {action}: {e}"})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"unexpected failure calling {action}: {e}"})


def run_agent(user_message: str, history: list = None, max_steps: int = MAX_STEPS) -> dict:
    """
    Runs the plan -> act -> observe loop (via Gemini's native function
    calling) until the model produces a plain-text final answer or
    max_steps is reached.

    `history` is the prior conversation as a list of clean
    {"role": "user"|"assistant", "content": "..."} turns (not the raw
    tool-call scratchpad from earlier turns).

    Returns: {"reply": str, "trace": [ {action, action_input, observation}, ... ]}
    """
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

        # Echo the model's own turn (its functionCall part(s)) back into the
        # conversation, then answer with the tool result(s) — this is the
        # exact shape Gemini's REST function-calling API expects.
        contents.append({"role": "model", "parts": parts})

        response_parts = []
        for call in fn_calls:
            name = call.get("name")
            args = call.get("args") or {}
            observation = _run_tool(name, args)
            trace.append({"action": name, "action_input": args, "observation": observation})
            response_parts.append({
                "functionResponse": {"name": name, "response": {"result": observation}},
            })
        contents.append({"role": "function", "parts": response_parts})

    # Ran out of steps — ask once more, but without tools, so Gemini is
    # forced to answer in plain text from whatever observations we've got.
    contents.append({
        "role": "user",
        "parts": [{"text": "Please give your final answer now, in plain text, based on the tool results above."}],
    })
    content = generate(contents, system_instruction=system_instruction, temperature=0.1)
    reply = "".join(p.get("text", "") for p in content.get("parts", [])).strip()
    reply = reply or "Sorry, I'm having trouble finishing that request."
    trace.append({"forced_final_answer": reply})
    return {"reply": reply, "trace": trace}
