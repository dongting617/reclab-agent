"""Token accounting and explicit-price estimates, never a provider billing claim."""
import math


def summarize_usage(events, prices=None):
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    requests = missing = failures = 0
    for event in events:
        if event.get("event") == "model_error":
            if event.get("request_not_sent") is True:
                continue
            requests += 1
            failures += 1
            missing += 1
            continue
        if event.get("event") != "model_turn":
            continue
        requests += 1
        usage = event.get("usage")
        if not isinstance(usage, dict):
            missing += 1
            continue
        values = [usage.get("input_tokens"), usage.get("output_tokens"),
                  (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)]
        if any(type(x) is not int or x < 0 for x in values) or values[2] > values[0]:
            missing += 1
            continue
        totals["input_tokens"] += values[0]
        totals["output_tokens"] += values[1]
        totals["cached_input_tokens"] += values[2]
    estimate = None
    if prices is not None:
        keys = {"currency", "input_per_million", "output_per_million", "cached_input_per_million"}
        if not isinstance(prices, dict) or set(prices) != keys or not isinstance(prices["currency"], str) or not prices["currency"].strip():
            raise ValueError("prices require currency and input/output/cached input rates per million tokens")
        for key in keys - {"currency"}:
            value = prices[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("token prices must be finite and nonnegative")
        estimate = ((totals["input_tokens"]-totals["cached_input_tokens"])*prices["input_per_million"]
                    +totals["cached_input_tokens"]*prices["cached_input_per_million"]
                    +totals["output_tokens"]*prices["output_per_million"])/1_000_000
    return {**totals, "model_requests": requests, "failed_requests": failures,
            "requests_without_usage": missing, "usage_complete": missing == 0,
            "estimated_cost": estimate, "currency": prices["currency"] if prices else None,
            "price_assumption": prices, "billing_verified": False,
            "note": "Estimate covers reported tokens only; unreported requests and GPU/service charges are excluded."}
