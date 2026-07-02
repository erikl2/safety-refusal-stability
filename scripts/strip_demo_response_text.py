#!/usr/bin/env python3
"""strip_demo_response_text.py — make the demo prompts file strictly labels-only.

Deletes the truncated `text` preview from every response entry
(prompts[].models[*].responses[].text), keeping temp/seed/label. The prompt-level
`text` (the public AdvBench/HarmBench request) is retained — the demo needs it and it's
already-public benchmark content. Idempotent. CONTENT-MIN: never prints any field value,
only counts + the post-strip max string length (which should be the prompt text).

Usage: python scripts/strip_demo_response_text.py <path/to/prompts.json>
"""
import json
import sys


def maxstr(o):
    if isinstance(o, str):
        return len(o)
    if isinstance(o, dict):
        return max([maxstr(v) for v in o.values()] + [0])
    if isinstance(o, list):
        return max([maxstr(v) for v in o] + [0])
    return 0


def main() -> int:
    path = sys.argv[1]
    d = json.load(open(path))
    n = 0
    for p in d.get("prompts", []):
        for mv in p.get("models", {}).values():
            for r in mv.get("responses", []):
                if isinstance(r, dict) and "text" in r:
                    del r["text"]
                    n += 1
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    # verify nothing remains
    remaining = any("text" in r for p in d.get("prompts", [])
                    for mv in p.get("models", {}).values()
                    for r in mv.get("responses", []) if isinstance(r, dict))
    print(f"stripped responses[].text: {n} | remaining: {remaining} | "
          f"max string after strip: {maxstr(d)} (prompt text only)")
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
