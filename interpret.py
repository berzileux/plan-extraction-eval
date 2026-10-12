"""
Step 3: a small local language model reads the raw text found inside each room
and proposes a room name and printed size. Its answer is never trusted on its own:

  - it only sees text that OCR found inside that room (no images, no network),
  - a size is accepted only if both numbers can be found in that text,
  - if the rules from step 2 and the model disagree, the room is flagged.

Runs against a local Ollama server (default http://127.0.0.1:11434).

  python interpret.py --model qwen2.5:3b
"""
import argparse
import json
import re
import sys
from progress import Progress
from runtrace import stamp
import urllib.request
from urllib.parse import urlparse

from dims import normalize, parse_dims
from read_labels import score_room

SYSTEM = (
    "You read OCR text taken from inside ONE room of an architectural floor plan. "
    "The text may contain misreads. Return JSON only, with keys: "
    '"name" (the room name as written, or null), '
    '"width_ft" and "depth_ft" (the printed room dimensions in decimal feet, or null). '
    "Convert feet-and-inches to decimal feet (12'-6\" is 12.5). "
    "Use ONLY information present in the text. If a value is not printed, use null. "
    "Never estimate or invent a number."
)

_FT = re.compile(r"(\d{1,3})\s*(?:'|ft|FT)(?:\s*-?\s*(\d{1,2})\s*(?:\"|in|IN)?)?")
_PLAIN = re.compile(r"(?<![\d.])(\d{1,3})(?![\d'\"])")
TOL = 0.06


_SEP = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(?:'|\"|ft|FT)?\s*(?:x|X|\u00d7|by|BY)\s*(\d{1,3}(?:\.\d+)?)(?![\d.])")


def candidate_pairs(texts):
    """Every (width, depth) pair printed in ONE text item, in feet. A size counts as grounded
    only if both numbers sit in the same printed pair ('12 by 14', "12'-6\" x 14'"). Loose digits
    elsewhere in the room's text (OCR noise, door numbers) never count."""
    pairs = []
    for raw in texts:
        t = normalize(raw or "")
        p = parse_dims(t)
        if p:
            pairs.append((p[0], p[1]))
        for m in _SEP.finditer(t):
            pairs.append((float(m.group(1)), float(m.group(2))))
    return pairs


def grounded_pair(w, d, pairs):
    return any((abs(w - a) <= TOL and abs(d - b) <= TOL) or (abs(w - b) <= TOL and abs(d - a) <= TOL)
               for a, b in pairs)


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if 3 <= v <= 60 else None


def clean_name(v):
    if not isinstance(v, str):
        return ""
    v = re.sub(r"[^A-Za-z0-9 '\-/&.]", "", v).strip()
    return v[:40]


def interpret_room(rm, call):
    texts = rm.get("raw_text") or []
    if not texts:
        return
    try:
        out = call(texts)
    except Exception as e:  # network, JSON, timeout: never crash the run
        rm["model_error"] = str(e)[:120]
        return
    if not isinstance(out, dict):
        rm["model_error"] = "model did not return an object"
        return

    name = clean_name(out.get("name"))
    if name and not rm.get("label"):
        rm["label"] = name
        rm["label_source"] = "model"

    w, d = _num(out.get("width_ft")), _num(out.get("depth_ft"))
    if w is None or d is None:
        return
    if not grounded_pair(w, d, candidate_pairs(texts)):
        rm["model_note"] = "model proposed a size that is not in the text; ignored"
        return

    rules = rm.get("stated_dims_ft")
    if rules:
        if abs(rules[0] - w) > 0.1 or abs(rules[1] - d) > 0.1:
            rm["needs_review"] = True
            rm["review_reason"] = "rules and model read different printed sizes"
            rm["model_dims_ft"] = [round(w, 2), round(d, 2)]
        return
    rm["stated_dims_ft"] = [round(w, 2), round(d, 2)]
    rm["dims_source"] = "model"
    if name:
        rm["label"] = name
        rm["label_source"] = "model"
    score_room(rm)


def local_only(url):
    return urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")


def make_ollama(url, model):
    def call(texts):
        body = {
            "model": model, "stream": False, "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps({"ocr_strings": texts})},
            ],
        }
        req = urllib.request.Request(
            url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(json.load(r)["message"]["content"])
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="viewer/results.json")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--quiet", action="store_true", help="no progress line")
    ap.add_argument("--allow-remote", action="store_true",
                    help="allow a non-local server (plan text is sent to it)")
    args = ap.parse_args()
    if not local_only(args.url) and not args.allow_remote:
        sys.exit("Refusing to send plan text to a non-local server. Use --allow-remote to override.")

    with open(args.results) as f:
        data = json.load(f)
    call = make_ollama(args.url, args.model)
    # Count first so the report has a real denominator. A room with no text read
    # out of it is never sent to the model.
    work = [(plan, rm) for plan in data["plans"] for rm in plan["rooms"] if rm.get("raw_text")]
    bar = Progress(len(work), "rooms", quiet=args.quiet)
    n = 0
    for _, rm in work:
        interpret_room(rm, call)
        n += 1
        bar.tick()
    took = bar.finish()
    data["method"] = data.get("method", "") + " + local LLM interpretation (" + args.model + ")"
    gid = stamp(data, "interpret")
    data["layers"]["interpret"]["model"] = args.model
    data["layers"]["interpret"]["server"] = args.url
    with open(args.results, "w") as f:
        json.dump(data, f, indent=2)
    errs = sum(1 for p in data["plans"] for r in p["rooms"] if r.get("model_error"))
    data["layers"]["interpret"]["seconds"] = round(took, 1)
    print("%d rooms interpreted in %s, %d model errors (group id %s; the regions came "
          "from group %s)" % (n, Progress._clock(took), errs, gid, data.get("group_id")))


if __name__ == "__main__":
    main()
