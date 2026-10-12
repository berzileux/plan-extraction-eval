"""
Step 2: read room labels and printed dimensions with a small OCR model
(Florence-2 base, runs on CPU), match them to the rooms found by extract.py,
and flag rooms where the printed size and the measured size disagree.

The model reads text. It does not measure anything. Areas still come from code.

  python read_labels.py --trust-remote-code            # real model
  python read_labels.py --fake-ocr fake_ocr.json       # no model, for tests
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
from runtrace import stamp

import extract
from dims import parse_dims, strip_dims, clean_label

MODEL_NAME = "microsoft/Florence-2-base"
TILE = 768
OVERLAP = 128
DISAGREE_PCT = 10.0


def tiles(w, h):
    step = TILE - OVERLAP
    xs = list(range(0, max(1, w - OVERLAP), step)) or [0]
    ys = list(range(0, max(1, h - OVERLAP), step)) or [0]
    for y in ys:
        for x in xs:
            yield x, y, min(x + TILE, w), min(y + TILE, h)


def ocr_model(trust, revision):
    if not trust:
        sys.exit("Florence-2 loads code from its model repository. Review it, then re-run "
                 "with --trust-remote-code (and pin --revision to a commit you reviewed).")
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32, revision=revision).eval()
    proc = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, revision=revision)

    def run(pil_img):
        task = "<OCR_WITH_REGION>"
        inputs = proc(text=task, images=pil_img, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=512, num_beams=3, do_sample=False)
        raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
        res = proc.post_process_generation(raw, task=task, image_size=pil_img.size)[task]
        return list(zip(res.get("labels", []), res.get("quad_boxes", [])))

    return run


def ocr_tesseract(img_bgr, min_conf=30.0):
    """Free local OCR. Needs the tesseract program installed (apt install tesseract-ocr)."""
    import pytesseract
    from pytesseract import Output
    from ingest import merge_words
    h, w = img_bgr.shape[:2]
    scale = 2.0 if max(h, w) < 2000 else 1.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(gray, config="--oem 3 --psm 11", output_type=Output.DICT, timeout=300)
    words = []
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if not t or conf < min_conf:
            continue
        x0 = d["left"][i] / scale
        y0 = d["top"][i] / scale
        words.append({"text": t, "x0": x0, "y0": y0,
                      "x1": x0 + d["width"][i] / scale, "y1": y0 + d["height"][i] / scale})
    return merge_words(words)


def ocr_plan(img_bgr, run):
    from PIL import Image
    h, w = img_bgr.shape[:2]
    found = []
    for x0, y0, x1, y1 in tiles(w, h):
        crop = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        for text, quad in run(Image.fromarray(crop)):
            xs = quad[0::2]
            ys = quad[1::2]
            cx = x0 + sum(xs) / len(xs)
            cy = y0 + sum(ys) / len(ys)
            text = text.replace("</s>", "").replace("<s>", "").strip()
            if text:
                found.append({"text": text, "cx": cx, "cy": cy})
    seen, out = set(), []
    for f in found:
        key = (f["text"], round(f["cx"] / 20), round(f["cy"] / 20))
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def score_room(rm):
    """Set stated area, gap and review flag from rm['stated_dims_ft'] (if any)."""
    rm.pop("review_reason", None)
    if rm.get("stated_dims_ft"):
        stated = rm["stated_dims_ft"][0] * rm["stated_dims_ft"][1]
        rm["stated_sqft"] = round(stated, 1)
        diff = abs(rm["area_sqft"] - stated) / stated * 100
        rm["agreement_gap_pct"] = round(diff, 1)
        rm["needs_review"] = diff > DISAGREE_PCT
        if rm["needs_review"]:
            rm["review_reason"] = "printed size and measured area differ by %.0f%%" % diff
    else:
        rm.pop("stated_sqft", None)
        rm.pop("agreement_gap_pct", None)
        rm["needs_review"] = True
        rm["review_reason"] = "no printed dimensions found"


def attach(plan, items, masks):
    for rm in plan["rooms"]:
        rm["label_text"] = []
        rm["raw_text"] = []
        rm.pop("stated_dims_ft", None)
        rm.pop("dims_source", None)
    for it in items:
        x, y = int(it["cx"]), int(it["cy"])
        for rm in plan["rooms"]:
            m = masks[rm["id"]]
            if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x]:
                rm["raw_text"].append(it["text"])
                dims = parse_dims(it["text"])
                if dims:
                    if "stated_dims_ft" not in rm:
                        rm["stated_dims_ft"] = list(dims)
                        rm["dims_source"] = "rules"
                    rest = strip_dims(it["text"])
                    if rest:
                        rm["label_text"].append(rest)
                else:
                    cleaned = clean_label(it["text"])
                    if cleaned:
                        rm["label_text"].append(cleaned)
                break
    for rm in plan["rooms"]:
        rm["label"] = " ".join(rm["label_text"]) if rm["label_text"] else ""
        score_room(rm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="plans")
    ap.add_argument("--results", default="viewer/results.json")
    ap.add_argument("--fake-ocr", help="JSON {plan: [{text,cx,cy}]} instead of running the model")
    ap.add_argument("--text-db", default="text_db", help="folder written by ingest.py")
    ap.add_argument("--engine", choices=["florence", "tesseract"], default="florence",
                    help="text reader for image plans (PDF text layer is used first if present)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    fake = json.load(open(args.fake_ocr)) if args.fake_ocr else None
    run = None

    for plan in data["plans"]:
        img = cv2.imread(str(Path(args.plans) / plan["plan"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rooms = extract.extract_rooms(gray, plan["pixels_per_foot"])
        masks = {rm["id"]: rm["_mask"] for rm in rooms}
        db = Path(args.text_db) / (Path(plan["plan"]).stem + ".json")
        if fake is not None:
            items = fake.get(plan["plan"], [])
        elif db.is_file() and json.load(open(db)).get("items"):
            items = json.load(open(db))["items"]
            print("%s: using the PDF text layer (no OCR)" % plan["plan"])
        elif args.engine == "tesseract":
            items = ocr_tesseract(img)
            print("%s: read with Tesseract" % plan["plan"])
        else:
            if run is None:
                run = ocr_model(args.trust_remote_code, args.revision)
            items = ocr_plan(img, run)
        attach(plan, items, masks)
        flagged = sum(1 for r in plan["rooms"] if r["needs_review"])
        print("%s: %d text items, %d of %d rooms need review" % (
            plan["plan"], len(items), flagged, len(plan["rooms"])))

    data["method"] = "classical-cv areas + text from PDF layer or Florence-2" if fake is None else "fake-ocr test"
    print("group id: %s (layer 2 text lifter; the sheet was measured in group %s)" % (
        stamp(data, "read_labels"), data.get("group_id")))
    with open(args.results, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
