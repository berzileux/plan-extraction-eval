"""
Layer 1: turn a PDF plan into inputs for the rest of the pipeline.

  - renders each page to a PNG in plans/ (used by extract.py for geometry)
  - pulls the PDF's own text layer with exact coordinates into text_db/<name>.json
    (no OCR needed: CAD exports carry real text)

  python ingest.py plans_pdf/house.pdf --dpi 200 --inches-per-foot 0.25

A scanned PDF has no text layer. Then text_db is empty and read_labels.py falls
back to the OCR model on the rendered PNG.
"""
import argparse
import json
import sys
from pathlib import Path

import pymupdf as fitz

MAX_PAGES = 20


def ingest(pdf_path, dpi, plans_dir, db_dir):
    stem = Path(pdf_path).stem
    plans_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    out = []
    for pno in range(min(len(doc), MAX_PAGES)):
        page = doc[pno]
        pix = page.get_pixmap(dpi=dpi)
        png = "%s_p%d.png" % (stem, pno + 1)
        pix.save(str(plans_dir / png))
        k = dpi / 72.0
        items = []
        for x0, y0, x1, y1, word, block, line, wno in page.get_text("words"):
            items.append({"text": word, "x0": x0 * k, "y0": y0 * k, "x1": x1 * k, "y1": y1 * k})
        lines = merge_words(items)
        with open(db_dir / (Path(png).stem + ".json"), "w") as f:
            json.dump({"plan": png, "source": "pdf-text-layer", "items": lines}, f, indent=1)
        out.append((png, len(lines)))
    return out


def merge_words(words, gap_factor=1.2, keep_parts=False):
    """Group words into rows by vertical centre, then join neighbours that are close
    together, so '29'-5\"', 'x', '19'-5\"' become one item even when the 'x' sits lower."""
    items = sorted(words, key=lambda w: (w["y0"] + w["y1"]) / 2)
    rows = []
    for w in items:
        cy = (w["y0"] + w["y1"]) / 2
        h = max(1.0, w["y1"] - w["y0"])
        if rows and abs(cy - rows[-1]["cy"]) <= max(rows[-1]["h"], h) * 0.6:
            r = rows[-1]
            r["words"].append(w)
            n = len(r["words"])
            r["cy"] = (r["cy"] * (n - 1) + cy) / n
            r["h"] = max(r["h"], h)
        else:
            rows.append({"cy": cy, "h": h, "words": [w]})
    lines = []
    for r in rows:
        cur = None
        for w in sorted(r["words"], key=lambda w: w["x0"]):
            h = max(1.0, w["y1"] - w["y0"], r["h"] * 0.5)
            if cur is not None and 0 <= w["x0"] - cur["x1"] < h * gap_factor * 1.5:
                cur["text"] += " " + w["text"]
                cur["parts"].append(w)
                cur["x1"] = max(cur["x1"], w["x1"])
                cur["y0"] = min(cur["y0"], w["y0"])
                cur["y1"] = max(cur["y1"], w["y1"])
            else:
                if cur is not None:
                    lines.append(cur)
                cur = dict(w)
                cur["parts"] = [w]
        if cur is not None:
            lines.append(cur)
    out = []
    for L in lines:
        it = {"text": L["text"], "cx": (L["x0"] + L["x1"]) / 2, "cy": (L["y0"] + L["y1"]) / 2}
        if keep_parts:
            it["parts"] = [{"text": p["text"], "cx": (p["x0"] + p["x1"]) / 2,
                            "cy": (p["y0"] + p["y1"]) / 2, "x0": p["x0"], "x1": p["x1"]}
                           for p in L["parts"]]
        out.append(it)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--inches-per-foot", type=float, default=None,
                    help="drawing scale, e.g. 0.25 for 1/4 in = 1 ft, to print pixels_per_foot")
    ap.add_argument("--plans", default="plans")
    ap.add_argument("--text-db", default="text_db")
    args = ap.parse_args()
    if not Path(args.pdf).is_file():
        sys.exit("not a file: " + args.pdf)
    for png, n in ingest(args.pdf, args.dpi, Path(args.plans), Path(args.text_db)):
        msg = "%s: %d text lines from the PDF text layer" % (png, n)
        if args.inches_per_foot:
            msg += "; pixels_per_foot = %.1f" % (args.dpi * args.inches_per_foot)
        print(msg)
        if n == 0:
            print("  no text layer (scanned?). read_labels.py will use the OCR model instead.")


if __name__ == "__main__":
    main()
