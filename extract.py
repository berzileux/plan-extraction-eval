"""
Baseline room-area extraction from 2D floor plan images using classical
computer vision. No model, no network, CPU only.

Method: threshold dark pixels as walls, close door-sized gaps, label the
enclosed empty regions, drop the exterior, convert pixels to square feet
using a per-plan scale from the manifest.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
from runtrace import stamp
import numpy as np

MAX_PIXELS = 25_000_000


def load_manifest(path):
    scales = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            scales[row["plan"].strip()] = float(row["pixels_per_foot"])
    return scales


def extract_rooms(gray, ppf, gap_ft=4.0, min_sqft=20.0, max_sqft=2500.0):
    h, w = gray.shape
    _, walls = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    r = max(1, int(gap_ft * ppf / 2))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sealed = cv2.dilate(walls, k)

    free = (sealed == 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(free, connectivity=4)

    border = set(np.unique(np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])))
    border.discard(0)

    labels = labels.astype(np.int32)
    for lab in border:
        labels[labels == lab] = 0

    # grow each room back into the wall band it lost to dilation
    orig_free = walls == 0
    for _ in range(r):
        grown = cv2.dilate(labels.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        fill = (labels == 0) & orig_free & (grown > 0)
        labels[fill] = grown[fill]

    # Text, furniture and fixtures inside a room are dark and cut holes in it.
    # Fill small holes that contain no other room so areas and text matching work.
    max_hole_px = 80.0 * ppf * ppf
    for lab in np.unique(labels):
        if lab == 0:
            continue
        m = (labels == lab).astype(np.uint8)
        cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        for c, hh in zip(cnts, hier[0]):
            if hh[3] < 0:
                continue
            hole = np.zeros_like(m)
            cv2.drawContours(hole, [c], -1, 1, thickness=-1)
            if hole.sum() > max_hole_px:
                continue
            inside = labels[hole.astype(bool)]
            if np.any((inside != 0) & (inside != lab) & (inside != -1)):
                continue
            hole_b = hole.astype(bool)
            if np.all((inside == 0) | (inside == lab)):
                labels[hole_b] = lab

    rooms = []
    rid = 0
    for lab in np.unique(labels):
        if lab == 0:
            continue
        mask = labels == lab
        px = int(mask.sum())
        sqft = px / (ppf * ppf)
        if sqft < min_sqft or sqft > max_sqft:
            continue
        rid += 1
        ys, xs = np.nonzero(mask)
        rooms.append({
            "id": rid,
            "area_px": px,
            "area_sqft": round(sqft, 1),
            "centroid": [int(xs.mean()), int(ys.mean())],
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "_mask": mask,
        })
    return rooms


def plan_quality(rooms, shape):
    """Say once, at plan level, when the geometry step is clearly unreliable."""
    h, w = shape[:2]
    m = max(6, int(0.01 * max(h, w)))
    edge = [r["id"] for r in rooms
            if r["bbox"][0] <= m or r["bbox"][1] <= m or r["bbox"][2] >= w - m or r["bbox"][3] >= h - m]
    tiny = [r["id"] for r in rooms if r["area_sqft"] < 40.0]
    frag = len(rooms) >= 8 and len(tiny) / len(rooms) >= 0.4
    reasons = []
    if edge:
        reasons.append("%d region(s) touch the page edge (probably the outside or margins counted as rooms)" % len(edge))
    if frag:
        reasons.append("%d of %d regions are under 40 sq ft (rooms probably broken into fragments)" % (len(tiny), len(rooms)))
    return {"geometry_reliable": not reasons, "reasons": reasons}


def draw_overlay(bgr, rooms):
    out = bgr.copy()
    rng = np.random.default_rng(7)
    layer = out.copy()
    for rm in rooms:
        color = tuple(int(c) for c in rng.integers(80, 230, 3))
        layer[rm["_mask"]] = color
    out = cv2.addWeighted(layer, 0.45, out, 0.55, 0)
    for rm in rooms:
        x0, y0 = rm["bbox"][0], rm["bbox"][1]
        txt = "#%d  %.0f sf" % (rm["id"], rm["area_sqft"])
        pos = (x0 + 8, y0 + 18)
        cv2.putText(out, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="plans")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out", default="viewer")
    args = ap.parse_args()

    scales = load_manifest(args.manifest)
    out_dir = Path(args.out)
    (out_dir / "out").mkdir(parents=True, exist_ok=True)

    results = {"method": "classical-cv-baseline", "plans": []}
    group = stamp(results, "extract")
    print("group id: %s" % group)
    for name, ppf in scales.items():
        src = Path(args.plans) / name
        if src.parent.resolve() != Path(args.plans).resolve():
            print("skip (path escapes plans dir):", name, file=sys.stderr)
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print("skip (unreadable):", name, file=sys.stderr)
            continue
        if img.shape[0] * img.shape[1] > MAX_PIXELS:
            print("skip (too large):", name, file=sys.stderr)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rooms = extract_rooms(gray, ppf)
        overlay = draw_overlay(img, rooms)
        out_name = Path(name).stem + "_overlay.png"
        cv2.imwrite(str(out_dir / "out" / out_name), overlay)
        results["plans"].append({
            "quality": plan_quality(rooms, img.shape),
            "plan": name,
            "group_id": group,
            "pixels_per_foot": ppf,
            "overlay": "out/" + out_name,
            "rooms": [{k: v for k, v in rm.items() if k != "_mask"} for rm in rooms],
        })
        print("%s: %d rooms" % (name, len(rooms)))

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
