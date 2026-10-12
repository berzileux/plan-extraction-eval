"""Layer 2b: find the dimension chains on a plan and turn them into trusted lengths.

A dimension chain is a thin straight line with extension lines rising from each
joint, and a printed length written on each segment. The chain is not evidence
about the geometry, it is the geometry, already in feet. Finding it binds every
number to a measured span of pixels instead of to whatever text happens to sit
near a region.

The reader is allowed to be bad. Reading is the expensive, unreliable step, so
as little of it as possible is done:

  1. a cheap reader (Tesseract) tries every segment
  2. anything that disagrees with its own drawn length is thrown away
  3. whatever is left must sum to the chain's printed overall, and unread gaps
     that measure the same as each other are the same dimension repeated
  4. only what is still unresolved goes to an expensive reader

On the Putnam sheet that is two calls to a vision model instead of eleven.
The arithmetic lives in dimensions.py and is tested there.

  python dimlines.py                        # every plan in manifest.csv, Tesseract
  python dimlines.py --reader trocr         # hand-lettered or scanned sheets
  python dimlines.py --only putnam.png --expensive ollama --model qwen2.5vl:3b --threads 4
  python dimlines.py --only putnam.png --max-chains 2 --debug    # why a chain read nothing

  python dimlines.py --help                 # every flag

Pick the cheap reader to match the sheet. Tesseract is for printed CAD text; on hand
lettering it reads almost nothing, however high the resolution, and trocr is the one
to use.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from dimensions import (believe, fmt, group_gaps, parse_length, scale_from_chain,
                        solve_chain)
from progress import Progress
from ruleset import rule
from runtrace import Trace, new_group, stamp
from runtrace import version as rt_version

PLANS = Path("plans")
OUT = Path("viewer/out")
MAX_PIXELS = 120_000_000

# A line needs at least this many joints before it is treated as a chain of
# segments. Three, not four: an interior chain often runs wall to wall, and a wall
# is a full-height line that the tick finder deliberately ignores, so its outermost
# joint is invisible. The chain 15'3" | 7'10.5" | 14'10" on the sample sheet comes
# back with three joints and its middle two segments, and those two carry a room
# width. Requiring four discarded every interior chain on the drawing, which is
# where the room dimensions are, while keeping only the perimeter chains, which
# measure window piers and describe no room at all.
#
# Two joints would be one segment with nothing to cross-check it, so three is the
# floor. What keeps the false ones out is MIN_READINGS: with three joints there are
# only two segments, and both have to be read and believed.
MIN_CHAIN_TICKS = rule("detection.min_chain_ticks")
# How far from a chain its overall may sit, and how closely their spans must agree.
OVERALL_NEAR_FT = rule("detection.overall_near_ft")
OVERALL_SPAN_TOL = rule("detection.overall_span_tol")
# A line with fewer believed readings than this is not a dimension chain. Stair
# treads and hatching produce convincing tick patterns and no numbers at all.
MIN_READINGS = rule("reading.min_readings")

PROMPT = (
    "Transcribe the handwritten text in this image, character for character. "
    "It is one architectural dimension: a number, a foot mark, then inches, "
    "sometimes with a fraction. Copy only what you can actually see. "
    "Do not guess, do not complete a partial number, do not repeat an example. "
    "Reply with the transcription alone and nothing else. "
    "If the image has no readable text, reply with a single question mark."
)


# ---------------------------------------------------------------- detection

def thin_runs(ink, orient, min_px, max_thick, bridge_ft=None, ppf=None):
    """Straight ink runs at least min_px long and no more than max_thick across.

    Horizontal returns (y, x0, x1), vertical returns (x, y0, y1).

    Gaps are closed before the run is measured. A dimension line is interrupted
    wherever a glyph sits on it or an arrowhead meets it, and without closing, one
    line comes back as several pieces whose spans match nothing. It is invisible at
    low resolution, where the gaps are a pixel or two, and breaks everything at full
    size, where the same gap is ten pixels: an overall line split in half stops
    matching the chain it belongs to, so no chain total is ever read and no gap in
    any chain can be solved.
    """
    bridge_ft = rule("detection.bridge_ft") if bridge_ft is None else bridge_ft
    if bridge_ft and ppf:
        bridge = max(3, int(bridge_ft * ppf))
        ck = (1, bridge) if orient == "v" else (bridge, 1)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, ck))
    k = (1, min_px) if orient == "v" else (min_px, 1)
    m = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if orient == "v" and w <= max_thick:
            out.append((x + w // 2, y, y + h))
        if orient == "h" and h <= max_thick:
            out.append((y + h // 2, x, x + w))
    return out


def candidate_lines(gray, ppf):
    """Every long thin horizontal line, with the joints found on it.

    Run this on a transposed image to get the vertical ones.
    """
    ink = (gray < 150).astype(np.uint8)
    thick = max(3, int(0.25 * ppf))
    # Lines get their gaps closed; ticks do not. A tick is short, and bridging would
    # weld neighbouring ones into a single run and move every joint.
    ticks = thin_runs(ink, "v", max(8, int(0.9 * ppf)), thick, bridge_ft=0)
    lines = thin_runs(ink, "h", int(8 * ppf), thick, ppf=ppf)
    out = []
    for y, x0, x1 in lines:
        near = sorted(x for x, a, b in ticks
                      if x0 - thick * 2 <= x <= x1 + thick * 2
                      and a - thick * 3 <= y <= b + thick * 3
                      and (b - a) <= int(rule("detection.tick_max_ft") * ppf))
        merged = []
        for x in near:
            if merged and x - merged[-1][-1] <= thick * 2:
                merged[-1].append(x)
            else:
                merged.append([x])
        out.append({"y": int(y), "x0": int(x0), "x1": int(x1),
                    "ticks": [int(np.mean(g)) for g in merged]})
    return out


# ANCHOR BEFORE 20261011_173128
def _ray_ink(ink, x, y, dx, dy, r0, r1, skip):
    """Share of ink along a ray from (x, y), leaving out points within `skip` rows of y."""
    hits = n = 0
    for k in np.arange(r0, r1, 1.0):
        px, py = int(round(x + dx * k)), int(round(y + dy * k))
        if abs(py - y) <= skip:
            continue
        n += 1
        if 0 <= py < ink.shape[0] and 0 <= px < ink.shape[1] and ink[py, px]:
            hits += 1
    return hits / n if n >= 4 else None


def _line_centre(ink, x, y, ppf):
    """Where the drawn line really runs beside x, and its half thickness.

    The row a line is found on is the middle of a closed, opened run and can sit a
    few pixels off the ink, which puts one side's probes inside the line itself.
    """
    lim = max(3, int(0.12 * ppf))
    w = int(0.5 * ppf)
    cs, hs = [], []
    for cx in list(range(x - w, x - w // 3)) + list(range(x + w // 3, x + w)):
        if not 0 <= cx < ink.shape[1]:
            continue
        idx = np.flatnonzero(ink[max(0, y - lim):y + lim + 1, cx])
        if len(idx) == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        r = min(runs, key=lambda r: abs((r[0] + r[-1]) / 2 - lim))
        if r[-1] - r[0] + 1 <= lim + 2:
            cs.append(max(0, y - lim) + (r[0] + r[-1]) / 2)
            hs.append((r[-1] - r[0] + 1) / 2)
    if len(cs) < 5:
        return y, max(2, int(0.04 * ppf))
    return int(round(np.median(cs))), int(np.ceil(np.median(hs))) + 1


def _stroke(ink, x, y, pointing, side, ppf, skip):
    """The angle of an arrowhead stroke leaving a tip at (x, y) on one side, or None."""
    r0 = rule("detection.arrow_stroke_min_ft") * ppf
    r1 = rule("detection.arrow_stroke_max_ft") * ppf
    w = rule("detection.arrow_stroke_width_ft") * ppf
    for ang in range(int(rule("detection.arrow_angle_min_deg")),
                     int(rule("detection.arrow_angle_max_deg")) + 1, 5):
        a = np.radians(ang)
        dx, dy = -pointing * np.cos(a), side * np.sin(a)
        on = _ray_ink(ink, x, y, dx, dy, r0, r1, skip)
        if on is None or on < rule("detection.arrow_stroke_ink"):
            continue
        # Paper on both sides of the stroke over its far half: a pen stroke, not a wall
        # or a band of ink the probe happens to run inside.
        flanks = [_ray_ink(ink, x - q * w * dy, y + q * w * dx, dx, dy, (r0 + r1) / 2, r1, -1)
                  for q in (-1, 1)]
        if all(f is not None and f <= rule("detection.arrow_flank_ink") for f in flanks):
            return ang
    return None


def arrow_tips(gray, line, ppf, lo, hi):
    """Arrowhead tips on a horizontal line between x=lo and x=hi, as (x, pointing).

    An arrowhead is two short strokes, one either side of the line, converging on
    the point where the line meets what it measures. pointing is +1 for a tip
    pointing to larger x. A place where both directions answer is two lines
    crossing, not an arrowhead, and is dropped.
    """
    ink = gray < 150
    found = []
    for x in range(max(0, lo), min(gray.shape[1] - 1, hi) + 1):
        y, skip = _line_centre(ink, x, line["y"], ppf)
        for pointing in (1, -1):
            if (_stroke(ink, x, y, pointing, -1, ppf, skip) is not None
                    and _stroke(ink, x, y, pointing, 1, ppf, skip) is not None):
                found.append((x, pointing))
    both = {x for x, p in found if (x, -p) in found}
    return [(x, p) for x, p in found if all(abs(x - b) > 3 for b in both)]


def add_end_joints(gray, line, ppf):
    """Give a chain the joints at the walls its line ends against.

    The tick finder ignores anything longer than a tick, so where a chain's line
    runs on to the wall it measures to, the last stretch has no joint and is never
    cut into a piece or read: 18'7" at the bottom of BED ROOM C, 15'7" and 13'6" on
    the second floor. The mark there is the arrowhead the draughtsman drew, its tip
    touching the wall. This looks for that mark only near where the line stops,
    beyond the outermost tick, on a stretch long enough to carry a room dimension,
    and takes the outermost tip pointing outward. It never decides whether the line
    is a dimension; the line is already a chain.
    """
    window = int(rule("detection.end_mark_window_ft") * ppf)
    shortest = rule("detection.single_min_ft") * ppf
    t = line["ticks"]
    marks = []
    if t and t[0] - line["x0"] >= shortest:
        tips = [x for x, p in arrow_tips(gray, line, ppf, line["x0"], line["x0"] + window) if p == -1]
        if tips:
            marks.append(min(tips))
    if t and line["x1"] - t[-1] >= shortest:
        tips = [x for x, p in arrow_tips(gray, line, ppf, line["x1"] - window, line["x1"]) if p == 1]
        if tips:
            marks.append(max(tips))
    if marks:
        line["ticks"] = sorted(t + marks)
        line["end_marks"] = sorted(marks)
    return line
# ANCHOR AFTER 20261011_173128


def span_px(line):
    t = line["ticks"]
    return (t[-1] - t[0]) if len(t) >= 2 else (line["x1"] - line["x0"])


def drop_duplicate_chains(chains, ppf):
    """Two lines can cover the same run: the segmented chain and its overall.

    Keep the one with more joints, since an overall carries only its arrowheads.
    """
    # ANCHOR BEFORE 20261012_033432
    # A copy has every joint on a joint of the other line: the same line found twice, or
    # an overall whose joints are its ends and the chain's extension lines running on
    # through it. Equal length and a nearby row are not enough. Room widths drawn over the
    # same run as a row of window piers are two chains, each with joints the other lacks,
    # and a line on the other plan of the sheet can be just as long a few feet away.
    tol = max(3, int(rule("detection.line_max_thick_ft") * ppf))

    def joints_on(a, b):
        return all(any(abs(x - y) <= tol for y in b["ticks"]) for x in a["ticks"])
    # ANCHOR AFTER 20261012_033432
    kept = []
    for c in sorted(chains, key=lambda c: (-span_px(c), -len(c["ticks"]))):
        near = None
        for k in kept:
            if (abs(span_px(k) - span_px(c)) / ppf < 1.0
                    and abs(k["y"] - c["y"]) < OVERALL_NEAR_FT * ppf
                    # ANCHOR BEFORE 20261012_033433
                    and (joints_on(c, k) or joints_on(k, c))):
                    # ANCHOR AFTER 20261012_033433
                near = k
                break
        if near is None:
            kept.append(c)
        elif len(c["ticks"]) > len(near["ticks"]):
            kept[kept.index(near)] = c
    return kept


def vertical_view(gray):
    """The image turned so a vertical chain reads as a horizontal one.

    Rotation, not transposition. Transposing mirrors the sheet, and mirrored
    hand lettering is unreadable to every OCR engine; a quarter turn clockwise
    leaves the rotated dimension text upright, which is how it is drawn.
    """
    return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)


def unrotate(rx, ry, height):
    """A point in the rotated view back to where it sits on the sheet."""
    return int(ry), int(height - 1 - rx)


def detect(gray, ppf, axis="both"):
    """Chains and overall-line candidates on one or both axes.

    Vertical chains are found in the rotated view and stay in its coordinates;
    read_chain is given the same view, and only the overlay maps back.

    Returns (chains, singles). A single is a line that measures one thing between
    two arrowheads: most room dimensions are drawn that way, and only subdivided
    runs become chains.
    """
    found, singles = [], []
    for ax in (("h", "v") if axis == "both" else (axis,)):
        src = gray if ax == "h" else vertical_view(gray)
        lines = candidate_lines(src, ppf)
        chains = drop_duplicate_chains(
            # ANCHOR BEFORE 20261011_173129
            [add_end_joints(src, l, ppf) for l in lines if len(l["ticks"]) >= MIN_CHAIN_TICKS], ppf)
            # ANCHOR AFTER 20261011_173129
        for c in chains:
            c["axis"] = ax
            # The overall is a parallel line covering the same run. It is NOT
            # identified by having fewer joints: the chain's extension lines carry on
            # past it and register as joints on it too, and the higher the resolution
            # the more of them are found, so at full size an overall can end up with
            # more joints than the chain it belongs to. Match on position and span
            # only, and try the sparsest first, since that is most often the real one.
            near, rejected = [], []
            for o in lines:
                if o is c:
                    continue
                dy = abs(o["y"] - c["y"])
                dspan = abs(span_px(o) - span_px(c))
                if len(o["ticks"]) < 2:
                    continue
                if dy >= OVERALL_NEAR_FT * ppf:
                    continue
                if dspan > OVERALL_SPAN_TOL * max(1, span_px(c)):
                    rejected.append((o["y"], len(o["ticks"]), round(dspan / ppf, 2)))
                    continue
                near.append(o)
            c["overall_candidates"] = sorted(near, key=lambda o: len(o["ticks"]))
            if DEBUG["on"]:
                print("      %d overall candidate(s); %d parallel line(s) rejected on span%s"
                      % (len(near), len(rejected),
                         (": " + ", ".join("y=%d ticks=%d off by %.2f ft" % r
                                           for r in rejected[:4])) if rejected else ""))
            found.append(c)
        for l in single_candidates(lines, chains, ppf):
            l["axis"] = ax
            singles.append(l)
    # ANCHOR BEFORE 20261012_033434
    mark_inside(found, gray.shape[0], ppf)
    # ANCHOR AFTER 20261012_033434
    found.sort(key=lambda c: -span_px(c))
    singles.sort(key=lambda l: -(l["x1"] - l["x0"]))
    return found, singles


def by_axis(chains, limit):
    """Take the longest `limit` chains on each axis, not the longest overall.

    One cap across both axes is a trap: a building is usually taller in feet
    than any single run is wide, so the vertical chains fill every slot and the
    horizontal ones, which is where most printed lengths are, never get read.
    """
    out = []
    for ax in ("h", "v"):
        # ANCHOR BEFORE 20261012_033435
        # Inside the building first, then the perimeter, each longest first. A perimeter
        # chain is long and regular, so on length alone it wins every slot, and it
        # measures window piers; the interior chains that measure rooms were left over.
        mine = sorted((c for c in chains if c["axis"] == ax), key=lambda c: not c.get("inside"))
        out.extend(mine[:limit])
        # ANCHOR AFTER 20261012_033435
    return out


# ANCHOR BEFORE 20261012_033436
def mark_inside(chains, height, ppf):
    """Set "inside" on each chain that runs inside a building rather than around it.

    A chain inside the footprint is crossed, within its own run, by a chain on the other
    axis that has joints on both sides of it: the rooms that chain measures lie on either
    side of this line. A perimeter chain lies beyond the outer wall, where the chains on
    the other axis have already stopped. Only the chains are used, because the room
    regions that would give the footprint directly do not exist until the next layer.
    """
    tol = max(3, int(rule("detection.line_max_thick_ft") * ppf))

    def on_sheet(c):
        # (position across the line, joint positions along it), in sheet pixels. A
        # vertical chain lives in the rotated view: its row is the sheet x, and a rotated
        # x maps back to sheet y = height - 1 - x.
        if c["axis"] == "h":
            return c["y"], list(c["ticks"])
        return c["y"], [height - 1 - t for t in c["ticks"]]

    placed = [(c, on_sheet(c)) for c in chains]
    for c, (cross, joints) in placed:
        lo, hi = min(joints), max(joints)
        c["inside"] = any(
            o["axis"] != c["axis"] and lo - tol <= o_cross <= hi + tol
            and any(j < cross - tol for j in o_joints) and any(j > cross + tol for j in o_joints)
            for o, (o_cross, o_joints) in placed)
    return chains
# ANCHOR AFTER 20261012_033436


# ANCHOR BEFORE 20261011_153402
# Written beside the dimension file under a name of its own, so a geometry pass can
# never be mistaken for a reading run or replace what one wrote.
LINES_SUFFIX = "_line_geometry"


def believed_lines(dim_data, height):
    """Every line a reading proved is a dimension, in detection-view coordinates.

    A chain line counts when at least one of its segments was read and believed (its
    source is "read": a reading that disagreed with its drawn length is left as a gap
    and listed under rejected). A lone line counts when its single reading was believed.
    Solved segments are arithmetic, not readings, and prove nothing on their own.
    """
    out = []
    for c in (dim_data or {}).get("chains") or []:
        reads = [s for s in c.get("segments") or [] if s.get("source") == "read"]
        if reads:
            out.append({"axis": c["axis"], "row": int(c["y"]), "x0": int(c["x0"]),
                        "x1": int(c["x1"]), "what": "%s chain at %d" % (c["axis"], c["y"]),
                        "evidence": [s.get("printed") for s in reads]})
    for d in (dim_data or {}).get("single_dimensions") or []:
        if d.get("source") != "read":
            continue
        lo, hi = (d["lo"], d["hi"]) if d["axis"] == "h" else (height - 1 - d["hi"], height - 1 - d["lo"])
        out.append({"axis": d["axis"], "row": int(d["cross"]), "x0": int(lo), "x1": int(hi),
                    "what": d.get("chain") or "lone dimension", "evidence": [fmt(d["ft"])]})
    return out


def line_geometry(gray, ppf, dim_data):
    """The dimension lines to take out of the wall mask, found with no reading at all.

    Only a line that carries a believed reading is taken out. Shape is not evidence: a
    single-line wall and a wall-to-wall dimension line look the same and separate space
    the same way, and two guards built on shape broke walls on the sample sheet. A
    detected line is tied to a reading when both sit on the same axis, within a line's
    width of the same row, and their spans overlap by at least nine tenths of the
    shorter. Each erased stretch is pulled back from its ends by a line's width, so it
    stops short of the wall it meets. Coordinates are sheet pixels.
    """
    lo_ft, hi_ft = rule("detection.single_min_ft"), rule("detection.single_max_ft")
    clear = max(1, int(rule("detection.line_max_thick_ft") * ppf))
    height = gray.shape[0]
    proven = believed_lines(dim_data, height)
    used = set()
    out = {"candidates": 0, "believed_lines": len(proven), "stretches": [], "kept": [],
           "unmatched": []}
    for ax in ("h", "v"):
        src = gray if ax == "h" else vertical_view(gray)
        for l in candidate_lines(src, ppf):
            if len(l["ticks"]) < SINGLE_MIN_TICKS:
                continue
            if not lo_ft <= (l["x1"] - l["x0"]) / ppf <= hi_ft:
                continue
            out["candidates"] += 1
            match = None
            for i, p in enumerate(proven):
                overlap = min(l["x1"], p["x1"]) - max(l["x0"], p["x0"])
                shorter = min(l["x1"] - l["x0"], p["x1"] - p["x0"])
                if (p["axis"] == ax and abs(p["row"] - l["y"]) <= clear
                        and shorter > 0 and overlap >= 0.9 * shorter):
                    match = i
                    break
            a, b = l["x0"] + clear, l["x1"] - clear
            if ax == "h":
                seg = [a, l["y"], b, l["y"]]
            else:
                (xa, ya), (xb, yb) = unrotate(a, l["y"], height), unrotate(b, l["y"], height)
                seg = [xa, min(ya, yb), xb, max(ya, yb)]
            rec = {"axis": ax, "line": [int(v) for v in seg]}
            if match is None:
                out["kept"].append(rec)
                continue
            used.add(match)
            rec.update(what=proven[match]["what"], evidence=proven[match]["evidence"])
            out["stretches"].append(rec)
    out["unmatched"] = [p["what"] for i, p in enumerate(proven) if i not in used]
    return out
# ANCHOR AFTER 20261011_153402


# ------------------------------------------------------------------ reading

# A lone dimension line measures one thing between two arrowheads and has no
# intermediate joints. Most room dimensions are drawn this way; only runs that are
# subdivided become chains. These bounds keep the search to things that could be a
# room: shorter is a jamb or a closet, longer is a building outline.
SINGLE_MIN_FT = rule("detection.single_min_ft")
SINGLE_MAX_FT = rule("detection.single_max_ft")
SINGLE_MIN_TICKS = rule("detection.single_min_ticks")
SINGLE_MAX_TEXT_FT = rule("detection.single_max_text_ft")


def single_candidates(lines, chains, ppf):
    """Lines that could each be one dimension rather than a chain of them.

    A sheet is full of long thin horizontal lines that are not dimensions: the title
    block's rules, the border, the scale bar, the underline of every block of notes.
    Length alone does not tell them apart, and feeding them to the reader returns
    things like 'Tional parks buildings and reserve'.

    What a dimension line has and a rule does not is a mark at each end, where it
    meets what it measures. Requiring both ends to be marked is what separates them.
    """
    taken = {id(c) for c in chains}
    out = []
    for l in lines:
        if id(l) in taken or len(l["ticks"]) >= MIN_CHAIN_TICKS:
            continue
        if len(l["ticks"]) < SINGLE_MIN_TICKS:
            continue
        ft = (l["x1"] - l["x0"]) / ppf
        if SINGLE_MIN_FT <= ft <= SINGLE_MAX_FT:
            out.append(l)
    out.sort(key=lambda l: -(l["x1"] - l["x0"]))
    return out


# ANCHOR BEFORE 20261011_142336
def read_single(gray, line, ppf, reader, axis, height, why=None, record=None, crops_dir=None):
    # ANCHOR AFTER 20261011_142336
    """Read a lone dimension line, and believe it only if it measures what it says.

    There is no chain total to check against here, so the line's own length is the
    only evidence — which is enough, and is exactly what `believe` is for. A line
    whose printed text matches its own drawn length is a dimension line; a wall that
    happens to have a number near it is not, and will not match.

    record, when given, gets one entry per candidate whatever happened to it: where
    the line is, how long it is drawn, what the reader said and the outcome. With
    crops_dir, every image the reader was shown for a failed candidate is saved there.
    """
    # ANCHOR BEFORE 20261011_142337
    lo_s, hi_s = (line["x0"], line["x1"]) if axis == "h" else (
        height - 1 - line["x1"], height - 1 - line["x0"])
    entry = {"axis": axis, "lo": int(lo_s), "hi": int(hi_s), "cross": int(line["y"]),
             "drawn_ft": round((line["x1"] - line["x0"]) / ppf, 2), "reader": reader.kind,
             "raw_text": None, "stated_ft": None, "outcome": None, "crops": []}
    seen = []

    def note(key, detail=None):
        entry["outcome"] = key
        if record is not None:
            if crops_dir is not None and key != "believed" and seen:
                Path(crops_dir).mkdir(parents=True, exist_ok=True)
                for tag, img, _ in seen:
                    name = "%s_%d_%d-%d_%s.png" % (axis, line["y"], entry["lo"], entry["hi"],
                                                   tag.replace("+", "_"))
                    cv2.imwrite(str(Path(crops_dir) / name), img)
                    entry["crops"].append(name)
            entry["views"] = [{"view": tag, "text": t} for tag, _, t in seen]
            record.append(entry)
    # ANCHOR AFTER 20261011_142337
        if why is not None:
            why[key] = why.get(key, 0) + 1
            if DEBUG["on"] and detail:
                print("      %-6s %-9s %s" % (axis, fmt(measured), detail))

    measured = (line["x1"] - line["x0"]) / ppf
    band = crop_segment(gray, line, line["x0"], line["x1"], ppf)
    if band is None:
        note("no crop")
        return None
    tight = tight_text_crop(band, ppf)
    if tight is None:
        note("nothing written on it")
        return None
    # A dimension is a few characters. A line of notes or a title is not, and the
    # width of the writing says which before a reader is ever called.
    if tight.shape[1] > SINGLE_MAX_TEXT_FT * ppf:
        note("the writing on it is prose, not a number",
             "writing spans %.1f ft, too long to be a dimension" % (tight.shape[1] / ppf))
        return None
    # ANCHOR BEFORE 20261011_142338
    text = read_crop(tight, ppf, reader, axis, seen=seen)
    ft = parse_length(text)
    entry["raw_text"] = text if text else (seen[-1][2] if seen else None)
    entry["stated_ft"] = ft
    # ANCHOR AFTER 20261011_142338
    if ft is None:
        note("unreadable", "unreadable: %r" % text)
        return None
    if not believe(ft, measured):
        note("disagreed", "says %s but measures %s (%.0f%% out)"
             % (fmt(ft), fmt(measured), abs(ft - measured) / measured * 100))
        return None
    note("believed", "reads %s, measures %s" % (fmt(ft), fmt(measured)))
    lo, hi = (line["x0"], line["x1"]) if axis == "h" else (
        height - 1 - line["x1"], height - 1 - line["x0"])
    return {"ft": ft, "axis": axis, "lo": int(lo), "hi": int(hi), "cross": int(line["y"]),
            "source": "read", "reader": reader.kind, "raw_text": text,
            "measured_ft": round(measured, 2),
            "chain": "%s dimension at %d" % (axis, line["y"])}


def erase_rules(crop, ppf):
    """Remove the dimension line and the extension lines, keep the glyphs."""
    ink = (crop < 160).astype(np.uint8)
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (max(6, int(1.2 * ppf)), 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, crop.shape[0])))
    out = crop.copy()
    out[cv2.dilate(h | v, np.ones((2, 2), np.uint8)) > 0] = 255
    return out


# ANCHOR BEFORE 20261011_180853
# ANCHOR BEFORE 20261011_214903
def remove_chain_line(crop, chain, ppf, edge=False):
    # ANCHOR AFTER 20261011_214903
    """Take a chain's own line out of a crop cut by crop_segment, keeping the numerals.

    The line runs through the middle of the number on most of the sample sheet's interior
    chains, and the reader takes it for part of a glyph: 13'6" came back as 43'6" and as
    2', and 24'4" as ( 2474 ". Its row is known from the chain, so only those rows are
    cleared, and only in columns where no stroke carries on above or below it, so a
    numeral crossing the line keeps its stroke.
    """
    if crop is None:
        return None
    band = max(10, int(0.8 * ppf))
    row = chain["y"] - max(0, chain["y"] - band)
    ink = crop < 160
    lim = max(3, int(0.12 * ppf))
    lo, hi = max(0, row - lim), min(crop.shape[0], row + lim + 1)
    if lo >= hi:
        return crop
    prof = ink[lo:hi].sum(axis=1)
    if prof.max() == 0:
        return crop
    p = lo + int(np.argmax(prof))
    r0 = r1 = p
    while r0 - 1 >= lo and ink[r0 - 1].sum() >= 0.5 * prof.max():
        r0 -= 1
    while r1 + 1 < hi and ink[r1 + 1].sum() >= 0.5 * prof.max():
        r1 += 1
    # ANCHOR BEFORE 20261011_214904
    # With edge, the scanned line's ragged edge, a row or two deep each side, goes too.
    # Left behind it reads as a dotted line through the numerals (15'7" was lost to it),
    # but clearing it on every crop cost 17'11" and 12'6", so it is only used to retry a
    # crop that read nothing.
    if edge:
        for _ in range(2):
            if r0 - 1 >= 0 and ink[r0 - 1].sum() >= 0.1 * prof.max():
                r0 -= 1
            if r1 + 1 < crop.shape[0] and ink[r1 + 1].sum() >= 0.1 * prof.max():
                r1 += 1
    # ANCHOR AFTER 20261011_214904
    out = crop.copy()
    above = ink[r0 - 2] if r0 - 2 >= 0 else np.zeros(crop.shape[1], bool)
    below = ink[r1 + 2] if r1 + 2 < crop.shape[0] else np.zeros(crop.shape[1], bool)
    out[r0:r1 + 1, ~(above | below)] = 255
    return out
# ANCHOR AFTER 20261011_180853


def prep(crop, target_h=64):
    f = max(1.0, target_h / max(1, crop.shape[0]))
    c = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    c = cv2.threshold(cv2.GaussianBlur(c, (3, 3), 0), 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.copyMakeBorder(c, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)


class Reader:
    """One cropped segment in, one string out. Loaded lazily, so the expensive
    tier costs nothing until something actually needs it."""

    def __init__(self, kind, model=None, url=None, timeout=600, threads=0):
        self.kind, self.name, self.calls = kind, model, 0
        if kind == "tesseract":
            import pytesseract
            import shlex
            self.pt = pytesseract
            self.cfg = "--psm %d -c " + shlex.quote("tessedit_char_whitelist=0123456789'\"")
        elif kind == "trocr":
            # transformers prints the merged encoder/decoder config and warns that
            # encoder.pooler is freshly initialised. Both are left visible on purpose:
            # generation never touches the pooler, so the warning is harmless, but the
            # dump records exactly which model and settings produced a run.
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.name = model or "microsoft/trocr-base-handwritten"
            self.proc = TrOCRProcessor.from_pretrained(self.name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.name)
            self.model.eval()
            import torch
            torch.set_num_threads(threads or os.cpu_count() or 1)
            self.no_grad = torch.no_grad
        elif kind == "ollama":
            self.name = model or "qwen2.5vl:3b"
            self.url = (url or "http://127.0.0.1:11434").rstrip("/")
            self.timeout, self.threads = timeout, threads
            self._ask("Reply with: ok", None, warm=True)
        else:
            raise ValueError(kind)

    def __call__(self, crop):
        _READERS[self.kind] = self
        self.calls += 1
        if self.kind == "tesseract":
            for psm in (7, 8, 13):
                t = self.pt.image_to_string(crop, config=self.cfg % psm).strip()
                t = t.replace("\n", " ").strip()
                if parse_length(t) is not None:
                    return t
            return ""
        if self.kind == "trocr":
            from PIL import Image
            im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))
            with self.no_grad():
                ids = self.model.generate(
                    self.proc(images=im, return_tensors="pt").pixel_values, max_new_tokens=16)
            return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        img = crop
        if img.shape[1] > 512:  # every pixel becomes vision tokens
            f = 512.0 / img.shape[1]
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return self._ask(PROMPT, img)

    def _ask(self, prompt, img, warm=False):
        import base64
        import urllib.request
        opts = {"temperature": 0, "num_predict": 24}
        if self.threads:
            opts["num_thread"] = self.threads
        body = {"model": self.name, "stream": False, "prompt": prompt,
                "keep_alive": "15m", "options": opts}
        if img is not None:
            body["images"] = [base64.b64encode(cv2.imencode(".png", img)[1]).decode()]
        req = urllib.request.Request(self.url + "/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r).get("response", "").strip().replace("\n", " ")
        except Exception as exc:
            if warm:
                raise SystemExit("ollama did not answer at %s: %s\n"
                                 "Is it running, and is the model pulled?" % (self.url, exc))
            return ""


def crop_segment(gray, chain, a, b, ppf):
    band = max(10, int(0.8 * ppf))
    ins = max(2, int((b - a) * 0.06))
    y0, y1 = max(0, chain["y"] - band), min(gray.shape[0], chain["y"] + band)
    if b - a - 2 * ins < 6:
        return None
    return gray[y0:y1, a + ins:b - ins]


DEBUG = {"on": False, "dir": None, "n": 0}


# ANCHOR BEFORE 20261011_142410
def read_crop(crop, ppf, reader, axis="h", seen=None):
    # ANCHOR AFTER 20261011_142410
    """Try a crop with the rules erased, then as it is.

    A vertical chain is read in the rotated view, where the text is upright if
    it was drawn reading bottom to top, which is the usual convention. Some
    sheets letter it the other way, so an upside-down pass is tried as well
    before giving up. Each attempt is cheap for Tesseract and the loop stops at
    the first reading that parses.
    """
    # Tight first: a long segment has the same problem the overall line has, with
    # the number occupying a fraction of the strip. Readers scale their input to a
    # fixed size, so the surrounding blank is paid for in resolution. On a short
    # segment the tight crop is almost the whole crop, so nothing is lost.
    views = []
    # ANCHOR BEFORE 20261011_151208
    # Every run of writing on the strip, largest first: the largest is not always the
    # dimension. A circled door number beside 19'0" won on the sample sheet.
    runs = text_runs(crop, ppf)
    views += [("tight" if k == 0 else "run%d" % (k + 1), r) for k, r in enumerate(runs[:3])]
    # ANCHOR AFTER 20261011_151208
    views += [("clean", erase_rules(crop, ppf)), ("raw", crop)]
    if axis == "v":
        views += [(tag + "+180", cv2.rotate(v, cv2.ROTATE_180)) for tag, v in list(views)]
    tried = []
    for tag, v in views:
        img = prep(v)
        t = reader(img)
        tried.append("%s=%r" % (tag, t))
        # ANCHOR BEFORE 20261011_142411
        # Exactly what the reader was shown, and what it said, so a failure can be
        # looked at rather than guessed at.
        if seen is not None:
            seen.append((tag, img, t))
        # ANCHOR AFTER 20261011_142411
        if parse_length(t) is not None:
            if DEBUG["on"]:
                print("        %s -> %r" % (tag, t))
            return t
        if DEBUG["on"] and DEBUG["dir"]:
            DEBUG["n"] += 1
            cv2.imwrite(str(Path(DEBUG["dir"]) / ("crop%04d_%s.png" % (DEBUG["n"], tag))), img)
    if DEBUG["on"]:
        print("        nothing parsed from: %s" % "  ".join(tried))
    return ""


def tight_text_crop(band, ppf, pad_ft=0.3):
    """Crop a band down to the writing on it.

    An overall dimension line runs the width of the building with one number
    written in the middle of it. Handing a reader that whole strip fails for a
    mechanical reason: every reader scales its input to a fixed size, so a
    number occupying 3% of a 2800 px strip arrives a few pixels tall. Find the
    ink that is the right size to be text, keep the biggest run of it, and crop
    to that. Returns None when there is nothing text-shaped on the line.
    """
    # ANCHOR BEFORE 20261011_151209
    runs = text_runs(band, ppf, pad_ft)
    return runs[0] if runs else None


def text_runs(band, ppf, pad_ft=0.3):
    """Every run of text-sized ink on a band, cropped, largest first."""
    # ANCHOR AFTER 20261011_151209
    ink = (erase_rules(band, ppf) < 160).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    lo, hi = max(2, int(0.12 * ppf)), max(6, int(1.2 * ppf))
    marks = [(st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH],
              st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT])
             for i in range(1, n)
             if lo <= st[i, cv2.CC_STAT_HEIGHT] <= hi and st[i, cv2.CC_STAT_WIDTH] <= hi * 2]
    if not marks:
        return []
    marks.sort()
    runs, gap = [[marks[0]]], 0.8 * ppf     # a space wider than this ends the number
    for m in marks[1:]:
        if m[0] - runs[-1][-1][1] <= gap:
            runs[-1].append(m)
        else:
            runs.append([m])
    # ANCHOR BEFORE 20261011_151210
    pad = max(3, int(pad_ft * ppf))
    out = []
    for run in sorted(runs, key=lambda r: -sum(m[1] - m[0] for m in r)):
        x0 = max(0, min(m[0] for m in run) - pad)
        x1 = min(band.shape[1], max(m[1] for m in run) + pad)
        y0 = max(0, min(m[2] for m in run) - pad)
        y1 = min(band.shape[0], max(m[3] for m in run) + pad)
        if x1 - x0 >= 6 and y1 - y0 >= 6:
            out.append(band[y0:y1, x0:x1])
    return out
    # ANCHOR AFTER 20261011_151210


def read_overall(gray, chain, ppf, reader, max_tries=0):
    """The chain's printed total, from the separate line it is written on.

    Without it nothing can be solved. Returning None is correct and honest:
    deriving the total from the measured span and the manifest scale would let
    the scale back in through the side door and defeat the whole check.
    """
    cands = chain.get("overall_candidates", [])
    if max_tries:
        cands = cands[:max_tries]
    for o in cands:
        c = crop_segment(gray, o, o["ticks"][0], o["ticks"][-1], ppf)
        if c is None:
            continue
        # Crop to the writing first. The line is the width of the building and
        # the number is a small part of it, so the full strip reads as nothing.
        tight = tight_text_crop(c, ppf)
        if tight is None:
            continue
        if max_tries:
            t = reader(prep(tight))
        else:
            t = read_crop(tight, ppf, reader, chain["axis"])
        ft = parse_length(t)
        if DEBUG["on"]:
            print("      overall candidate at %d -> %r" % (o["y"], t))
        # it is the total, so it cannot be shorter than the run it spans
        if ft and ft >= span_px(chain) / ppf * 0.9:
            return ft
    return None


def needs_expensive(measured, values):
    """Which unresolved segments are worth an expensive reading.

    A gap that measures the same as another gap will be solved by arithmetic,
    so spending two minutes on it buys nothing. A lone gap will not be.
    """
    gaps = [i for i, v in enumerate(values) if v is None]
    singles = [g[0] for g in group_gaps(gaps, measured) if len(g) == 1]
    return sorted(singles, key=lambda i: -measured[i])


# ANCHOR BEFORE 20261011_145542
def read_chain_raw(gray, chain, ppf, cheap, progress=None, crops_dir=None):
    # ANCHOR AFTER 20261011_145542
    """Read every segment of one chain. No judgement, just what came back.

    Reading is the expensive step and validation is free, so they are separated:
    the text is collected once and can then be checked against any scale. That is
    what lets the scale be worked out from the drawing instead of trusted from the
    manifest, without reading anything twice.

    ppf is used only to size the crops, so a few percent out does no harm here.
    Each segment keeps what every view it was shown returned; with crops_dir, the
    images of a segment that produced nothing are saved there.
    """
    ticks = chain["ticks"]
    bounds = list(zip(ticks, ticks[1:]))
    out = []
    for a, b in bounds:
        # ANCHOR BEFORE 20261011_180854
        c = remove_chain_line(crop_segment(gray, chain, a, b, ppf), chain, ppf)
        # ANCHOR AFTER 20261011_180854
        text = ""
        # ANCHOR BEFORE 20261011_145543
        seen, crops = [], []
        # ANCHOR BEFORE 20261011_185021
        # A crop with nothing text-shaped on it is not shown to the reader, here or in the
        # wider read below: given blank paper and the last specks of the line, it answered
        # 1' on a 0.84 ft stub, and that was believed.
        if c is not None and text_runs(c, ppf):
        # ANCHOR AFTER 20261011_185021
            if DEBUG["on"]:
                print("      seg%02d  %5.2f ft measured" % (len(out), (b - a) / ppf))
            text = read_crop(c, ppf, cheap, chain["axis"], seen=seen)
        # ANCHOR BEFORE 20261011_151720
        # A joint can cut through the number it carries (19'4" came out as "19" on one
        # piece and the rest on the next), so a piece that read nothing is read once
        # more across itself and its two neighbours.
        if not text and c is not None:
            k = len(out)
            wa, wb = ticks[max(0, k - 1)], ticks[min(len(ticks) - 1, k + 2)]
            # ANCHOR BEFORE 20261011_180855
            wide = (remove_chain_line(crop_segment(gray, chain, wa, wb, ppf), chain, ppf)
                    if (wa, wb) != (a, b) else None)
            # ANCHOR AFTER 20261011_180855
            # ANCHOR BEFORE 20261011_185023
            if wide is not None and text_runs(wide, ppf):
            # ANCHOR AFTER 20261011_185023
                wseen = []
                text = read_crop(wide, ppf, cheap, chain["axis"], seen=wseen)
                seen += [("wide-" + tag, img, t) for tag, img, t in wseen]
        # ANCHOR AFTER 20261011_151720
        # ANCHOR BEFORE 20261011_214905
        # A piece that still read nothing is read once more with the line's ragged edge
        # cleared as well. Only then: a piece that read anything is never re-read, so this
        # can add a reading and cannot change one.
        if not text and c is not None:
            k = len(out)
            wa, wb = ticks[max(0, k - 1)], ticks[min(len(ticks) - 1, k + 2)]
            for tag, (ra, rb) in (("retry", (a, b)), ("retry-wide", (wa, wb))):
                if text or (tag == "retry-wide" and (wa, wb) == (a, b)):
                    continue
                rc = remove_chain_line(crop_segment(gray, chain, ra, rb, ppf), chain, ppf, edge=True)
                if rc is not None and text_runs(rc, ppf):
                    rseen = []
                    text = read_crop(rc, ppf, cheap, chain["axis"], seen=rseen)
                    seen += [(tag + "-" + v, img, t) for v, img, t in rseen]
        # ANCHOR AFTER 20261011_214905
        if crops_dir is not None and not text and seen:
            Path(crops_dir).mkdir(parents=True, exist_ok=True)
            for tag, img, _ in seen:
                name = "%s_%d_seg%02d_%d-%d_%s.png" % (chain["axis"], chain["y"], len(out),
                                                      a, b, tag.replace("+", "_"))
                cv2.imwrite(str(Path(crops_dir) / name), img)
                crops.append(name)
        out.append({"a": int(a), "b": int(b), "length_px": int(b - a),
                    "text": text, "reader": cheap.kind if text else None,
                    "views": [{"view": tag, "text": t} for tag, _, t in seen],
                    "crops": crops, "cropped": c is not None})
        # ANCHOR AFTER 20261011_145543
        if progress:
            progress.tick()
    return out


def chain_scale_votes(chains_raw, min_ft=None):
    """Every (pixels, printed feet) pair the readings offer, as a pixels-per-foot vote.

    Only longer segments vote. A draughtsman crams the short ones, so a 2 ft segment
    is not drawn to scale and would drag the estimate around.
    """
    min_ft = rule("reading.scale_min_segment_ft") if min_ft is None else min_ft
    votes = []
    for raws in chains_raw:
        for seg in raws:
            ft = parse_length(seg["text"])
            if ft and ft >= min_ft and seg["length_px"] > 0:
                votes.append(seg["length_px"] / ft)
    return votes


def scale_from_votes(votes, fallback, max_drift=None, min_votes=None):
    """The scale the drawing implies, or the fallback when it cannot be trusted.

    The median resists a misread. Votes far from the fallback are dropped first,
    because a badly misread number produces a wild ratio, not a slightly wrong one;
    a genuine manifest error is a few percent, not a factor of two.
    """
    max_drift = rule("reading.scale_max_drift") if max_drift is None else max_drift
    min_votes = rule("reading.scale_min_votes") if min_votes is None else min_votes
    near = sorted(v for v in votes if abs(v - fallback) / fallback <= max_drift)
    if len(near) < min_votes:
        return None, len(near)
    return near[len(near) // 2], len(near)


# ANCHOR BEFORE 20261011_134728
def choose_scale(manifest_ppf, derived, used=0, use_drawing=False):
    """The scale a run measures with, and the note saying what the drawing implied.

    The drawing's own scale is reported and, unless use_drawing is set, never used. On
    the sample sheet it came out 70.50, 72.57 and 73.14 px/ft in three runs, all low,
    against 74.90 measured off a printed 43'4" run: one fixed property of one image
    giving three answers is a misread, not a measurement. Returns (ppf, note or None).
    """
    if not derived or abs(derived - manifest_ppf) / manifest_ppf <= 0.02:
        return manifest_ppf, None
    said = ("the drawing's own dimensions imply %.2f px/ft from %d reading(s), against "
            "%.2f in manifest.csv (%.1f%% apart)"
            % (derived, used, manifest_ppf, abs(derived - manifest_ppf) / manifest_ppf * 100))
    if use_drawing:
        return derived, said + "; the drawing was used because --use-drawing-scale is set"
    return manifest_ppf, said + ("; the manifest was used. The derived scale is reported, "
                                 "not used; pass --use-drawing-scale to use it")
# ANCHOR AFTER 20261011_134728


def assemble(chain, raws, ppf, expensive=None, max_expensive=4, gray=None):
    """Judge what was read, at a given scale, and fill the gaps.

    Separated from reading so it can be redone at a corrected scale for free.
    """
    measured = [s["length_px"] / ppf for s in raws]
    stated = [parse_length(s["text"]) for s in raws]
    readers = [s["reader"] for s in raws]
    raw_text = [s["text"] for s in raws]

    believed = sum(1 for s, m in zip(stated, measured) if believe(s, m))
    overall = None
    if believed >= MIN_READINGS and gray is not None:
        overall = read_overall(gray, chain, ppf, expensive or _overall_reader(raws))
    values, info = solve_chain(measured, stated, overall)

    if expensive is not None and gray is not None and any(v is None for v in values):
        for i in needs_expensive(measured, values)[:max_expensive]:
            c = crop_segment(gray, chain, raws[i]["a"], raws[i]["b"], ppf=ppf)
            if c is None:
                continue
            t = read_crop(c, ppf, expensive, chain["axis"])
            ft = parse_length(t)
            if ft is not None:
                raw_text[i], stated[i], readers[i] = t, ft, expensive.kind
        if overall is None:
            overall = read_overall(gray, chain, ppf, expensive, max_tries=1)
        values, info = solve_chain(measured, stated, overall)

    segments = []
    for i, s in enumerate(raws):
        src = info["source"][i]
        segments.append({
            "a": s["a"], "b": s["b"], "length_px": s["length_px"],
            "measured_ft": round(measured[i], 2),
            "value_ft": None if values[i] is None else round(values[i], 4),
            "printed": fmt(values[i]),
            "source": src,
            "reader": readers[i] if src == "read" else None,
            "raw_text": raw_text[i] if src == "read" else None,
            "stated_ft": stated[i] if src == "read" else None,
        })
    return {
        "axis": chain["axis"], "y": chain["y"], "x0": chain["x0"], "x1": chain["x1"],
        "span_ft": round(span_px(chain) / ppf, 2),
        "overall_ft": None if overall is None else round(overall, 4),
        "overall_printed": fmt(overall),
        "overall_source": "read" if overall is not None else None,
        "readings": sum(1 for s in segments if s["source"] == "read"),
        "solved": sum(1 for s in segments if s["source"] == "solved"),
        "unknown": sum(1 for s in segments if s["source"] is None),
        "segments": segments,
        "rejected": info["rejected"],
        "warnings": info["warnings"],
    }


def _overall_reader(raws):
    """Whichever reader produced this chain's text, reused for its overall line."""
    for s in raws:
        if s["reader"]:
            return _READERS.get(s["reader"])
    return None


_READERS = {}


# ANCHOR BEFORE 20261011_145109
def chain_record(chain, raws, result, ppf, height):
    """What became of one chain candidate and each of its segments, kept or dropped.

    A dropped chain used to leave nothing behind, so a line whose numerals the reader
    missed looked the same as a line with no numerals at all. Positions are sheet
    pixels; for a vertical chain lo/hi run down the sheet.
    """
    segs = []
    for s, judged in zip(raws, result["segments"]):
        lo, hi = (s["a"], s["b"]) if chain["axis"] == "h" else (
            height - 1 - s["b"], height - 1 - s["a"])
        stated = parse_length(s["text"])
        segs.append({"lo": int(lo), "hi": int(hi), "drawn_ft": round(s["length_px"] / ppf, 2),
                     "raw_text": s["text"], "stated_ft": stated,
                     "outcome": ("believed" if judged["source"] == "read" else
                                 "solved" if judged["source"] == "solved" else
                                 "disagreed" if stated is not None else
                                 "no crop" if not s.get("cropped", True) else "unreadable"),
                     "views": s.get("views", []), "crops": s.get("crops", [])})
    return {"axis": chain["axis"], "cross": int(chain["y"]), "x0": int(chain["x0"]),
            "x1": int(chain["x1"]), "ticks": len(chain["ticks"]),
            "kept": is_real_chain(result), "readings": result["readings"], "segments": segs}
# ANCHOR AFTER 20261011_145109


# ANCHOR BEFORE 20261011_150218
# ANCHOR BEFORE 20261011_184001
def readings_as_singles(chain, raws, result, ppf, height, gray=None):
    # ANCHOR AFTER 20261011_184001
    """Readings on a line that did not survive as a chain, kept as lone dimensions.

    An interior dimension line is crossed by beam lines, door swings and other
    dimension lines, and each crossing registers as a joint. The printed number then
    sits on a short piece of the run it measures and disagrees with that piece. The
    reading says which joints are real: the run it measures is the contiguous stretch
    of pieces, around the one it sits on, whose drawn length it agrees with. Only a
    clear best fit is kept; when the runner-up fits nearly as well, the reading is
    left out rather than placed on a guess. A chain that survives keeps its readings
    as segments and gives nothing here.
    """
    if is_real_chain(result):
        return []
    margin = rule("reading.merge_margin_rel")
    out = []
    # ANCHOR BEFORE 20261011_184002
    # The draughtsman's own arrowheads say which run a number measures. Closest fit
    # assumes the drawing is to scale, and 18'7" at the bottom of BED ROOM C is drawn
    # 0.46 ft short, which makes the run across the partition above it the closer fit.
    # A run with an arrowhead at each end, each pointing out of the run, is the one
    # meant; closest fit decides only when no fitting run is marked that way.
    win = max(2, int(rule("detection.arrow_stroke_max_ft") * ppf))
    tips = {}

    def marked_at(x, pointing):
        if (x, pointing) not in tips:
            tips[(x, pointing)] = gray is not None and any(
                p == pointing for _, p in arrow_tips(gray, chain, ppf, x - win, x + win))
        return tips[(x, pointing)]
    # ANCHOR AFTER 20261011_184002
    for i, s in enumerate(raws):
        v = parse_length(s["text"])
        if v is None:
            continue
        fits = []
        for a in range(i + 1):
            for b in range(i, len(raws)):
                drawn = (raws[b]["b"] - raws[a]["a"]) / ppf
                if believe(v, drawn):
                    fits.append((abs(v - drawn), a, b))
        if not fits:
            continue
        fits.sort()
        # ANCHOR BEFORE 20261011_184003
        marked = [f for f in fits if marked_at(raws[f[1]]["a"], -1) and marked_at(raws[f[2]]["b"], 1)]
        if marked:
            fits, placed_by = marked, "end marks"
        else:
            placed_by = "closest fit"
            if len(fits) > 1 and fits[0][0] > margin * fits[1][0]:
                continue
        # ANCHOR AFTER 20261011_184003
        _, a, b = fits[0]
        start, end = raws[a]["a"], raws[b]["b"]
        lo, hi = (start, end) if chain["axis"] == "h" else (height - 1 - end, height - 1 - start)
        # A number read again on a neighbouring piece, across the joint that split it,
        # lands on the same run: one dimension, kept once.
        if any(o["ft"] == v and o["lo"] == lo and o["hi"] == hi for o in out):
            continue
        out.append({"ft": v, "axis": chain["axis"], "lo": int(lo), "hi": int(hi),
                    "cross": int(chain["y"]), "source": "read", "reader": s.get("reader"),
                    "raw_text": s["text"], "measured_ft": round((end - start) / ppf, 2),
                    "pieces": b - a + 1,
                    # ANCHOR BEFORE 20261011_184004
                    "placed_by": placed_by,
                    # ANCHOR AFTER 20261011_184004
                    "chain": "%s dimension at %d" % (chain["axis"], chain["y"])})
    return out
# ANCHOR AFTER 20261011_150218


def is_real_chain(result):
    """Stair treads and hatching make convincing joints and no numbers."""
    return result["readings"] >= MIN_READINGS


# ------------------------------------------------------------------ drawing

def overlay(bgr, chains, ppf):
    vis = bgr.copy()
    h = bgr.shape[0]
    for c in chains:
        hor = c["axis"] == "h"
        for s in c["segments"]:
            p0 = (s["a"], c["y"]) if hor else unrotate(s["a"], c["y"], h)
            p1 = (s["b"], c["y"]) if hor else unrotate(s["b"], c["y"], h)
            colour = {"read": (0, 150, 0), "solved": (200, 120, 0)}.get(s["source"], (0, 0, 220))
            cv2.line(vis, p0, p1, colour, max(2, int(ppf / 20)))
            cv2.circle(vis, p0, max(3, int(ppf / 12)), (120, 120, 120), -1)
            if s["value_ft"]:
                at = (p0[0], p0[1] - int(ppf * 0.4)) if hor else (p0[0] + int(ppf * 0.3), p0[1])
                cv2.putText(vis, s["printed"], at, cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.4, ppf / 90), colour, max(1, int(ppf / 40)), cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------- main

def load_manifest(path="manifest.csv"):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            name = (r.get("plan") or "").strip()
            if not name:
                continue
            try:
                ppf = float((r.get("pixels_per_foot") or "").strip())
            except ValueError:
                sys.exit("manifest.csv: pixels_per_foot for %s is not a number: %r\n"
                         "Set a real scale, or run: python manifest-update.py plans/%s"
                         % (name, r.get("pixels_per_foot"), name))
            if not 0.5 <= ppf <= 500:
                sys.exit("manifest.csv: pixels_per_foot for %s is %g, outside 0.5-500" % (name, ppf))
            rows[name] = ppf
    return rows


# ANCHOR BEFORE 20261012_033438
def lone_summary(believed, tried, why):
    """The lone-dimension status line.

    A reading placed on a chain line that was not kept as a chain is a lone dimension
    too, but no lone line was tried for it, so it has no outcome among the lines tried.
    Counted together, 12 believed sat beside outcomes that were all failures.
    """
    own = why.get("believed", 0)
    text = "%d lone dimension(s) believed, of %d line(s) tried%s" % (
        own, tried, ("   " + ", ".join("%s %d" % kv for kv in sorted(why.items()))) if why else "")
    if believed > own:
        text += "; %d more believed on chain lines not kept as chains" % (believed - own)
    return text
# ANCHOR AFTER 20261012_033438


def run_plan(name, ppf, args, cheap, expensive, group):
    path = PLANS / name
    bgr = cv2.imread(str(path))
    if bgr is None:
        print("  cannot read %s, skipping" % path)
        return None
    if bgr.shape[0] * bgr.shape[1] > MAX_PIXELS:
        sys.exit("%s is %dx%d, over the %d megapixel limit"
                 % (name, bgr.shape[1], bgr.shape[0], MAX_PIXELS // 1_000_000))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t_plan = time.time()
    trace = Trace(name, params={"pixels_per_foot_given": ppf, "axis": args.axis,
                                "cheap_reader": cheap.kind,
                                "expensive_reader": None if expensive is None else expensive.kind},
                  group=group)
    with trace.stage("detect chains") as st:
        found, singles = detect(gray, ppf, args.axis)
        st["candidates"] = len(found)
        st["single_candidates"] = len(singles)

    chosen = by_axis(found, args.max_chains)
    bar = Progress(sum(max(0, len(c["ticks"]) - 1) for c in chosen), name + " segments",
                   quiet=DEBUG["on"])
    raws = []
    with trace.stage("read chains") as st:
        for c in chosen:
            if DEBUG["on"]:
                print("    %s chain at %d, %d joints, span %.1f ft"
                      % (c["axis"], c["y"], len(c["ticks"]), span_px(c) / ppf))
            src = gray if c["axis"] == "h" else vertical_view(gray)
            # ANCHOR BEFORE 20261011_145110
            raws.append(read_chain_raw(src, c, ppf, cheap, progress=bar,
                                       crops_dir=getattr(args, "crops_dir", None)))
            # ANCHOR AFTER 20261011_145110
        st["chains"] = len(chosen)
    bar.finish()

    # What scale does the drawing itself imply? Worked out from the readings before
    # any of them are judged, so a wrong manifest cannot reject the evidence that
    # would have corrected it.
    # Lone dimension lines: one read each, and kept only if the line measures what
    # it says. Done before the scale is settled so their readings can vote on it.
    singles = singles[:args.max_singles]
    lone, single_why = [], {}
    # ANCHOR BEFORE 20261011_142353
    single_records = []
    # ANCHOR AFTER 20261011_142353
    if singles:
        sbar = Progress(len(singles), name + " single dimensions", quiet=DEBUG["on"])
        why = {}
        for l in singles:
            src = gray if l["axis"] == "h" else vertical_view(gray)
            # ANCHOR BEFORE 20261011_142354
            got = read_single(src, l, ppf, cheap, l["axis"], gray.shape[0], why=why,
                              record=single_records)
            # ANCHOR AFTER 20261011_142354
            if got:
                lone.append(got)
            sbar.tick()
        sbar.finish()
        single_why = why

    derived, used = scale_from_votes(chain_scale_votes(raws), ppf)
    notes = []
    # ANCHOR BEFORE 20261011_134717
    work_ppf, note = choose_scale(ppf, derived, used,
                                  use_drawing=args.use_drawing_scale and not args.trust_manifest)
    if note:
        notes.append(note)
    # ANCHOR AFTER 20261011_134717

    results, dropped = [], 0
    # ANCHOR BEFORE 20261011_145111
    chain_records = []
    # ANCHOR AFTER 20261011_145111
    for c, r in zip(chosen, raws):
        src = gray if c["axis"] == "h" else vertical_view(gray)
        out = assemble(c, r, work_ppf, expensive, args.max_expensive, gray=src)
        # ANCHOR BEFORE 20261011_145112
        chain_records.append(chain_record(c, r, out, work_ppf, gray.shape[0]))
        # ANCHOR BEFORE 20261011_184005
        lone.extend(readings_as_singles(c, r, out, work_ppf, gray.shape[0], gray=src))
        # ANCHOR AFTER 20261011_184005
        # ANCHOR AFTER 20261011_145112
        if is_real_chain(out):
            results.append(out)
        else:
            dropped += 1

    if dropped:
        notes.append("%d line(s) looked like chains but carried no readable numbers" % dropped)

    out = {"plan": name, "pixels_per_foot": work_ppf, "pixels_per_foot_in_manifest": ppf,
           "single_dimensions": lone, "single_outcomes": single_why,
           # ANCHOR BEFORE 20261011_142355
           # every lone candidate and what became of it, not only the counts
           "single_candidates": single_records,
           # ANCHOR BEFORE 20261011_145113
           # every chain read, including those dropped for carrying no number
           "chain_candidates": chain_records,
           # ANCHOR AFTER 20261011_145113
           # ANCHOR AFTER 20261011_142355
           "pixels_per_foot_from_drawing": None if derived is None else round(derived, 2),
           "readings_used_for_scale": used, "chains": results, "notes": notes}
    trace.data["chains"] = results
    for n in notes:
        trace.warn(n)
    # A diagnostic run is partial by design, and must not replace a full one. Writing
    # it beside the real file keeps a 25-minute result from being destroyed by a
    # four-minute look at why something failed.
    suffix = "_dimlines_debug" if DEBUG["on"] else "_dimlines"
    OUT.mkdir(parents=True, exist_ok=True)
    tfile = OUT.parent / ("trace_%s.json" % trace.id)
    trace.write(tfile)
    out["trace_id"] = trace.id
    out["trace_file"] = tfile.name
    stamp(out, "dimlines", group)

    cv2.imwrite(str(OUT / (Path(name).stem + suffix + "_overlay.png")),
                overlay(bgr, results, ppf))
    with open(OUT / (Path(name).stem + suffix + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    if DEBUG["on"]:
        print("  debug run: wrote %s%s.json, left the real one alone"
              % (Path(name).stem, suffix))

    took = time.time() - t_plan
    out["seconds"] = round(took, 1)
    out["segments_read"] = sum(len(r["segments"]) for r in results)
    # ANCHOR BEFORE 20261012_033437
    print("  " + lone_summary(len(lone), len(singles), single_why))
    # ANCHOR AFTER 20261012_033437
    print("  %d chain(s): %d read, %d solved, %d unknown; %d expensive call(s)"
          % (len(results),
             sum(r["readings"] for r in results),
             sum(r["solved"] for r in results),
             sum(r["unknown"] for r in results),
             0 if expensive is None else expensive.calls))
    print("  took %s for %d segment(s) across %d candidate line(s)  (%s)"
          % (Progress._clock(took), bar.total, len(chosen),
             time.strftime("%H:%M:%S", time.localtime())))
    for n in notes:
        print("  note: %s" % n)
    for r in results:
        print("    %s chain at %d, %d segments, overall %s%s"
              % (r["axis"], r["y"], len(r["segments"]), r["overall_printed"],
                 "" if r["overall_ft"] else " (not read, gaps stay unknown)"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default="manifest.csv", help="the plan list and its scales")
    ap.add_argument("--only", default="", help="one plan name from the manifest")
    ap.add_argument("--axis", default="both", choices=("both", "h", "v"),
                    help="which chains to look for: across the sheet, down it, or both. Room widths come from the horizontal ones and depths from the vertical.")
    ap.add_argument("--reader", default="tesseract", choices=("tesseract", "trocr"),
                    help="the cheap reader, run over every segment")
    ap.add_argument("--expensive", default="", choices=("", "ollama", "trocr"),
                    help="optional second reader, used only where the first failed "
                         "and arithmetic cannot fill the gap")
    ap.add_argument("--model", default=None,
                    help="model name for trocr or ollama, e.g. qwen2.5vl:3b")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--threads", type=int, default=0,
                    help="worker threads for the reader; defaults to your core count")
    ap.add_argument("--timeout", type=int, default=600,
                    help="seconds to wait for one ollama reading")
    ap.add_argument("--max-chains", type=int, default=16,
                    help="longest chains to read per axis, not in total")
    ap.add_argument("--max-singles", type=int, default=60,
                    help="most lone dimension lines to read per plan")
    ap.add_argument("--max-expensive", type=int, default=4,
                    help="most expensive readings per chain")
    # ANCHOR BEFORE 20261011_134729
    ap.add_argument("--trust-manifest", action="store_true",
                    help="keep the manifest scale even when the drawing disagrees. This is "
                         "the default; the flag is kept so existing scripts still run, and it "
                         "wins over --use-drawing-scale.")
    ap.add_argument("--use-drawing-scale", action="store_true",
                    help="measure with the scale the drawing's own readings imply instead of "
                         "the manifest's. Off by default: the derived scale is always reported, "
                         "but on the sample sheet it has been 2 to 6 percent low in every run.")
    # ANCHOR AFTER 20261011_134729
    ap.add_argument("--group", default=None,
                    help="group id to stamp on this run, so several layers share one")
    ap.add_argument("--debug", action="store_true",
                    help="print every reader attempt per segment and save the crops it "
                         "could not parse, so a silent chain can be diagnosed")
    ap.add_argument("--debug-dir", default="dimlines_debug",
                    help="where --debug writes the crops that would not parse")
    # ANCHOR BEFORE 20261011_142404
    ap.add_argument("--singles-only", action="store_true",
                    help="read the lone dimension lines only, never the chain segments, and "
                         "write <plan>_singles.json and every failed crop under --singles-dir. "
                         "Nothing in the dimension layer's own output is touched")
    ap.add_argument("--singles-dir", default=None,
                    help="where --singles-only writes; required with it")
    # ANCHOR AFTER 20261011_142404
    # ANCHOR BEFORE 20261011_145124
    ap.add_argument("--chains-only", default=None, metavar="LIST",
                    help="read only these chain candidates, comma separated as axis:position "
                         "or axis:position:along, e.g. v:863,h:4834:6000 (along is a point the "
                         "line must span, for two lines on one row). No lone pass, no other "
                         "chain; writes <plan>_chains.json and failed crops under --chains-dir")
    ap.add_argument("--chains-dir", default=None,
                    help="where --chains-only writes; required with it")
    ap.add_argument("--crops-dir", default=None,
                    help="also save the images of every chain segment that produced nothing")
    # ANCHOR AFTER 20261011_145124
    # ANCHOR BEFORE 20261011_153403
    ap.add_argument("--lines-only", action="store_true",
                    help="write which detected lines carry a believed reading in this plan's "
                         "dimension file, as <plan>%s.json. No reader runs, no trace is "
                         "written, and the dimension file is only read" % LINES_SUFFIX)
    ap.add_argument("--lines-dir", default=str(OUT),
                    help="where --lines-only writes its file")
    ap.add_argument("--dims-dir", default=str(OUT),
                    help="where --lines-only reads <plan>_dimlines.json from")
    # ANCHOR AFTER 20261011_153403
    a = ap.parse_args(argv)

    if a.debug:
        DEBUG["on"] = True
        DEBUG["dir"] = a.debug_dir
        Path(a.debug_dir).mkdir(parents=True, exist_ok=True)
        print("debug on: unparsed crops go to %s/" % a.debug_dir)
    rows = load_manifest(a.manifest)
    if a.only:
        rows = {k: v for k, v in rows.items() if k == a.only}
        if not rows:
            sys.exit("%s is not in %s" % (a.only, a.manifest))
    group = new_group(a.group)
    print("dimlines version %s, group id %s" % (rt_version(), group))
    # ANCHOR BEFORE 20261011_153404
    if a.lines_only:
        # Geometry only. Returning before any reader is built is what keeps this in
        # seconds, and before the summary write is what keeps a reading run intact.
        Path(a.lines_dir).mkdir(parents=True, exist_ok=True)
        for name, ppf in rows.items():
            t = time.time()
            gray = cv2.imread(str(PLANS / name), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print("  cannot read %s, skipping" % (PLANS / name))
                continue
            # Read directly: the rooms-layer helpers are defined below main().
            dim_file = Path(a.dims_dir) / (Path(name).stem + "_dimlines.json")
            try:
                dim_data = json.loads(dim_file.read_text()) if dim_file.exists() else None
            except ValueError:
                dim_data = None
            if not dim_data:
                print("  %s: no dimension file, so no line is proven and none is erased" % name)
            geo = line_geometry(gray, ppf, dim_data)
            geo.update(plan=name, pixels_per_foot=ppf, image_px=[gray.shape[1], gray.shape[0]],
                       dimension_trace=(dim_data or {}).get("trace_id"),
                       seconds=round(time.time() - t, 2))
            stamp(geo, "dimlines-lines", group)
            path = Path(a.lines_dir) / (Path(name).stem + LINES_SUFFIX + ".json")
            with open(path, "w") as f:
                json.dump(geo, f, indent=1)
            print("%s: %d of %d candidate line(s) carry a believed reading and will be erased; "
                  "%d proven line(s) matched no candidate; %.2f s -> %s"
                  % (name, len(geo["stretches"]), geo["candidates"], len(geo["unmatched"]),
                     geo["seconds"], path))
        return
    # ANCHOR AFTER 20261011_153404
    cheap = Reader(a.reader, a.model if a.reader == "trocr" else None, threads=a.threads)
    expensive = None
    if a.expensive:
        expensive = Reader(a.expensive, a.model, a.ollama_url, a.timeout, a.threads)
    print("cheap reader %s, expensive reader %s"
          % (cheap.kind, expensive.kind if expensive else "none"))
    # ANCHOR BEFORE 20261011_142405
    if a.singles_only:
        # The cheap loop: detection, then one read per lone line. Minutes, not the
        # half hour the chain segments take, so the reader can be worked on in turns.
        if not a.singles_dir:
            sys.exit("--singles-only needs --singles-dir, so it never writes the real output")
        out_dir = Path(a.singles_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, ppf in rows.items():
            t = time.time()
            gray = cv2.imread(str(PLANS / name), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print("  cannot read %s, skipping" % (PLANS / name))
                continue
            _, singles = detect(gray, ppf, a.axis)
            singles = singles[:a.max_singles]
            records, why, lone = [], {}, []
            for l in singles:
                src = gray if l["axis"] == "h" else vertical_view(gray)
                got = read_single(src, l, ppf, cheap, l["axis"], gray.shape[0], why=why,
                                  record=records, crops_dir=out_dir / "crops")
                if got:
                    lone.append(got)
            res = {"plan": name, "pixels_per_foot": ppf, "chains": [], "single_dimensions": lone,
                   "single_outcomes": why, "single_candidates": records,
                   "reader": cheap.kind, "seconds": round(time.time() - t, 1)}
            stamp(res, "dimlines-singles", group)
            path = out_dir / (Path(name).stem + "_singles.json")
            with open(path, "w") as f:
                json.dump(res, f, indent=1)
            print("%s: %d of %d lone dimension(s) believed (%s) in %.0f s -> %s"
                  % (name, len(lone), len(singles),
                     ", ".join("%s %d" % kv for kv in sorted(why.items())), res["seconds"], path))
        return
    # ANCHOR AFTER 20261011_142405
    # ANCHOR BEFORE 20261011_145125
    if a.chains_only:
        # Named chain candidates only, picked from every detected line before
        # duplicates are dropped: the de-duplication discards the second-floor lower
        # chain as a copy of a stair line, and that chain carries three room widths.
        if not a.chains_dir:
            sys.exit("--chains-only needs --chains-dir, so it never writes the real output")
        wanted = []
        for item in a.chains_only.split(","):
            parts = item.strip().split(":")
            wanted.append((parts[0], int(parts[1]), int(parts[2]) if len(parts) > 2 else None))
        out_dir = Path(a.chains_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, ppf in rows.items():
            t = time.time()
            gray = cv2.imread(str(PLANS / name), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print("  cannot read %s, skipping" % (PLANS / name))
                continue
            tol = max(3, int(rule("detection.line_max_thick_ft") * ppf))
            lines = {ax: [l for l in candidate_lines(gray if ax == "h" else vertical_view(gray), ppf)
                          if len(l["ticks"]) >= MIN_CHAIN_TICKS] for ax in ("h", "v")}
            results, records, missing, nseg, lone = [], [], [], 0, []
            for ax, pos, along in wanted:
                # vertical lines live in the rotated view: their row is the sheet x,
                # and a sheet y maps to rotated x = height - 1 - y
                at = None if along is None else (along if ax == "h" else gray.shape[0] - 1 - along)
                cands = [l for l in lines[ax] if abs(l["y"] - pos) <= tol
                         and (at is None or l["x0"] <= at <= l["x1"])]
                if not cands:
                    missing.append("%s:%d" % (ax, pos))
                    continue
                c = dict(min(cands, key=lambda l: (abs(l["y"] - pos), -len(l["ticks"]))), axis=ax)
                src = gray if ax == "h" else vertical_view(gray)
                # ANCHOR BEFORE 20261011_173130
                c = add_end_joints(src, c, ppf)
                # ANCHOR AFTER 20261011_173130
                raws = read_chain_raw(src, c, ppf, cheap, crops_dir=out_dir / "crops")
                nseg += len(raws)
                res = assemble(c, raws, ppf, None, gray=src)
                results.append(res)
                records.append(chain_record(c, raws, res, ppf, gray.shape[0]))
                # ANCHOR BEFORE 20261011_184006
                lone.extend(readings_as_singles(c, raws, res, ppf, gray.shape[0], gray=src))
                # ANCHOR AFTER 20261011_184006
            res_all = {"plan": name, "pixels_per_foot": ppf,
                       "chains": [r for r in results if is_real_chain(r)],
                       "single_dimensions": lone, "chain_candidates": records,
                       "missing": missing, "segments_read": nseg, "reader": cheap.kind,
                       "seconds": round(time.time() - t, 1)}
            stamp(res_all, "dimlines-chains", group)
            path = out_dir / (Path(name).stem + "_chains.json")
            with open(path, "w") as f:
                json.dump(res_all, f, indent=1)
            print("%s: %d chain(s), %d segment(s) read, %d believed on kept chains, %d kept as "
                  "lone dimensions, %d missing, %.0f s -> %s"
                  % (name, len(results), nseg,
                     sum(r["readings"] for r in results if is_real_chain(r)), len(lone),
                     len(missing), res_all["seconds"], path))
        return
    # ANCHOR AFTER 20261011_145125

    done = []
    for name, ppf in rows.items():
        print("\n%s  (%.2f px/ft)" % (name, ppf))
        r = run_plan(name, ppf, a, cheap, expensive, group)
        if r:
            done.append({"plan": name, "trace_id": r["trace_id"],
                         "chains": len(r["chains"]), "seconds": r.get("seconds"),
                         "segments": sum(len(c["segments"]) for c in r["chains"])})
    summary = {"group_id": group, "plans": done}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "dimlines.json", "w") as f:
        json.dump(summary, f, indent=1)
    total = sum(d.get("seconds") or 0 for d in done)
    print("\n%d plan(s) done in %s. Details in %s/"
          % (len(done), Progress._clock(total), OUT))


if __name__ == "__main__":
    main()


# ------------------------------------------------ consumed by the rooms layer

def chain_index(data, height):
    """Joint positions in sheet coordinates, with the length of each segment between them.

    This is what makes a chain useful to the rooms layer. A joint is a wall
    centreline, so a room whose edges land on two joints of the same chain has
    a width that is printed on the drawing rather than measured off pixels.

    An unsolved segment is kept as None rather than throwing the chain away. A
    hole only spoils the distances that span it: on a chain reading
    15'3" | 4" | 8'7" | 2" | 13'6", the two closet jambs are far too small for
    any reader, but the parlour either side of them is still measurable.

    Vertical chains were read in the rotated view, so their joints are mapped
    back and put in ascending sheet order.
    """
    idx = {"h": [], "v": []}
    for c in data.get("chains", []):
        vals = [s["value_ft"] for s in c["segments"]]
        if not vals or all(v is None for v in vals):
            continue
        ticks = [c["segments"][0]["a"]] + [s["b"] for s in c["segments"]]
        if c["axis"] == "h":
            pos = ticks
        else:
            pos = [height - 1 - t for t in ticks][::-1]
            vals = vals[::-1]
        idx[c["axis"]].append({"pos": pos, "seg": vals,
                               # ANCHOR BEFORE 20261011_142007
                               # where the chain sits across the axis it measures
                               "cross": int(c["y"]),
                               # ANCHOR AFTER 20261011_142007
                               "ref": "%s chain at %d" % (c["axis"], c["y"])})
    return idx


def dimension_points(data, height):
    """Every known length, with the stretch of sheet it actually spans.

    A draughtsman writes a dimension on the run it measures, so the segment covers
    the thing it describes. Keeping the whole span, not just the midpoint, is what
    lets a length be matched to the right room: the 15'7" under Bed Room C is drawn
    level with the hall beside it, and only its span shows it belongs to the bedroom.

    `lo`/`hi` are along the axis measured, `cross` is the perpendicular position.
    Vertical chains were read in the rotated view and are mapped back here.
    """
    # Lone dimension lines sit beside the chain segments: both are a length with a
    # stretch of sheet it covers, and nothing downstream needs to tell them apart.
    out = [dict(p) for p in data.get("single_dimensions", [])]
    for c in data.get("chains", []):
        for seg in c["segments"]:
            if seg["value_ft"] is None:
                continue
            if c["axis"] == "h":
                lo, hi, cross = seg["a"], seg["b"], c["y"]
            else:
                # rotated x runs opposite to sheet y
                lo, hi = height - 1 - seg["b"], height - 1 - seg["a"]
                cross = c["y"]
            out.append({"ft": seg["value_ft"], "axis": c["axis"],
                        "lo": int(lo), "hi": int(hi), "cross": int(cross),
                        "x": int((lo + hi) / 2) if c["axis"] == "h" else int(cross),
                        "y": int(cross) if c["axis"] == "h" else int((lo + hi) / 2),
                        "source": seg["source"],
                        "chain": "%s chain at %d" % (c["axis"], c["y"])})
    return out


# ANCHOR BEFORE 20261011_141710
def dimension_in_box(points, axis, box, pad_px=0, cover=None, reasons=None, offset_px=None):
    # ANCHOR AFTER 20261011_141710
    """The length drawn across this room, on the given axis.

    Three things must hold, and the first is what makes this reliable:

      - the segment lies within the room's own extent, so a run that reaches past
        the room belongs to something else
      - it covers most of that extent, so a closet jamb inside the room is not
        mistaken for the room
      - it sits beside or inside the room on the other axis

    Without the span test, the 15'7" drawn along the bottom of a bedroom would also
    be claimed by the narrow hall it passes over, because the midpoint falls inside
    the hall too.

    offset_px, when given, replaces pad_px on the other axis only: a drawing
    dimensions a room from outside it, so the run may sit that far beyond the room
    while still having to span it on the axis it measures.
    """
    cover = rule("matching.cover_rel") if cover is None else cover
    x0, y0, x1, y1 = box
    lo_r, hi_r = (x0, x1) if axis == "h" else (y0, y1)
    cross_lo, cross_hi = (y0, y1) if axis == "h" else (x0, x1)
    extent = max(1.0, hi_r - lo_r)
    best, seen, out_of_band, reaches_past, too_short = None, 0, 0, 0, 0
    for p in points:
        if p["axis"] != axis:
            continue
        seen += 1
        # ANCHOR BEFORE 20261011_141711
        side_pad = pad_px if offset_px is None else offset_px
        # ANCHOR AFTER 20261011_141711
        if not (cross_lo - side_pad <= p["cross"] <= cross_hi + side_pad):
            out_of_band += 1
            continue
        if p["lo"] < lo_r - pad_px or p["hi"] > hi_r + pad_px:
            reaches_past += 1
            continue
        if (p["hi"] - p["lo"]) < cover * extent:
            too_short += 1
            continue
        if best is None or p["ft"] > best["ft"]:
            best = p
    if reasons is not None and best is None:
        reasons.append("in the room: %d run(s) on this axis; %d beside the room, "
                       "%d reach past it, %d cover under %.0f%% of it"
                       % (seen, out_of_band, reaches_past, too_short, cover * 100))
    return None if best is None else (best["ft"], best["chain"], 0.0)


def dimension_over_point(points, axis, sx, sy, near_px, reasons=None):
    """The length drawn across the room's name.

    The last resort, and the one that survives a bad region. Everything else keys
    off the traced outline, so when the outline is wrong the right dimension is
    rejected with it: the Old Kitchen traces as 12'0" x 15'11" when the room is
    15'3" x 23'7", and no test against that box can accept the true width.

    The name, though, is read where the architect wrote it, in the middle of the
    room, and the dimension is drawn across the room. So a run that passes over the
    name and lies near it is that room's dimension, whatever the outline says.

    Nearest wins here, not longest: without an outline to bound it there is nothing
    to stop a longer run two rooms away from being picked.
    """
    best, over, too_far = None, 0, None
    for p in points:
        if p["axis"] != axis:
            continue
        along, cross = (sx, sy) if axis == "h" else (sy, sx)
        if not (p["lo"] <= along <= p["hi"]):
            continue
        over += 1
        d = abs(p["cross"] - cross)
        if d > near_px:
            too_far = d if too_far is None else min(too_far, d)
            continue
        if best is None or d < best[1]:
            best = (p, d)
    if reasons is not None and best is None:
        if not over:
            reasons.append("at the name: no run passes over it")
        else:
            reasons.append("at the name: %d run(s) pass over it, nearest %.0f px away "
                           "(limit %.0f)" % (over, too_far or 0, near_px))
    return None if best is None else (best[0]["ft"], best[0]["chain"], float(best[1]))


def span_between(idx, axis, lo, hi, tol_px, reasons=None):
    """The printed distance between the two joints a room's edges land on.

    Returns (feet, which chain, how far off the joints were in pixels), or None.
    Both joints must come from the same chain, and every segment between them
    must be known: adding up across a hole, or across two different chains, is
    not a distance the drawing states.
    """
    best = None
    for ch in idx.get(axis, []):
        i = min(range(len(ch["pos"])), key=lambda k: abs(ch["pos"][k] - lo))
        j = min(range(len(ch["pos"])), key=lambda k: abs(ch["pos"][k] - hi))
        if i == j:
            continue
        if i > j:
            i, j = j, i
        off = abs(ch["pos"][i] - min(lo, hi)) + abs(ch["pos"][j] - max(lo, hi))
        if off > 2 * tol_px:
            continue
        run = ch["seg"][i:j]
        if any(v is None for v in run):
            continue
        ft = sum(run)
        if ft <= 0:
            continue
        if best is None or off < best[2]:
            best = (ft, ch["ref"], off)
    if reasons is not None and best is None:
        reasons.append("on chain joints: %d chain(s) on this axis, none has a joint "
                       "near both edges within %.0f px" % (len(idx.get(axis, [])), tol_px))
    return best


def load_for_plan(plan_name, out_dir=OUT):
    """The chains written for this plan by an earlier dimlines run, if any."""
    f = Path(out_dir) / (Path(plan_name).stem + "_dimlines.json")
    if not f.exists():
        return None
    try:
        with open(f) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None
