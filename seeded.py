"""
Experimental mode for drawings where room boundaries are not closed lines
(historic survey sheets, hand-drawn sketches, plans with dashed or missing walls).

extract.py floods enclosed empty space, which shatters on such drawings. This
mode starts from the text instead:

  1. Read the sheet in small tiles, upscaled, on a copy with long lines removed
     (so room names touching walls or stair hatching are still readable).
  2. Keep only text that matches a room-name lexicon. Each hit is a seed.
  3. Grow every seed into the free space between long ink lines (watershed).
     Open space far from every label (margins, title block, notes) gets a
     background seed so it cannot be claimed by a room.
  4. Check each region against printed dimension strings found on the sheet.
     A room is marked verified only if a printed size agrees with the region.

Areas are approximate. Nothing here is marked "ok" without a printed size
agreeing with the measured region. CPU only, no network.

  python seeded.py --manifest manifest.csv --plans plans --out viewer
"""
import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import dimensions
import dimlines
import ruleset
from dims import all_lengths, all_singles, normalize, parse_dims
from runtrace import Trace, check_label, new_group, stamp

MAX_PIXELS = 120_000_000
WORK_PPF = 14.0          # geometry runs at about this many pixels per foot
OCR_PPF = 36.0           # text is upscaled to about this many pixels per foot for OCR
TILE_FT = 40.0
TILE_OVERLAP_FT = 6.0

# The vocabulary lives in lexicon.json, with a note on what each list is for and
# which words are fitted to a particular drawing rather than generic. It decides
# which OCR lines are room names; it moves no measurement.
LEXICON = ruleset.lexicon()
MODIFIERS = ruleset.modifiers()


def _letters(tok):
    return re.sub(r"[^A-Z0-9]", "", tok.upper())


def read_name(text, lexicon=None):
    """Return a canonical room name found in OCR text, or None.
    'WEST*FARLOR' -> 'WEST PARLOR', "BED'ROOM:A" -> 'BED ROOM A'."""
    lex = list(lexicon or LEXICON)
    toks = [t for t in re.sub(r"[^A-Za-z0-9]+", " ", text or "").upper().split() if t]
    seq = []
    for t in toks:
        kind = None
        word = None
        if re.fullmatch(r"ROOM[A-H1-9]", t):
            seq.append(("roomword", "ROOM"))
            seq.append(("tag", t[-1]))
            continue
        mt = re.fullmatch(r"([A-Z]{4,})(\d)", t)
        if mt and mt.group(1) in lex:
            seq.append(("room", mt.group(1)))
            seq.append(("tag", mt.group(2)))
            continue
        if sum(c.isalpha() for c in t) >= 4:
            t = re.sub(r"\d", "", t)
        if len(t) >= 3:
            if t in lex:
                kind, word = "room", t
            elif len(t) >= 4:
                m = difflib.get_close_matches(t, lex, n=1,
                                              cutoff=ruleset.rule("lexicon.room_cutoff"))
                if m:
                    kind, word = "room", m[0]
            if kind is None and t in MODIFIERS:
                kind, word = "mod", t
            if kind is None and len(t) >= 4:
                m = difflib.get_close_matches(
                    t, MODIFIERS, n=1, cutoff=ruleset.rule("lexicon.modifier_cutoff"))
                if m:
                    kind, word = "mod", m[0]
            if kind is None and difflib.SequenceMatcher(None, t, "ROOM").ratio() >= 0.85 and len(t) >= 4:
                kind, word = "roomword", "ROOM"
        elif re.fullmatch(r"[A-H1-9]", t):
            kind, word = "tag", t
        if kind:
            seq.append((kind, word))
    if not any(k == "room" for k, _ in seq):
        return None
    out = []
    for i, (k, w) in enumerate(seq):
        if k == "tag" and not (out and seq[i - 1][0] in ("room", "roomword")):
            continue
        out.append(w)
    return " ".join(out)


def text_layer(img_bgr, ppf, keep_crossings=False):
    """The drawing with long lines removed, so text that touches walls reads cleanly.
    Plain removal (default) is best for room names. A dimension string sits ON its dimension
    line, so plain removal erases the middle of the digits; keep_crossings=True keeps ink that is
    a stroke crossing the line (long in the other direction), which is better for dimension text
    but also keeps more noise."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(g, 160, 255, cv2.THRESH_BINARY_INV)
    L = max(5, int(1.2 * ppf))
    k = max(3, int(round(0.35 * ppf)))
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    cross_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    cross_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    if keep_crossings:
        hor_rm = cv2.bitwise_and(hor, cv2.bitwise_not(cv2.bitwise_and(cross_v, cv2.bitwise_not(ver))))
        ver_rm = cv2.bitwise_and(ver, cv2.bitwise_not(cv2.bitwise_and(cross_h, cv2.bitwise_not(hor))))
        txt = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.bitwise_or(hor_rm, ver_rm)))
        min_area = 4
    else:
        lines = cv2.dilate(cv2.bitwise_or(hor, ver), np.ones((3, 3), np.uint8))
        txt = cv2.bitwise_and(ink, cv2.bitwise_not(lines))
        min_area = 6
    n, lab, st, _ = cv2.connectedComponentsWithStats(txt, connectivity=8)
    small = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < min_area])
    txt[small] = 0
    if not keep_crossings:
        txt = cv2.morphologyEx(txt, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return 255 - txt


def _tile_boxes(w, h, ppf):
    ts = max(200, int(TILE_FT * ppf))
    step = max(100, ts - int(TILE_OVERLAP_FT * ppf))
    for y in range(0, max(1, h - int(TILE_OVERLAP_FT * ppf)), step):
        for x in range(0, max(1, w - int(TILE_OVERLAP_FT * ppf)), step):
            yield x, y, min(x + ts, w), min(y + ts, h)


def _ocr_tile(gray, box, up, psm, min_conf):
    import pytesseract
    from pytesseract import Output
    x0, y0, x1, y1 = box
    crop = gray[y0:y1, x0:x1]
    interp = cv2.INTER_CUBIC if up >= 1 else cv2.INTER_AREA
    big = cv2.resize(crop, None, fx=up, fy=up, interpolation=interp)
    d = pytesseract.image_to_data(big, config="--oem 3 --psm %d" % psm,
                                  output_type=Output.DICT, timeout=180)
    words = []
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        try:
            c = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if t and c >= min_conf:
            wx = x0 + d["left"][i] / up
            wy = y0 + d["top"][i] / up
            words.append({"text": t, "x0": wx, "y0": wy,
                          "x1": wx + d["width"][i] / up, "y1": wy + d["height"][i] / up})
    return words


def ocr_items(gray, ppf, psm=11, min_conf=25.0, workers=None):
    """Tiled OCR. Returns merged text lines [{text,cx,cy}] in the pixel frame of `gray`."""
    from ingest import merge_words
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    h, w = gray.shape
    up = OCR_PPF / ppf
    boxes = list(_tile_boxes(w, h, ppf))
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 1) as ex:
        parts = list(ex.map(lambda b: _ocr_tile(gray, b, up, psm, min_conf), boxes))
    words = [x for p in parts for x in p]
    return merge_words(words, keep_parts=True)


def florence_items(gray, ppf, run):
    """Same job as ocr_items, using the Florence-2 reader from read_labels.py.
    `run(pil_image)` returns [(text, quad)] with quad = x1,y1,...,x4,y4 in tile pixels.
    Tiles are read one at a time (the model is not thread safe)."""
    from PIL import Image
    h, w = gray.shape
    items = []
    for x0, y0, x1, y1 in _tile_boxes(w, h, ppf):
        crop = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2RGB)
        for text, quad in run(Image.fromarray(crop)):
            text = (text or "").replace("</s>", "").replace("<s>", "").strip()
            if not text:
                continue
            cx = x0 + sum(quad[0::2]) / len(quad[0::2])
            cy = y0 + sum(quad[1::2]) / len(quad[1::2])
            items.append({"text": text, "cx": cx, "cy": cy,
                          "parts": [{"text": text, "cx": cx, "cy": cy}]})
    return items


def unrotate(items, k, shape):
    """Map items found on np.rot90(img, k) back to the original pixel frame."""
    h, w = shape[:2]
    out = []
    for it in items:
        rx, ry = it["cx"], it["cy"]
        if k == 1:
            x, y = w - 1 - ry, rx
        else:
            x, y = ry, h - 1 - rx
        out.append({"text": it["text"], "cx": x, "cy": y})
    return out


def split_name_groups(it, ppf, lexicon=None, split_ft=3.0):
    """An OCR line can join two labels far apart through the marks between them ('HALL ----- BATH').
    Words of one room name sit within a foot or so of each other. Cut the line wherever two
    consecutive room-name words are more than split_ft apart. Returns [(text, parts)]; one group
    (the whole line) when nothing needs cutting or the parts carry no extent (Florence lines)."""
    parts = it.get("parts") or []
    if len(parts) < 2 or any("x0" not in p for p in parts):
        return [(it["text"], parts)]
    parts = sorted(parts, key=lambda p: p["x0"])
    hits = [i for i, p in enumerate(parts) if read_name(p["text"], lexicon)]
    if len(hits) < 2:
        return [(it["text"], parts)]
    clusters = [[hits[0]]]
    for a, b in zip(hits, hits[1:]):
        if parts[b]["x0"] - parts[a]["x1"] > split_ft * ppf:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    if len(clusters) < 2:
        return [(it["text"], parts)]
    groups = [[] for _ in clusters]
    for i, p in enumerate(parts):
        best, bd = None, None
        for k, c in enumerate(clusters):
            lo, hi = parts[c[0]], parts[c[-1]]
            d = max(lo["x0"] - p["x1"], p["x0"] - hi["x1"], 0.0)
            if bd is None or d < bd:
                best, bd = k, d
        if bd <= split_ft * ppf:
            groups[best].append(p)
    return [(" ".join(p["text"] for p in g), g) for g in groups if g]


def find_seeds(items, ppf, lexicon=None, trace=None):
    """One seed per room label. Nearby duplicates (from several OCR passes) collapse to
    the reading with the most recognised words."""
    cands = []
    for it0 in items:
        for gtext, gparts in split_name_groups(it0, ppf, lexicon):
            name = read_name(gtext, lexicon)
            if not name:
                continue
            # Place the seed on the words that spell the name, not on the middle of the whole
            # OCR line: stray marks next to a label can drag the line's centre out of the room.
            hit = [p for p in gparts if read_name(p["text"], lexicon)]
            if hit:
                cx = sum(p["cx"] for p in hit) / len(hit)
                cy = sum(p["cy"] for p in hit) / len(hit)
            else:
                cx, cy = it0["cx"], it0["cy"]
            cands.append((len(name.split()), name, cx, cy, it0["text"]))
    cands.sort(key=lambda c: -c[0])
    radius = 5.0 * ppf
    seeds = []
    for nwords, name, x, y, text in cands:
        near = next(((n, sx, sy) for n, sx, sy in seeds
                     if (sx - x) ** 2 + (sy - y) ** 2 < radius ** 2), None)
        if trace is not None:
            trace.add("seeds", {"name": name, "x": round(x), "y": round(y), "ocr_text": text,
                                "recognised_words": nwords,
                                "decision": "kept" if near is None else
                                "collapsed into %s at (%d,%d)" % (near[0], near[1], near[2])})
        if near is not None:
            continue
        seeds.append((name, x, y))
    return seeds


def extend_casings(walls, ink, axis, ppf):
    """Carry each straight wall on through the casing drawn at its end.

    A door casing is often drawn as moulded steps, each too short and off the wall's
    line, so the straight wall stops at the first step and the gap left is wider than
    the door: doors 4 and 8 on the sample sheet, 2'-1" and 2'-4", measured 3.1 and
    3.4 ft between straight walls. From each end of a wall the ink attached to it is
    followed along the wall, inside a casing's depth either side of its line, for at
    most a casing's length, and the wall is drawn on that far. It reads only where a
    wall ends, never whether a line is a wall.
    """
    depth = max(1, int(round(ruleset.rule("segmentation.casing_depth_ft") * ppf)))
    reach = max(1, int(round(ruleset.rule("segmentation.casing_reach_ft") * ppf)))
    h, w = walls.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(walls, connectivity=8)
    out = walls.copy()
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        if axis == "v":
            c, half, ends, size = x + bw // 2, max(1, bw // 2), ((y, -1), (y + bh - 1, 1)), h
        else:
            c, half, ends, size = y + bh // 2, max(1, bh // 2), ((x, -1), (x + bw - 1, 1)), w
        for e, step in ends:
            last = e
            for k in range(1, reach + 1):
                p = e + step * k
                if not 0 <= p < size:
                    break
                band = (ink[p, max(0, c - depth):c + depth + 1] if axis == "v"
                        else ink[max(0, c - depth):c + depth + 1, p])
                if not band.any():
                    break
                last = p
            if last != e:
                a, b = sorted((e, last))
                if axis == "v":
                    out[a:b + 1, max(0, c - half):c + half + 1] = 255
                else:
                    out[max(0, c - half):c + half + 1, a:b + 1] = 255
    return out


def corner_casings(walls, ink, axis, ppf):
    """The jamb of a door at a corner, carried on from the end of the perpendicular wall.

    At a corner door the jamb hangs off the end of a wall that runs the other way: door
    4 on the sample sheet sits where the entry hall's partition meets the south wall, and
    its casing rises off the south wall's end. There is no run on the door's own axis to
    carry on, so the casing is followed from that end across the wall, inside the same
    casing depth either side of the end and for no more than the same reach, and drawn
    on the other axis as the jamb. Like extend_casings it starts only at the end of a
    wall already found and decides how far that wall's corner goes; it never decides
    whether a line is a wall.

    walls are the runs on `axis`; the result is a map of jambs on the other axis.
    """
    depth = max(1, int(round(ruleset.rule("segmentation.casing_depth_ft") * ppf)))
    reach = max(1, int(round(ruleset.rule("segmentation.casing_reach_ft") * ppf)))
    h, w = walls.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(walls, connectivity=8)
    out = np.zeros_like(walls)
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        if axis == "h":
            c, half, ends, size = y + bh // 2, max(1, bh // 2), (x, x + bw - 1), h
        else:
            c, half, ends, size = x + bw // 2, max(1, bw // 2), (y, y + bh - 1), w
        for e in ends:
            for step in (-1, 1):
                last = c
                for k in range(1, half + reach + 1):
                    p = c + step * k
                    if not 0 <= p < size:
                        break
                    band = (ink[p, max(0, e - depth):e + depth + 1] if axis == "h"
                            else ink[max(0, e - depth):e + depth + 1, p])
                    if not band.any():
                        break
                    last = p
                if abs(last - c) > half + 1:
                    a, b = sorted((c, last))
                    if axis == "h":
                        out[a:b + 1, max(0, e - half):e + half + 1] = 255
                    else:
                        out[max(0, e - half):e + half + 1, a:b + 1] = 255
    return out


def line_map(gray, ppf, min_ft=2.5, erase=None):
    _, ink = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Dimension lines proven by a believed reading are taken out before the opening
    # below, so what is left of one, under its text or beside a wall, is too short to
    # survive it and cannot stand in for a wall. erase is ((x0, y0), (x1, y1)) in this
    # image's pixels.
    # A line is taken out only of the ink its own direction's walls are built from. Cut
    # out of all the ink, a horizontal dimension line also cut every wall it crossed,
    # and a wall stub left under 2.5 ft vanished: the bath's west wall above its door
    # went that way and the bath flowed into the hall.
    ink_h, ink_v = ink, ink
    if erase:
        width = max(2, int(round(ruleset.rule("detection.line_max_thick_ft") * ppf)))
        ink_h, ink_v = ink.copy(), ink.copy()
        for (x0, y0), (x1, y1) in erase:
            target = ink_h if abs(x1 - x0) >= abs(y1 - y0) else ink_v
            cv2.line(target, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 0, width)
    L = max(5, int(min_ft * ppf))
    hor = cv2.morphologyEx(ink_h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    ver = cv2.morphologyEx(ink_v, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    # Doorways are gaps in a straight wall. Bridging each wall along its own direction
    # closes them without thickening the wall or reaching round a corner, so a room stops
    # at its door instead of flowing into the next one.
    door = int(ruleset.rule("segmentation.door_max_ft") * ppf)
    if door > 1:
        # A door at a corner has a short jamb on one side, shorter than any wall the
        # opening above keeps, so the closing below had nothing to bridge from. Short
        # straight runs that touch a kept wall are jambs and go back in first.
        jamb = max(2, int(ruleset.rule("segmentation.jamb_min_ft") * ppf))
        kept = cv2.dilate(cv2.bitwise_or(hor, ver), np.ones((3, 3), np.uint8))
        for lines_on_axis, k, src in ((hor, (jamb, 1), ink_h), (ver, (1, jamb), ink_v)):
            short = cv2.morphologyEx(src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
            _, cc = cv2.connectedComponents(short, connectivity=8)
            touching = np.unique(cc[(kept > 0) & (short > 0)])
            lines_on_axis[np.isin(cc, touching[touching > 0])] = 255
        hor = extend_casings(hor, ink_h, "h", ppf)
        ver = extend_casings(ver, ink_v, "v", ppf)
        jambs_v, jambs_h = corner_casings(hor, ink_v, "h", ppf), corner_casings(ver, ink_h, "v", ppf)
        ver, hor = cv2.bitwise_or(ver, jambs_v), cv2.bitwise_or(hor, jambs_h)
        hor = cv2.morphologyEx(hor, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (door, 1)))
        ver = cv2.morphologyEx(ver, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, door)))
    return cv2.bitwise_or(hor, ver)


def segment(gray, ppf, seeds_xy, gap_ft=2.0, bg_far_ft=15.0, bg_boxes=None, erase=None):
    """Label-seeded watershed. Returns an int label image: 0 = wall/none, i = seed i
    (1-based), len(seeds)+1 = background. bg_boxes: (x0,y0,x1,y1) areas planted as background
    (title block, legend, tables) so no room can grow into them."""
    from skimage.segmentation import watershed
    lines = line_map(gray, ppf, erase=erase)
    r = max(1, int(gap_ft * ppf / 2))
    walls = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    free = walls == 0
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    h, w = gray.shape
    markers = np.zeros((h, w), np.int32)
    n = len(seeds_xy)
    snap = int(2.0 * ppf)
    for i, (x, y) in enumerate(seeds_xy, start=1):
        x, y = int(round(x)), int(round(y))
        if 0 <= y < h and 0 <= x < w and not free[y, x]:
            # a label that sits on a wall line would get no region: move it to the nearest free pixel
            y0, y1, x0, x1 = max(0, y - snap), min(h, y + snap + 1), max(0, x - snap), min(w, x + snap + 1)
            wy, wx = np.nonzero(free[y0:y1, x0:x1])
            if wy.size:
                k = int(np.argmin((wy + y0 - y) ** 2 + (wx + x0 - x) ** 2))
                x, y = int(wx[k] + x0), int(wy[k] + y0)
                for _ in range(int(6 * ppf)):  # climb to the middle of that side, away from the wall
                    ny0, ny1, nx0, nx1 = max(0, y - 1), min(h, y + 2), max(0, x - 1), min(w, x + 2)
                    win = dist[ny0:ny1, nx0:nx1]
                    j = np.unravel_index(int(np.argmax(win)), win.shape)
                    if win[j] <= dist[y, x]:
                        break
                    y, x = ny0 + int(j[0]), nx0 + int(j[1])
        cv2.circle(markers, (x, y), max(2, int(0.35 * ppf)), i, -1)
    ext = n + 1
    far = np.full((h, w), 1e9, np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for (x, y) in seeds_xy:
        far = np.minimum(far, np.hypot(xs - x, ys - y))
    cand = (dist > 3.0 * ppf) & (far > bg_far_ft * ppf)
    markers[cand & (markers == 0)] = ext
    for (x0, y0, x1, y1) in (bg_boxes or []):
        sl = (slice(max(0, int(y0)), min(h, int(y1) + 1)), slice(max(0, int(x0)), min(w, int(x1) + 1)))
        markers[sl][markers[sl] == 0] = ext
    b = 6
    markers[:b, :] = ext
    markers[-b:, :] = ext
    markers[:, :b] = ext
    markers[:, -b:] = ext
    lab = watershed(-dist, markers, mask=free).astype(np.int32)
    ink_free = lines == 0
    for _ in range(r + 1):
        grown = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        fill = (lab == 0) & ink_free & (grown > 0)
        lab[fill] = grown[fill]
    return lab, ext


def region_mask(lab, i, seed_xy, ppf, hole_sqft=60.0):
    """The connected piece of label i that holds its seed, small holes filled."""
    m = (lab == i).astype(np.uint8)
    n, cc = cv2.connectedComponents(m, connectivity=4)
    if n <= 1:
        return np.zeros(m.shape, bool)
    x, y = int(seed_xy[0]), int(seed_xy[1])
    k = cc[min(max(y, 0), m.shape[0] - 1), min(max(x, 0), m.shape[1] - 1)]
    if k == 0:
        sizes = [(cc == j).sum() for j in range(1, n)]
        k = 1 + int(np.argmax(sizes))
    comp = (cc == k).astype(np.uint8)
    cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for c, hh in zip(cnts, hier[0]):
            if hh[3] >= 0 and cv2.contourArea(c) < hole_sqft * ppf * ppf:
                cv2.drawContours(comp, [c], -1, 1, thickness=-1)
    return comp.astype(bool)


def dimension_blobs(layer, ppf, orient):
    """Boxes around runs of text-sized ink that could be one dimension string. orient 'v' finds
    text running up or down the page (dimension chains down the side of a sheet)."""
    ink = (layer < 128).astype(np.uint8) * 255
    along = max(3, int(0.5 * ppf))
    across = max(1, int(0.15 * ppf))
    kern = (across, along) if orient == "v" else (along, across)
    bl = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, kern))
    n, _, st, _ = cv2.connectedComponentsWithStats(bl, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in st[i, :4])
        height, length = (w, h) if orient == "v" else (h, w)
        if 0.3 * ppf <= height <= 1.6 * ppf and length >= 1.2 * ppf and length >= 1.4 * height:
            boxes.append((x, y, w, h))
    return boxes


def _read_blob(layer, box, orient, ppf, min_conf=0.0):
    import pytesseract
    from pytesseract import Output
    x, y, w, h = box
    pad = max(2, int(0.3 * ppf))
    crop = layer[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    text_h = float(h if orient == "h" else w)
    up = min(10.0, max(1.0, 40.0 / max(4.0, text_h)))
    best = None
    for k in ((0,) if orient == "h" else (1, -1)):
        c = np.rot90(crop, k) if k else crop
        big = cv2.resize(np.ascontiguousarray(c), None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        try:
            d = pytesseract.image_to_data(big, config="--oem 3 --psm 7",
                                          output_type=Output.DICT, timeout=30)
        except Exception:
            continue
        words = []
        for t, c in zip(d["text"], d["conf"]):
            try:
                cf = float(c)
            except (TypeError, ValueError):
                continue
            if (t or "").strip() and cf >= min_conf:
                words.append((t.strip(), cf))
        if not words:
            continue
        text = " ".join(t for t, _ in words)
        conf = sum(c for _, c in words) / len(words)
        wordy = re.search(r"[A-Za-z]{3,}", text) is not None   # a name or note, not a dimension
        score = conf if (all_lengths(text) and not wordy) else conf - 1000.0
        if best is None or score > best[0]:
            best = (score, text, conf)
    if best is None or best[0] < 0:
        return None
    return {"text": best[1], "cx": x + w / 2.0, "cy": y + h / 2.0, "conf": round(best[2], 1)}


def read_dimension_strings(layer, ppf, orient="v", workers=None):
    """Dedicated reader for dimension strings. Each candidate string is cropped on its own,
    turned upright (both ways for vertical text), upscaled to a readable text height and read as
    a single line. Only readings that contain a foot length are kept. Tesseract only."""
    boxes = dimension_blobs(layer, ppf, orient)
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 1) as ex:
        res = list(ex.map(lambda b: _read_blob(layer, b, orient, ppf), boxes))
    return [r for r in res if r]


def find_schedules(items, ppf, min_rows=3):
    """Tables of repeated 'mark  W x H x T  note' rows, such as a door schedule: three or more
    text lines stacked one above the other, each holding at least two printed lengths joined by
    'x'. Returns [{bbox, rows:[{mark, lengths_ft, text}]}]. Used to keep schedule text out of
    room matching and to report the table; it is not needed for room areas."""
    cand = []
    for it in items:
        t = normalize(it["text"])
        lens = all_lengths(t)
        if len(lens) >= 2 and re.search(r"[xX\u00d7]", t):
            cand.append((it["cy"], it["cx"], it["text"], lens, it))
    cand.sort(key=lambda c: c[0])
    out, chain = [], []

    def close():
        if len(chain) >= min_rows:
            xs = [c[4]["cx"] for c in chain]
            ys = [c[0] for c in chain]
            rows = []
            for _, _, text, lens, _ in chain:
                m = re.match(r"^[^0-9]*?(\d{1,2})(?![\d'\"])\s", normalize(text) + " ")
                rows.append({"mark": m.group(1) if m else None, "lengths_ft": lens, "text": text})
            out.append({"bbox": [int(min(xs) - 6 * ppf), int(min(ys) - ppf),
                                 int(max(xs) + 6 * ppf), int(max(ys) + ppf)], "rows": rows})

    for c in cand:
        if chain and (c[0] - chain[-1][0] > 2.5 * ppf or abs(c[1] - chain[-1][1]) > 8 * ppf):
            close()
            chain = []
        chain.append(c)
    close()
    return out


def collect_dims(h_items, v_items, max_per_item=3, skipped=None):
    """Every single printed length on the sheet with its position. Pairs ('W x D') are not
    included; those are read by read_labels / interpret. An OCR item holding more than
    max_per_item lengths is a whole dimension chain read as one string: its lengths would all
    be given the same position, so it is not used (and listed in `skipped` if given)."""
    out = []
    for orient, items in (("h", h_items), ("v", v_items)):
        seen = set()
        for it in items:
            singles = all_singles(it["text"])
            if len(singles) > max_per_item:
                if skipped is not None:
                    skipped.append({"orient": orient, "n_lengths": len(singles), "text": it["text"][:80],
                                    "x": round(it["cx"]), "y": round(it["cy"])})
                continue
            for v in singles:
                key = (orient, v, round(it["cx"] / 25), round(it["cy"] / 25))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"ft": v, "cx": it["cx"], "cy": it["cy"], "orient": orient,
                            "text": it["text"]})
    return out


def aligned_lengths(bbox, dims, ppf, margin_ft=1.5, cross_ft=None):
    """Printed lengths whose position lines up with the region: same columns for horizontal
    text (width), same rows for vertical text (depth), and also close to the region on the
    other axis (cross_ft), so a length printed on another floor plan or at the far side of the
    sheet cannot match by coincidence. Returns (width_ft, depth_ft, per-axis list of
    {ft, cx, cy, rel_err})."""
    x0, y0, x1, y1 = bbox
    wft = (x1 - x0 + 1) / ppf
    dft = (y1 - y0 + 1) / ppf
    if cross_ft is None:
        cross_ft = ruleset.rule("matching.text_cross_ft")
    m = margin_ft * ppf
    c = cross_ft * ppf
    out = {"width": [], "depth": []}
    for key, span, lo, hi, cross, clo, chi, axis, meas in (
        ("width", "cx", x0 - m, x1 + m, "cy", y0 - c, y1 + c, "h", wft),
        ("depth", "cy", y0 - m, y1 + m, "cx", x0 - c, x1 + c, "v", dft),
    ):
        for d in dims:
            if d["orient"] == axis and lo <= d[span] <= hi and clo <= d[cross] <= chi:
                out[key].append({"ft": d["ft"], "cx": round(d["cx"]), "cy": round(d["cy"]),
                                 "rel_err": round(abs(d["ft"] - meas) / d["ft"], 3)})
    return wft, dft, out


def verify_region(bbox, dims, ppf, tol=0.08, margin_ft=1.5, detail=False, cross_ft=None):
    """Compare a region's width and depth with aligned printed lengths.
    Returns {'width': printed_ft|None, 'depth': printed_ft|None}; with detail=True also
    'measured_ft' and 'considered' (every aligned length and its relative error)."""
    wft, dft, aligned = aligned_lengths(bbox, dims, ppf, margin_ft, cross_ft)
    best = {"width": None, "depth": None}
    for key in ("width", "depth"):
        ok = [c for c in aligned[key] if c["rel_err"] <= tol]
        if ok:
            best[key] = min(ok, key=lambda c: c["rel_err"])["ft"]
    if detail:
        best["measured_ft"] = [round(wft, 2), round(dft, 2)]
        best["considered"] = aligned
    return best


def shape_stats(mask):
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    area = int(mask.sum())
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.contourArea(cv2.convexHull(np.vstack(cnts))) if cnts else 0.0
    solidity = area / hull if hull > 0 else 0.0
    return bbox, area, solidity


def clean_raw_texts(texts, max_items=40, max_chars=1500):
    """Order-preserving de-duplicate of the OCR strings found inside a room. Several OCR passes
    read the same text, and junk lines (marks, single characters) add nothing; both bloat the
    prompt sent to the local model."""
    seen, out, total = set(), [], 0
    for t in texts:
        key = re.sub(r"[^a-z0-9]", "", (t or "").lower())
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        if len(out) >= max_items or total + len(t) > max_chars:
            break
        out.append(t)
        total += len(t)
    return out


def plan_extents(lab, ext, gap_px):
    """Each floor plan's footprint, from the regions themselves, in working pixels.

    A sheet can carry several plans side by side. Columns that hold any room region
    are grouped, and an empty run wider than gap_px starts a new plan: rooms on one
    plan abut across walls and doors, while two drawings on a sheet are set well apart.
    Returns [(x0, y0, x1, y1), ...] left to right.
    """
    rooms = (lab > 0) & (lab != ext)
    cols = np.nonzero(rooms.any(axis=0))[0]
    if not cols.size:
        return []
    groups = [[int(cols[0]), int(cols[0])]]
    for c in cols[1:]:
        if c - groups[-1][1] > gap_px:
            groups.append([int(c), int(c)])
        else:
            groups[-1][1] = int(c)
    out = []
    for a, b in groups:
        rows = np.nonzero(rooms[:, a:b + 1].any(axis=1))[0]
        out.append((a, int(rows[0]), b, int(rows[-1])))
    return out


def plan_of(foot, plans, margin_rel):
    """Which plan a run of ink belongs to, or None when it is not clearly one.

    foot is (x0, y0, x1, y1). The nearest plan wins only when it is clearly nearer
    than the next: a chain drawn in the space between two plans says nothing about
    which one it measures.
    """
    if not plans:
        return None
    dist = []
    for i, (px0, py0, px1, py1) in enumerate(plans):
        dx = max(0, px0 - foot[2], foot[0] - px1)
        dy = max(0, py0 - foot[3], foot[1] - py1)
        dist.append((float(np.hypot(dx, dy)), i))
    dist.sort()
    if len(dist) == 1 or dist[0][0] <= margin_rel * dist[1][0]:
        return dist[0][1]
    return None


def walls_crossed(lines, axis, along, c_from, c_to, faces_px):
    """Walls met on the straight path across a run, from c_from to c_to.

    axis is the run's axis, so the path runs along the other one at `along`. The two
    faces of one wall, closer than faces_px, count once.
    """
    h, w = lines.shape
    lo, hi = sorted((int(round(c_from)), int(round(c_to))))
    a = int(round(along))
    if axis == "h":
        if not 0 <= a < w:
            return 0
        path = lines[max(0, lo):min(h, hi + 1), a] > 0
    else:
        if not 0 <= a < h:
            return 0
        path = lines[a, max(0, lo):min(w, hi + 1)] > 0
    count, gap = 0, None
    for on in path:
        if on:
            if gap is None or gap > faces_px:
                count += 1
            gap = 0
        elif gap is not None:
            gap += 1
    return count


def is_partition(lines, axis, at, plan, side, min_px, skip_px):
    """Whether a joint on a perimeter chain lands on a wall running into the building.

    A perimeter chain is divided at window jambs and at partitions alike. A jamb is a
    reveal a few inches deep; a partition runs into the plan for room lengths. axis is
    the chain's axis and `at` the joint's position along it; side says which edge of
    the plan the chain lies beyond.
    """
    x0, y0, x1, y1 = plan
    a = int(round(at))
    if axis == "v":
        if not 0 <= a < lines.shape[0]:
            return False
        span = (lines[max(0, a - 1):a + 2, x0 + skip_px:x0 + min_px] if side == "low"
                else lines[max(0, a - 1):a + 2, x1 - min_px:x1 - skip_px])
        return span.size > 0 and (span > 0).any(axis=0).mean() >= 0.8
    if not 0 <= a < lines.shape[1]:
        return False
    span = (lines[y0 + skip_px:y0 + min_px, max(0, a - 1):a + 2] if side == "low"
            else lines[y1 - min_px:y1 - skip_px, max(0, a - 1):a + 2])
    return span.size > 0 and (span > 0).any(axis=1).mean() >= 0.8


def chain_lookup(chain_data, full_height, scale, wppf, trace=None, tol_ft=None, near_ft=None,
                 lab=None, ext=None, lines=None):
    """Give build_rooms a way to ask "what do the chains say this room measures?".

    Geometry runs on a shrunken copy of the sheet while the chains were found on
    the full-size one, so a room's bounding box is scaled back up before it is
    compared with the joints. Returns None when there are no usable chains, so
    the older checks carry on as before.
    """
    tol_ft = ruleset.rule("matching.joint_tol_ft") if tol_ft is None else tol_ft
    near_ft = ruleset.rule("matching.name_near_ft") if near_ft is None else near_ft
    # Not "no chains": a drawing can carry its room dimensions entirely as lone
    # dimension lines and have no chain at all, and that plan still needs matching.
    if not chain_data or not (chain_data.get("chains") or chain_data.get("single_dimensions")):
        return None
    import dimlines
    idx = dimlines.chain_index(chain_data, full_height)
    points = dimlines.dimension_points(chain_data, full_height)
    if not (idx["h"] or idx["v"]) and not points:
        if trace is not None:
            trace.warn("dimension chains were found but none has a single solved segment, "
                       "so none can measure a room")
        return None
    tol_px = tol_ft * wppf / max(scale, 1e-9)

    near_px = near_ft * wppf / max(scale, 1e-9)

    # With the regions and the wall map at hand, a dimension may only measure a room it
    # can belong to: on the same plan, not across an interior wall, and on a perimeter
    # chain only between joints that are partitions. Without them (a bare chain file)
    # the rules run as they always have.
    geo = lab is not None and lines is not None
    if geo:
        plans = plan_extents(lab, ext, ruleset.rule("matching.plan_gap_ft") * wppf)
        margin_rel = ruleset.rule("matching.plan_margin_rel")
        faces_px = ruleset.rule("matching.wall_faces_ft") * wppf
        part_px = int(ruleset.rule("matching.partition_min_ft") * wppf)
        offset_px = ruleset.rule("matching.box_offset_ft") * wppf / max(scale, 1e-9)
        if trace is not None:
            trace.data["params"]["plans_found"] = [[round(v / scale) for v in p] for p in plans]

    def placed(axis, lo, hi, cross, wbox, room_plan, from_point=None):
        """(usable, perimeter side or None, why not). Sheet in, working pixels inside."""
        lo_w, hi_w, c_w = lo * scale, hi * scale, cross * scale
        foot = (lo_w, c_w, hi_w, c_w) if axis == "h" else (c_w, lo_w, c_w, hi_w)
        pl = plan_of(foot, plans, margin_rel)
        if pl is None:
            return False, None, "between plans"
        if pl != room_plan:
            return False, None, "on another plan"
        bx0, by0, bx1, by1 = wbox
        clo, chi = (by0, by1) if axis == "h" else (bx0, bx1)
        alo, ahi = (bx0, bx1) if axis == "h" else (by0, by1)
        px0, py0, px1, py1 = plans[pl]
        plo, phi = (py0, py1) if axis == "h" else (px0, px1)
        side = "low" if c_w < plo else ("high" if c_w > phi else None)
        if from_point is not None:
            start = from_point[1]
        else:
            start = None if clo <= c_w <= chi else (clo if c_w < clo else chi)
        if start is not None:
            # Only the room's own outer wall may stand between it and a dimension
            # drawn outside the building; any other wall means another space. Probed
            # across the stretch the run and the room share, and the median taken, so
            # a single doorway in the way cannot hide the wall it sits in.
            a, b = max(lo_w, alo), min(hi_w, ahi)
            if a > b:
                a = b = min(max((lo_w + hi_w) / 2, alo), ahi)
            counts = sorted(walls_crossed(lines, axis, a + (b - a) * f, start, c_w, faces_px)
                            for f in (0.1, 0.3, 0.5, 0.7, 0.9))
            if counts[2] > (1 if side else 0):
                return False, None, "across a wall"
        return True, side, None

    def usable_chains(axis, wbox, room_plan, why):
        out, refused = [], {}
        for ch in idx.get(axis, []):
            ok, side, reason = placed(axis, ch["pos"][0], ch["pos"][-1], ch["cross"],
                                      wbox, room_plan)
            if not ok:
                refused[reason] = refused.get(reason, 0) + 1
                continue
            if side is None:
                out.append(ch)
                continue
            # Perimeter chain: keep only joints that are partitions, merging the
            # segments between them; a gap anywhere inside makes the merged one unknown.
            keep = [k for k, p in enumerate(ch["pos"])
                    if is_partition(lines, axis, p * scale, plans[room_plan], side,
                                    part_px, int(faces_px))]
            if len(keep) < 2:
                refused["perimeter, joints not partitions"] = refused.get(
                    "perimeter, joints not partitions", 0) + 1
                continue
            segs = []
            for a, b in zip(keep, keep[1:]):
                run = ch["seg"][a:b]
                segs.append(None if any(v is None for v in run) else sum(run))
            out.append(dict(ch, pos=[ch["pos"][k] for k in keep], seg=segs))
        if refused:
            why.append("refused: " + ", ".join("%d %s" % (n, r) for r, n in sorted(refused.items())))
        return {axis: out}

    def usable_points(axis, wbox, room_plan, why, from_point=None):
        out, refused = [], {}
        for p in points:
            if p["axis"] != axis:
                continue
            ok, side, reason = placed(axis, p["lo"], p["hi"], p["cross"], wbox, room_plan,
                                      from_point)
            if ok and side is not None and not all(
                    is_partition(lines, axis, e * scale, plans[room_plan], side, part_px,
                                 int(faces_px)) for e in (p["lo"], p["hi"])):
                ok, reason = False, "perimeter, ends not partitions"
            if ok:
                out.append(p)
            else:
                refused[reason] = refused.get(reason, 0) + 1
        if refused:
            why.append("refused%s: %s" % (" at the name" if from_point else "",
                                          ", ".join("%d %s" % (n, r) for r, n in sorted(refused.items()))))
        return out

    def ask(bbox, seed=None):
        box = tuple(v / scale for v in bbox)
        x0, y0, x1, y1 = box
        sx, sy = (seed[0] / scale, seed[1] / scale) if seed else (None, None)
        # First choice: the room's edges land on two joints of one chain, which
        # gives the distance the drawing states between those exact points.
        # Second: a length written inside the room. Weaker, because it relies on
        # where the draughtsman put the text, but it is how the sheet is meant to
        # be read and it catches rooms whose walls are not chain joints.
        out, log = {}, []
        rules = {}
        room_plan = plan_of(tuple(bbox), plans, margin_rel) if geo else None
        for key, axis in (("width", "h"), ("depth", "v")):
            why = []
            lo, hi = (x0, x1) if axis == "h" else (y0, y1)
            if geo and room_plan is None:
                why.append("the room is not clearly on one plan, so nothing is matched to it")
                chains_ok, pts_box, pts_name = {axis: []}, [], []
            elif geo:
                chains_ok = usable_chains(axis, bbox, room_plan, why)
                pts_box = usable_points(axis, bbox, room_plan, why)
                pts_name = (usable_points(axis, bbox, room_plan, why,
                                          from_point=((seed[0], seed[1]) if axis == "h"
                                                      else (seed[1], seed[0])))
                            if seed else [])
            else:
                chains_ok, pts_box, pts_name = idx, points, points
            got = dimlines.span_between(chains_ok, axis, lo, hi, tol_px, why)
            rule = "chain joints" if got else None
            if not got:
                got = dimlines.dimension_in_box(pts_box, axis, box, tol_px, reasons=why,
                                                offset_px=offset_px if geo else None)
                rule = "in the room" if got else None
            if not got and seed:
                # last: anchored on the room's name, which is read in the right place
                # even when the outline is not
                got = dimlines.dimension_over_point(pts_name, axis, sx, sy, near_px, why)
                rule = "at the name" if got else None
            out[key] = got
            if got:
                rules[key] = rule
                log.append("%s: %s from %s (%s)" % (key, dimensions.fmt(got[0]), got[1], rule))
            else:
                log += ["%s: %s" % (key, w) for w in why]
        out["log"] = log
        out["rules"] = rules
        if geo:
            out["plan"] = room_plan
        return out

    if trace is not None:
        trace.data["params"]["chains_available"] = {"h": len(idx["h"]), "v": len(idx["v"]),
                                                    "lengths_placed": len(points)}
    return ask


def build_rooms(lab, seeds, ppf, items, dims, trace=None, chains=None):
    rooms = []
    for i, (name, sx, sy) in enumerate(seeds, start=1):
        # Set only for a room that was checked against printed lengths; a room confirmed by
        # its chains never is, and used to inherit the previous room's list in the trace.
        trace_considered = None
        mask = region_mask(lab, i, (sx, sy), ppf)
        px = int(mask.sum())
        if px == 0:
            rooms.append({"id": i, "label": name, "area_px": 0, "area_sqft": 0.0,
                          "centroid": [int(sx), int(sy)], "bbox": [int(sx), int(sy), int(sx), int(sy)],
                          "raw_text": [], "needs_review": True,
                          "review_reason": "no region found around this label"})
            continue
        bbox, px, solidity = shape_stats(mask)
        sqft = px / (ppf * ppf)
        auto = name.startswith("(")
        ys, xs = np.nonzero(mask)
        raw = clean_raw_texts([it["text"] for it in items
                               if 0 <= int(it["cy"]) < mask.shape[0] and 0 <= int(it["cx"]) < mask.shape[1]
                               and mask[int(it["cy"]), int(it["cx"])]])
        rm = {"id": i, "label": name, "area_px": px, "area_sqft": round(sqft, 1),
              "centroid": [int(xs.mean()), int(ys.mean())], "bbox": bbox,
              "solidity": round(solidity, 2), "raw_text": raw,
              # the region's own width and depth, which is what the printed size is
              # compared against. Kept on the room so results.json and the viewer
              # carry it, not only the trace.
              "measured_ft": [round((bbox[2] - bbox[0] + 1) / ppf, 2),
                              round((bbox[3] - bbox[1] + 1) / ppf, 2)]}
        # printed "W x D" inside the region, if any
        pair = next((parse_dims(t) for t in raw if parse_dims(t)), None)
        notes = []
        verified = False
        # A dimension chain is the strongest evidence there is: the room's edges
        # land on two joints and the distance between them is printed on the
        # drawing. Try that before anything measured off pixels.
        from_chain = chains(bbox, (sx, sy)) if chains else None
        if from_chain:
            rm["match_log"] = from_chain.pop("log", [])
            # The plan the room sits on, so a value is only ever shared within one plan.
            if "plan" in from_chain:
                rm["plan"] = from_chain.pop("plan")
            # Which rule gave each side, and from which chain, so a value lent to a
            # neighbour carries where it came from.
            rm["chain_rules"] = from_chain.pop("rules", {})
            rm["chain_side_refs"] = {k: v[1] for k, v in from_chain.items() if v}
            rm["chain_sides"] = {k: (None if not v else round(v[0], 3))
                                 for k, v in from_chain.items()}
        if from_chain and from_chain["width"] and from_chain["depth"]:
            w, d = from_chain["width"], from_chain["depth"]
            stated = w[0] * d[0]
            rm["stated_dims_ft"] = [round(w[0], 3), round(d[0], 3)]
            rm["dims_source"] = "dimension chain"
            rm["stated_sqft"] = round(stated, 1)
            rm["chain_refs"] = [w[1], d[1]]
            rm["agreement_gap_pct"] = round(abs(sqft - stated) / stated * 100, 1)
            ok, why = sides_agree(bbox, ppf, w[0], d[0])
            rm["side_errors_pct"] = why
            if ok:
                verified = True
                rm["fill_pct"] = round(sqft / stated * 100, 1)
            else:
                notes.append("the chains give %s x %s but the region measures %s"
                             % (dimensions.fmt(w[0]), dimensions.fmt(d[0]),
                                ", ".join("%s %+.0f%%" % (k, v) for k, v in why.items())))
        elif from_chain and (from_chain["width"] or from_chain["depth"]):
            side = "width" if from_chain["width"] else "depth"
            rm["chain_partial"] = {side: round(from_chain[side][0], 3),
                                   "chain": from_chain[side][1]}
            notes.append("only the %s of this room lands on a dimension chain" % side)
        if verified:
            pass
        elif pair:
            stated = pair[0] * pair[1]
            gap = abs(sqft - stated) / stated * 100
            rm["stated_dims_ft"] = list(pair)
            rm["dims_source"] = "rules"
            rm["stated_sqft"] = round(stated, 1)
            rm["agreement_gap_pct"] = round(gap, 1)
            if gap <= 10:
                verified = True
            else:
                notes.append("printed size and measured area differ by %.0f%%" % gap)
        else:
            if solidity >= 0.75:
                got = verify_region(bbox, dims, ppf, detail=trace is not None)
                if trace is not None:
                    trace_considered = got.pop("considered")
                    got.pop("measured_ft", None)   # the room already carries this
                rm["chain_match_ft"] = got
                if got["width"] and got["depth"]:
                    verified = True
                    rm["dims_source"] = "dimension strings"
                    rm["stated_dims_ft"] = [got["width"], got["depth"]]
                    rm["stated_sqft"] = round(got["width"] * got["depth"], 1)
                    rm["agreement_gap_pct"] = round(
                        abs(sqft - rm["stated_sqft"]) / rm["stated_sqft"] * 100, 1)
                elif got["width"] or got["depth"]:
                    notes.append("only one side matches a printed length")
                else:
                    notes.append("no printed length matches this region")
            else:
                notes.append("irregular outline, not compared with printed lengths")
        if sqft < ruleset.rule("matching.min_room_sqft"):
            verified = False
            notes.append("region too small to be a room")
        if auto:
            verified = False
            rm["found_by"] = "shape only"
            notes.insert(0, "no room name was read for this space; found by shape only")
        rm["needs_review"] = not verified
        if not verified:
            rm["review_reason"] = "; ".join(notes) or "unverified"
        if trace is not None:
            rm["trace_ref"] = "%s#room%d" % (trace.id, rm["id"])
            ent = {"room": rm["id"], "label": name, "seed": [round(sx), round(sy)],
                   "area_sqft": rm["area_sqft"], "bbox": bbox, "solidity": rm["solidity"],
                   "measured_ft": rm["measured_ft"],
                   "printed_pair_in_region": list(pair) if pair else None,
                   "verified": verified, "reasons": notes,
                   "ocr_strings_kept": len(raw)}
            # whatever the chains gave this room, so a trace and the viewer tell the
            # same story about the same run
            for k in ("chain_sides", "chain_refs", "chain_partial", "match_log", "stated_dims_ft",
                      "stated_sqft", "dims_source", "agreement_gap_pct", "side_errors_pct"):
                if k in rm:
                    ent[k] = rm[k]
            if not pair and solidity >= 0.75 and trace_considered is not None:
                ent["aligned_printed_lengths"] = trace_considered
                ent["tolerance"] = 0.08
            trace.add("rooms", ent)
        rooms.append(rm)
    share_across_bands(rooms, ppf)
    return rooms


# What a room carries for its plan when no plans were worked out for the sheet.
_NO_PLANS = "no plans found"


def share_across_bands(rooms, ppf, tol_ft=None, gap_tol=10.0):
    """Give a room the dimension its neighbours in the same band already have.

    Two rooms between the same pair of walls have the same depth, whatever else
    differs about them; two rooms in the same column have the same width. That is
    not a guess, it is what being in the same band means.

    It matters because a draughtsman writes each dimension once. The bath on the
    sample sheet has its width written in it and no depth, and the bedroom beside
    it has the depth written in it and no width. Separately neither can be
    confirmed. Together both can, because they span the same two walls.

    A borrowed value is recorded as borrowed, with the room it came from.
    """
    tol_ft = ruleset.rule("matching.band_tol_ft") if tol_ft is None else tol_ft
    tol = tol_ft * ppf
    # A width is shared down a column, a depth along a band. Which direction the
    # neighbour lies in does not matter: what matters is that the two rooms run
    # between the same pair of walls on that axis. Bed Room A can take Bed Room B's
    # width only if both reach from the same left wall to the same right wall; if
    # their extents differ, their widths differ, and borrowing would be wrong.
    known = {"width": [], "depth": []}
    # Only a value found on the room's own walls is lent. The name rule answers for
    # rooms whose outline is wrong; lending such a value would give a neighbour a
    # number nobody checked against the neighbour's walls.
    rank = {"chain joints": 1, "in the room": 2, "at the name": 3}
    lend_max = ruleset.rule("matching.lend_max_rule")
    for r in rooms:
        sides = r.get("chain_sides") or {}
        for side in ("width", "depth"):
            if sides.get(side):
                how = (r.get("chain_rules") or {}).get(side)
                if how is not None and rank.get(how, 99) > lend_max:
                    continue
                origin = "%s (%s, %s)" % (r["label"], how or "unrecorded",
                                          (r.get("chain_side_refs") or {}).get(side, "?"))
                known[side].append((_extent(r["bbox"], side), sides[side], r["label"], origin,
                                    r.get("plan", _NO_PLANS)))

    for r in rooms:
        sides = r.get("chain_sides")
        if not sides or (sides.get("width") and sides.get("depth")):
            continue
        for side in ("width", "depth"):
            if sides.get(side):
                continue
            extent = _extent(r["bbox"], side)
            # A sheet can carry two plans side by side, and a room on one spans the same
            # band as rooms on the other without sharing a wall with them: BED ROOM C took
            # the first floor's WEST PARLOR depth that way. Only a room on the same plan
            # lends; a room not clearly on one plan neither lends nor borrows. With no
            # plans known at all (a bare chain file) every room counts as on one plan.
            here = r.get("plan", _NO_PLANS)
            for other, ft, label, origin, there in known[side]:
                if there != here or here is None:
                    continue
                if abs(other[0] - extent[0]) <= tol and abs(other[1] - extent[1]) <= tol:
                    sides[side] = ft
                    r.setdefault("chain_borrowed", {})[side] = label
                    r.setdefault("chain_borrowed_from", {})[side] = origin
                    break
        if not (sides.get("width") and sides.get("depth")):
            continue
        stated = sides["width"] * sides["depth"]
        r["stated_dims_ft"] = [sides["width"], sides["depth"]]
        r["stated_sqft"] = round(stated, 1)
        r["agreement_gap_pct"] = round(abs(r["area_sqft"] - stated) / stated * 100, 1)
        r["dims_source"] = "dimension chain (%s borrowed from %s)" % (
            ", ".join(r["chain_borrowed"]),
            ", ".join(sorted(set(r["chain_borrowed"].values()))))
        ok, why = sides_agree(r["bbox"], ppf, sides["width"], sides["depth"])
        r["side_errors_pct"] = why
        if ok and r["area_sqft"] >= ruleset.rule("matching.min_room_sqft"):
            r["needs_review"] = False
            r["fill_pct"] = round(r["area_sqft"] / stated * 100, 1)
            r.pop("review_reason", None)
        else:
            r["review_reason"] = ("the drawing gives %s x %s but the region measures %s"
                                  % (fmt_ft(sides["width"]), fmt_ft(sides["depth"]),
                                     ", ".join("%s %+.0f%%" % (k, v) for k, v in why.items())))


def sides_agree(bbox, ppf, width_ft, depth_ft, tol=None):
    """Compare the room's width and depth with the printed ones, side by side.

    Not area against width times depth. A room is not a rectangle: the Dining Room
    on the sample sheet has closets and a stair cut out of it, so it fills 293 sq ft
    of a 14'10" x 24'4" envelope. Its width and depth are right to under one percent,
    and comparing areas would call that an 19% disagreement and throw it away.

    The area is still worth knowing and is reported as how much of the envelope the
    room fills, but it does not decide anything.
    """
    tol = ruleset.rule("matching.side_tol_rel") if tol is None else tol
    x0, y0, x1, y1 = bbox
    got_w = (x1 - x0 + 1) / ppf
    got_d = (y1 - y0 + 1) / ppf
    err = {"width": round((got_w - width_ft) / width_ft * 100, 1),
           "depth": round((got_d - depth_ft) / depth_ft * 100, 1)}
    return (abs(err["width"]) <= tol * 100 and abs(err["depth"]) <= tol * 100), err


def walls_seen_from(lines, seed, ppf):
    """The wall a room's label sees on each side, in working pixels, or None.

    A fan of rays leaves the label left, right, up and down, each stopping at the
    first wall-map pixel. One ray can slip through a doorway or stop on a stair line,
    so the median of the fan is taken as the wall. Erased dimension lines are not in
    the wall map, so they are never seen.
    """
    h, w = lines.shape
    sx, sy = int(round(seed[0])), int(round(seed[1]))
    fan, step = ruleset.rule("matching.leak_fan_ft"), ruleset.rule("matching.leak_fan_step_ft")
    offsets = [int(round(o * ppf)) for o in np.arange(-fan, fan + 1e-9, step)]
    out = {}
    for side, (dx, dy) in (("left", (-1, 0)), ("right", (1, 0)), ("top", (0, -1)), ("bottom", (0, 1))):
        hits = []
        for o in offsets:
            x, y = (sx, sy + o) if dx else (sx + o, sy)
            if not (0 <= x < w and 0 <= y < h) or lines[y, x]:
                continue
            while 0 <= x < w and 0 <= y < h and not lines[y, x]:
                x, y = x + dx, y + dy
            if 0 <= x < w and 0 <= y < h:
                hits.append(x if dx else y)
        out[side] = int(np.median(hits)) if hits else None
    return out


def refuse_leaks(rooms, seeds, lines, ppf, trace=None):
    """Mark every room whose region reaches past its own walls, so it cannot agree.

    A region bounded by its walls stops at them. One that reaches further than a wall
    is thick has gone round a wall end, through a doorway, into the next space, and its
    width or depth is then not the room's however well it matches a printed size: WEST
    PARLOR read 15'3" x 19'4" correctly and agreed while its region ran 1.20 ft past its
    east wall. The walls are the ones the room's own label sees in the wall map the
    regions were grown in, so no truth is needed.
    """
    margin = ruleset.rule("matching.leak_margin_ft")
    entries = {e["room"]: e for e in (trace.data.get("rooms") or [])} if trace is not None else {}
    for rm in rooms:
        if not rm.get("area_px"):
            continue
        _, sx, sy = seeds[rm["id"] - 1]
        walls = walls_seen_from(lines, (sx, sy), ppf)
        x0, y0, x1, y1 = rm["bbox"]
        past = {"left": walls["left"] - x0 if walls["left"] is not None else None,
                "right": x1 - walls["right"] if walls["right"] is not None else None,
                "top": walls["top"] - y0 if walls["top"] is not None else None,
                "bottom": y1 - walls["bottom"] if walls["bottom"] is not None else None}
        leak = {side: round(px / ppf, 2) for side, px in past.items()
                if px is not None and px / ppf > margin}
        if not leak:
            continue
        rm["region_leak_ft"] = leak
        why = ("the region reaches past this room's own walls (%s), so it has run through "
               "an opening into the next space and its size is not the room's"
               % ", ".join("%s %.2f ft" % kv for kv in sorted(leak.items())))
        rm["needs_review"] = True
        rm["review_reason"] = why + ("; " + rm["review_reason"] if rm.get("review_reason") else "")
        ent = entries.get(rm["id"])
        if ent is not None:
            ent["verified"] = False
            ent["region_leak_ft"] = leak
            ent.setdefault("reasons", []).insert(0, why)


def _extent(bbox, side):
    """The pair of walls a room runs between on the axis that side measures."""
    x0, y0, x1, y1 = bbox
    return (x0, x1) if side == "width" else (y0, y1)


def fmt_ft(v):
    n = round(v * 12)
    return "%d'%d\"" % (n // 12, n % 12)


def plan_summary(rooms):
    total = len(rooms)
    ok = sum(1 for r in rooms if not r["needs_review"])
    if total == 0:
        banner = "No room names from the lexicon were recognised on this sheet."
    else:
        banner = ("Experimental label-seeded mode. Each region is grown from a room name and is "
                  "approximate. %d of %d rooms agree with a printed size; the rest need a human check."
                  % (ok, total))
    return {"geometry_reliable": total > 0 and ok * 2 >= total, "reasons": [], "banner": banner}


def draw_overlay(bgr, lab, ext, rooms, seeds, ppf):
    out = bgr.copy()
    layer = out.copy()
    rng = np.random.default_rng(5)
    for rm in rooms:
        color = tuple(int(c) for c in rng.integers(60, 230, 3))
        m = lab == rm["id"]
        layer[m] = color
    out = cv2.addWeighted(layer, 0.5, out, 0.5, 0)
    for rm in rooms:
        x, y = rm["centroid"]
        tag = "ok" if not rm["needs_review"] else "?"
        txt = "#%d %.0f sf %s" % (rm["id"], rm["area_sqft"], tag)
        cv2.putText(out, txt, (max(0, x - 40), y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (max(0, x - 40), y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def read_plan(img, ppf, fast=False, vertical=True, lexicon=None, workers=None, log=print,
              florence_run=None, tesseract=True, trace=None, sam=None,
              dimension_reader=True, layout=None, layout_conf=0.25, chain_data=None,
              line_geometry=None):
    """Full seeded read of one image. Returns (rooms, lab, ext, work_img, work_ppf, seeds).
    trace: a trace.Trace to record every stage and decision. sam: a SAM 2 predictor for the
    optional second opinion (see sam_check.py); its masks are left in rooms as sam_* fields.
    layout: a YOLO layout model (see layout_check.py); its boxes are left in read_plan.last_layout.
    chain_data: what dimlines.py wrote for this plan. When a room's edges land on the joints of a
    dimension chain, its size is printed on the drawing instead of measured off pixels, and that
    takes priority over everything else."""
    import time as _t
    def timed(name, fn, **info):
        t = _t.time()
        r = fn()
        if trace is not None:
            rec = {"stage": name, "seconds": round(_t.time() - t, 2), **info}
            if isinstance(r, list):
                rec["items"] = len(r)
            trace.data["stages"].append(rec)
        return r

    h, w = img.shape[:2]
    clean = text_layer(img, ppf)
    raw_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keepl = text_layer(img, ppf, keep_crossings=True)
    passes = [(clean, 11), (raw_gray, 11), (keepl, 11)]
    def read(g, psm):
        out = []
        if tesseract:
            out += ocr_items(g, ppf, psm=psm, workers=workers)
        if florence_run is not None:
            out += florence_items(g, ppf, florence_run)
        return out

    names = ["clean", "raw", "crossings-kept"]
    items = []
    for (g, psm), nm in zip(passes, names):
        items += timed("ocr horizontal", lambda g=g, psm=psm: read(g, psm), layer=nm, psm=psm,
                       engines=("tesseract " if tesseract else "") + ("florence" if florence_run else ""))
    if not fast and tesseract:
        # extra Tesseract-only passes: other page-segmentation modes catch different text
        for psm in (12, 6):
            items += timed("ocr horizontal", lambda psm=psm: ocr_items(clean, ppf, psm=psm, workers=workers),
                           layer="clean", psm=psm, engines="tesseract")
    log("  read %d text lines horizontally" % len(items))
    v_items = []
    if vertical:
        for lname, layer in (("clean", clean), ("crossings-kept", keepl)):
            for k in (1, -1):
                r = np.ascontiguousarray(np.rot90(layer, k))
                v_items += timed("ocr rotated", lambda r=r, k=k: unrotate(read(r, 11), k, img.shape),
                                 layer=lname, rotation=("ccw" if k == 1 else "cw"))
        log("  read %d text lines in rotated passes" % len(v_items))
    if dimension_reader and tesseract is not None:
        if vertical:
            v_items += timed("dimension strings (vertical)",
                             lambda: read_dimension_strings(keepl, ppf, "v", workers))
        if not fast:
            items += timed("dimension strings (horizontal)",
                           lambda: read_dimension_strings(keepl, ppf, "h", workers))
    read_plan.last_layout = None
    block_px = []
    if layout is not None:
        import layout_check
        dets = timed("layout detector", lambda: layout_check.detect(layout, img, conf=layout_conf))
        read_plan.last_layout = dets
        block_px = layout_check.block_boxes(dets)
        areas = layout_check.area_boxes(dets)
        items, dropped_i = layout_check.split_items(items, block_px)
        v_items, dropped_v = layout_check.split_items(v_items, block_px)
        lost = sorted({read_name(it["text"], lexicon) for it in dropped_i + dropped_v
                       if read_name(it["text"], lexicon)})
        log("  layout: %d box(es) (%s); %d text line(s) set aside" % (
            len(dets), ", ".join("%s %.2f" % (d["cls"], d["conf"]) for d in dets) or "none",
            len(dropped_i) + len(dropped_v)))
        if trace is not None:
            for d in dets:
                trace.add("layout", d)
            trace.data["params"]["layout_text_set_aside"] = len(dropped_i) + len(dropped_v)
            if not dets:
                trace.warn("layout detector found no boxes; nothing was set aside")
            if lost:
                trace.warn("room-name text inside layout boxes was set aside: %s. If a real room is "
                           "listed here, the detector box is wrong." % ", ".join(lost))
            if block_px and areas:
                big = max(areas, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                for b in block_px:
                    ix = max(0.0, min(b[2], big[2]) - max(b[0], big[0]))
                    iy = max(0.0, min(b[3], big[3]) - max(b[1], big[1]))
                    if (ix * iy) > 0.5 * (b[2] - b[0]) * (b[3] - b[1]):
                        trace.warn("a title/legend/table box sits mostly inside the drawing area; "
                                   "check the layout overlay")
                        break
    schedules = find_schedules(items, ppf)
    if trace is not None:
        for sc in schedules:
            trace.add("schedules", sc)
    seeds = find_seeds(items, ppf, lexicon, trace=trace)
    if layout is not None and trace is not None:
        areas = layout_check.area_boxes(read_plan.last_layout or [])
        if areas:
            for nm, x, y in seeds:
                if not any(layout_check.inside(a, x, y) for a in areas):
                    trace.warn("room label '%s' at (%d, %d) is outside every drawing-area box" % (nm, x, y))
    log("  %d room labels recognised: %s" % (len(seeds), ", ".join(s[0] for s in seeds)))
    chains = []
    dims = collect_dims(items, v_items, skipped=chains)
    if chains:
        log("  %d OCR line(s) held a whole dimension chain and were not used for alignment" % len(chains))
    if trace is not None:
        if chains:
            trace.data["params"]["dimension_chain_strings_skipped"] = len(chains)
            trace.warn("%d OCR line(s) held 4 or more lengths read as one string (a dimension chain); "
                       "their lengths have no usable position and were not used. Example: %r"
                       % (len(chains), chains[0]["text"]))
        for d in dims:
            trace.add("dims", {"ft": d["ft"], "orient": d["orient"], "x": round(d["cx"]),
                               "y": round(d["cy"]), "ocr_text": d["text"]})
        if not any(d["orient"] == "v" for d in dims):
            trace.warn("no vertical printed lengths were read, so no room can be confirmed on depth")
        if not dims:
            trace.warn("no printed lengths were read at all; consider a higher-resolution source")
    log("  %d printed lengths found (%d horizontal, %d vertical)" % (
        len(dims), sum(d["orient"] == "h" for d in dims), sum(d["orient"] == "v" for d in dims)))

    s = min(1.0, WORK_PPF / ppf)
    if s < 1.0:
        wimg = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        wimg = img
    wppf = ppf * s
    wseeds = [(n, x * s, y * s) for n, x, y in seeds]
    wdims = [dict(d, cx=d["cx"] * s, cy=d["cy"] * s) for d in dims]
    def in_schedule(it):
        return any(b[0] <= it["cx"] <= b[2] and b[1] <= it["cy"] <= b[3] for b in (sc["bbox"] for sc in schedules))

    witems = [dict(it, cx=it["cx"] * s, cy=it["cy"] * s) for it in items if not in_schedule(it)]
    if not wseeds:
        return [], np.zeros(wimg.shape[:2], np.int32), 1, wimg, wppf, []
    wgray = cv2.cvtColor(wimg, cv2.COLOR_BGR2GRAY)
    wblocks = [tuple(v * s for v in b) for b in block_px]
    # Lines a believed reading proves are dimensions, in working pixels. With no geometry
    # file this stays None and segmentation is exactly what it was.
    werase = None
    if line_geometry:
        werase = [((x0 * s, y0 * s), (x1 * s, y1 * s))
                  for x0, y0, x1, y1 in (st["line"] for st in line_geometry.get("stretches") or [])]
    if trace is not None:
        trace.data["params"]["dimension_lines_removed"] = (
            len(werase) if werase is not None else "no line geometry for this plan")
    lab, ext = timed("segment", lambda: segment(wgray, wppf, [(x, y) for _, x, y in wseeds], bg_boxes=wblocks, erase=werase),
                     seeds=len(wseeds), working_ppf=round(wppf, 2))
    wlines = line_map(wgray, wppf, erase=werase)
    rooms = build_rooms(lab, wseeds, wppf, witems, wdims, trace=trace,
                        chains=chain_lookup(chain_data, img.shape[0], s, wppf, trace,
                                            lab=lab, ext=ext,
                                            lines=wlines))
    refuse_leaks(rooms, wseeds, wlines, wppf, trace)
    if sam is not None:
        import sam_check
        sms = timed("sam2 masks", lambda: sam_check.sam_masks(sam, wimg, [(x, y) for _, x, y in wseeds], wppf))
        wms = [region_mask(lab, i, (x, y), wppf) for i, (_, x, y) in enumerate(wseeds, start=1)]
        sam_check.compare(rooms, wms, sms, wppf)
        read_plan.last_sam_masks = sms
        if trace is not None:
            for rm in rooms:
                trace.data["rooms"][rm["id"] - 1]["sam_iou"] = rm.get("sam_iou")
                trace.data["rooms"][rm["id"] - 1]["sam_area_sqft"] = rm.get("sam_area_sqft")
    return rooms, lab, ext, wimg, wppf, wseeds


def load_manifest(path):
    scales = {}
    with open(path, newline="") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            name, val = (row.get("plan") or "").strip(), (row.get("pixels_per_foot") or "").strip()
            try:
                ppf = float(val)
            except ValueError:
                raise SystemExit("%s line %d: pixels_per_foot must be a plain number, got %r" % (path, n, val))
            if not name or not (0.5 <= ppf <= 500):
                raise SystemExit("%s line %d: need a plan file name and a pixels_per_foot between 0.5 and 500"
                                 % (path, n))
            scales[name] = ppf
    return scales


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", default="plans")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out", default="viewer")
    ap.add_argument("--only", help="run just this plan file name")
    ap.add_argument("--fast", action="store_true", help="two OCR passes instead of four")
    ap.add_argument("--no-vertical", action="store_true", help="skip the rotated OCR passes")
    ap.add_argument("--engine", choices=["tesseract", "florence", "both"], default="tesseract",
                    help="text reader. florence uses the same local model as read_labels.py (slower); "
                         "both pools the two readers (slowest, reads the most)")
    ap.add_argument("--trust-remote-code", action="store_true", help="needed for --engine florence")
    ap.add_argument("--revision", default=None, help="pin the Florence-2 model revision you reviewed")
    ap.add_argument("--sam2", action="store_true",
                    help="optional second opinion from Segment Anything 2 (see sam_check.py)")
    ap.add_argument("--sam2-model", default=None, help="model id, default facebook/sam2-hiera-tiny")
    ap.add_argument("--layout", nargs="?", const="default", default=None, metavar="WEIGHTS",
                    help="optional YOLO layout splitter (see layout_check.py): a local .pt file or a "
                         "Hugging Face repo id[:filename]; with no value, the default repo id")
    ap.add_argument("--layout-revision", default=None, help="pin the Hugging Face commit you reviewed")
    ap.add_argument("--layout-conf", type=float, default=0.25, help="detector confidence cutoff")
    ap.add_argument("--explain", action="store_true",
                    help="after each plan, print what every room tried to match against "
                         "and why each rule did not answer")
    ap.add_argument("--no-dimension-reader", action="store_true",
                    help="skip the dedicated dimension-string reader")
    ap.add_argument("--group", default=None, metavar="ID",
                    help="group id of this run, the middle part of every trace id. Default: GROUP_ID from "
                         "the environment (run_layers.sh sets it once per run), else the time now")
    ap.add_argument("--lexicon-extra", default="", help="comma-separated extra room words, e.g. OFFICE,SCULLERY")
    ap.add_argument("--use-drawing-scale", action="store_true",
                    help="measure with the scale the dimension layer derived from the drawing's "
                         "readings instead of the manifest's. Off by default: it is always "
                         "reported, but on the sample sheet it has been 2 to 6 percent low")
    ap.add_argument("--trust-manifest", action="store_true",
                    help="keep the manifest scale (the default); wins over --use-drawing-scale")
    args = ap.parse_args()

    lex = None
    if args.lexicon_extra:
        lex = LEXICON + [w.strip().upper() for w in args.lexicon_extra.split(",") if w.strip()]

    frun = None
    if args.engine in ("florence", "both"):
        from read_labels import ocr_model
        frun = ocr_model(args.trust_remote_code, args.revision)

    sam = None
    if args.sam2:
        import sam_check
        sam = sam_check.load_sam2(args.sam2_model or sam_check.DEFAULT_MODEL)

    scales = load_manifest(args.manifest)
    layout = None
    if args.layout:
        import layout_check
        layout = layout_check.load_layout(layout_check.DEFAULT_MODEL if args.layout == "default"
                                          else args.layout, args.layout_revision)

    out_dir = Path(args.out)
    (out_dir / "out").mkdir(parents=True, exist_ok=True)
    try:
        group = new_group(args.group)
    except ValueError as e:
        sys.exit(str(e))
    print("group id: %s (the start of every trace id in this run)" % group)
    tech = {"text_reader": {"tesseract": "tesseract", "florence": "florence-2",
                            "both": "tesseract + florence-2"}[args.engine],
            "layout_splitter_yolo": ("on (%s)" % (args.layout if args.layout != "default" else "default weights"))
            if args.layout else "off",
            "sam2_second_opinion": "on" if args.sam2 else "off",
            "dimension_reader": "off" if args.no_dimension_reader else "on"}
    print("technology: " + "; ".join("%s=%s" % kv for kv in tech.items()))
    results = {"method": "label-seeded regions + tiled OCR (%s, experimental)" % args.engine, "plans": []}
    stamp(results, "seeded", group)
    results["technology"] = tech
    for name, ppf in scales.items():
        if args.only and name != args.only:
            continue
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
        print("%s:" % name)
        trace = Trace(name, {"engine": args.engine, "fast": args.fast, "vertical": not args.no_vertical,
                             "pixels_per_foot_given": ppf, "image_px": [img.shape[1], img.shape[0]],
                             "sheet_ft_at_given_scale": [round(img.shape[1] / ppf, 1), round(img.shape[0] / ppf, 1)],
                             "sam2": bool(sam), "layout": args.layout, "layout_revision": args.layout_revision,
                             "layout_conf": args.layout_conf, "lexicon_extra": args.lexicon_extra}, group=group)
        print("  trace id:", trace.id)
        print("  scale: %.2f px/ft, so this sheet spans about %.0f x %.0f ft. If that is not the real size "
              "of the drawn area, the scale in the manifest is wrong and every result is wrong."
              % (ppf, img.shape[1] / ppf, img.shape[0] / ppf))
        chain_data = dimlines.load_for_plan(name, out_dir / "out")
        # A dimension file that exists but holds nothing is the worst of the three
        # states, because it looks like a normal run. Every matching rule needs
        # chains or lone dimensions, so with neither the whole dimension path is
        # off and only the old text fallback is left. Say so here, loudly, rather
        # than let a run report "0 confirmed" as though matching had been tried.
        if chain_data is None:
            msg = ("no dimension file for this plan: run the dimension layer "
                   "(DIMS=1) or no printed size can be matched to any room")
        elif not (chain_data.get("chains") or chain_data.get("single_dimensions")):
            msg = ("the dimension file for this plan holds no chains and no lone "
                   "dimensions, so nothing can be matched to a room. Rerun the "
                   "dimension layer; a result from this run measures the segmenter "
                   "only, not the matching")
        else:
            msg = None
        if msg:
            print("  dimensions: " + msg)
            trace.warn(msg)
        # Optional: which dimension lines a believed reading proves (dimlines.py
        # --lines-only). Read here rather than required, so a plan without it segments
        # as before.
        line_geo = None
        geo_file = out_dir / "out" / (Path(name).stem + dimlines.LINES_SUFFIX + ".json")
        if geo_file.exists():
            try:
                line_geo = json.loads(geo_file.read_text())
            except ValueError:
                trace.warn("line geometry file %s is not readable JSON; not used" % geo_file.name)
            if line_geo is not None:
                print("  line geometry: %d proven dimension line(s) kept out of the walls"
                      % len(line_geo.get("stretches") or []))
        if chain_data and chain_data.get("pixels_per_foot_from_drawing"):
            ppf, note = dimlines.choose_scale(
                ppf, float(chain_data["pixels_per_foot_from_drawing"]),
                chain_data.get("readings_used_for_scale") or 0,
                use_drawing=args.use_drawing_scale and not args.trust_manifest)
            if note:
                print("  scale: " + note)
                trace.warn(note)
        # The scale actually measured with, so a scorer never has to infer it from a note.
        trace.data["params"]["pixels_per_foot_used"] = ppf
        rooms, lab, ext, wimg, wppf, wseeds = read_plan(
            img, ppf, fast=args.fast, vertical=not args.no_vertical, lexicon=lex,
            florence_run=frun, tesseract=args.engine != "florence", trace=trace, sam=sam,
            dimension_reader=not args.no_dimension_reader,
            layout=layout, layout_conf=args.layout_conf,
            chain_data=chain_data, line_geometry=line_geo)
        overlay = draw_overlay(wimg, lab, ext, rooms, wseeds, wppf)
        out_name = Path(name).stem + "_seeded_overlay.png"
        cv2.imwrite(str(out_dir / "out" / out_name), overlay)
        entry = {
            "plan": name, "pixels_per_foot": round(wppf, 3), "overlay": "out/" + out_name,
            "quality": plan_summary(rooms), "rooms": rooms,
        }
        if sam is not None and getattr(read_plan, "last_sam_masks", None) is not None:
            import sam_check
            sname = Path(name).stem + "_sam_overlay.png"
            cv2.imwrite(str(out_dir / "out" / sname),
                        sam_check.draw_sam_overlay(wimg, read_plan.last_sam_masks, rooms))
            entry["sam_overlay"] = "out/" + sname
        if layout is not None and getattr(read_plan, "last_layout", None) is not None:
            import layout_check
            lname = Path(name).stem + "_layout_overlay.png"
            cv2.imwrite(str(out_dir / "out" / lname),
                        layout_check.draw_layout_overlay(img, read_plan.last_layout))
            entry["layout_overlay"] = "out/" + lname
        tfile = out_dir / ("trace_%s.json" % trace.id)
        trace.write(tfile)
        if trace.data["schedules"]:
            entry["schedules"] = trace.data["schedules"]
        entry["group_id"] = group
        entry["trace_id"] = trace.id
        entry["trace_file"] = tfile.name
        results["plans"].append(entry)
        print("  trace written:", tfile)
        ok = sum(1 for r in rooms if not r["needs_review"])
        print("  %d rooms, %d confirmed by printed sizes, %d need a human check" % (
            len(rooms), ok, len(rooms) - ok))
        if args.explain:
            print("  -- what each room tried to match --")
            for r in rooms:
                print("  %s  (%s x %s measured)"
                      % (r["label"], fmt_ft(r["measured_ft"][0]), fmt_ft(r["measured_ft"][1])))
                for line in (r.get("match_log") or ["no dimension data for this plan"]):
                    print("      %s" % line)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    index = {"group_id": group, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "engine": args.engine,
             "layout": args.layout, "plans": [
                 {"plan": e["plan"], "trace_id": e["trace_id"], "trace_file": e["trace_file"],
                  "pixels_per_foot": e["pixels_per_foot"], "rooms": len(e["rooms"]),
                  "confirmed": sum(1 for r in e["rooms"] if not r["needs_review"])}
                 for e in results["plans"]]}
    with open(out_dir / ("group_%s.json" % group), "w") as f:
        json.dump(index, f, indent=2)
    print("group %s: %d plan(s); index written to %s" % (group, len(results["plans"]),
                                                         out_dir / ("group_%s.json" % group)))


if __name__ == "__main__":
    main()
