"""
Score a run against the printed sizes in truth.csv.

    python score.py                                   # viewer/results.json
    python score.py --results <out>/results.json
    python score.py --trace viewer/trace_<id>.json    # a trace on its own
    python score.py --json summary.json               # also write the numbers

truth.csv has one row per room on the drawing:

    plan,room,printed_w_ft,printed_d_ft,w_source,d_source,notes,box_x0,box_y0,box_x1,box_y1

w_source and d_source are "printed" (read off the sheet), "band" (shared from a
neighbour between the same walls) or "none" (not printed). The box is where the room
is on the full-size sheet, in pixels. A region belongs to the room whose box holds its
seed, the point the region was grown from. The label the pipeline read is never used
for this: a misread label must not lose a room.

Printed for each plan: rooms found, region accuracy per side (printed and band kept
apart), invented regions, and agreement split three ways. A false agreement, a room
marked as agreeing with the drawing when it does not, makes the exit status 1. That is
the one defect this project must never ship. Reads only; it writes nothing unless
--json is given.
"""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import ruleset

SOURCES = ("printed", "band", "none")
# A stated size that came off the drawing should equal the printed one to within
# rounding. One inch is the drawing's own precision.
STATED_SLACK_FT = 1.0 / 12 + 0.005


def load_truth(path):
    rows = []
    with open(path, newline="") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            for side in ("w", "d"):
                src = (row.get(side + "_source") or "").strip()
                if src not in SOURCES:
                    raise ValueError("%s line %d: %s_source must be one of %s, not %r"
                                     % (path, n, side, "/".join(SOURCES), src))
                val = (row.get("printed_%s_ft" % side) or "").strip()
                if (src == "none") != (val == ""):
                    raise ValueError("%s line %d: %s_source %s does not fit value %r"
                                     % (path, n, side, src, val))
            rows.append({
                "plan": row["plan"].strip(), "room": row["room"].strip(),
                "w": float(row["printed_w_ft"]) if row["printed_w_ft"].strip() else None,
                "d": float(row["printed_d_ft"]) if row["printed_d_ft"].strip() else None,
                "w_source": row["w_source"].strip(), "d_source": row["d_source"].strip(),
                "box": [float(row[k]) for k in ("box_x0", "box_y0", "box_x1", "box_y1")],
            })
    return rows


def sheet_ppf(trace):
    """The scale the run measured the full sheet with."""
    if trace["params"].get("pixels_per_foot_used"):
        return float(trace["params"]["pixels_per_foot_used"])
    ppf = float(trace["params"]["pixels_per_foot_given"])
    # Older runs said so only in a warning when the dimension layer overruled the manifest.
    for w in trace.get("warnings") or []:
        m = re.search(r"the drawing's own dimensions say ([0-9.]+)", w)
        if m:
            ppf = float(m.group(1))
    return ppf


def full_scale(trace):
    """Factor from the working image the regions were grown on to the full sheet."""
    ppf = sheet_ppf(trace)
    wppf = next((st["working_ppf"] for st in trace.get("stages") or []
                 if st.get("stage") == "segment" and st.get("working_ppf")), None)
    if not wppf:
        raise ValueError("trace %s has no segment stage with working_ppf"
                         % trace.get("trace_id"))
    return ppf / float(wppf)


def regions_from(trace, results_rooms=None):
    """One record per region. Agreement comes from results.json when there is one,
    because band sharing runs after the trace entry is written."""
    by_id = {r["id"]: r for r in results_rooms or []}
    k = full_scale(trace)
    out = []
    for ent in trace.get("rooms") or []:
        rm = by_id.get(ent["room"], {})
        agreed = (not rm["needs_review"]) if "needs_review" in rm else bool(ent.get("verified"))
        out.append({
            "id": ent["room"], "label": ent.get("label") or "?",
            "seed": [ent["seed"][0] * k, ent["seed"][1] * k],
            "measured": ent.get("measured_ft") or rm.get("measured_ft"),
            "area": ent.get("area_sqft", 0.0),
            "stated": rm.get("stated_dims_ft", ent.get("stated_dims_ft")),
            "agreed": agreed,
            "box": ([ent["bbox"][0] * k, ent["bbox"][1] * k,
                     (ent["bbox"][2] + 1) * k, (ent["bbox"][3] + 1) * k]
                    if ent.get("bbox") else None),
        })
    return out


def containment(r, ppf):
    """How far a region reaches past its own room's box, side by side, in feet.

    The boxes sit on the inner face of each room's walls, so a region that goes further
    than a wall is thick has gone through a wall or a door into the next space. That is
    invisible to the size check whenever the side it broke through carries no printed
    length: the attic that merged with the roof scored better, not worse.
    """
    t = r["truth"]
    if t is None or not r.get("box"):
        return {}
    x0, y0, x1, y1 = r["box"]
    b = t["box"]
    reach = {"left": b[0] - x0, "top": b[1] - y0, "right": x1 - b[2], "bottom": y1 - b[3]}
    margin = ruleset.rule("scoring.containment_margin_ft")
    # The margin is one wall's thickness, also where two walls stand back to back with a
    # cavity between them. The region's own wall face is a barrier in the mask, so a
    # region in the cavity got there through an opening, and its measured side then
    # carries a wall and the cavity: it is outside its room.
    return {side: round(px / ppf, 2) for side, px in reach.items() if px / ppf > margin}


def assign(regions, truth):
    """Map each region to the truth room whose box holds its seed; the smallest box
    wins where boxes touch."""
    for r in regions:
        x, y = r["seed"]
        hits = [t for t in truth if t["box"][0] <= x <= t["box"][2] and t["box"][1] <= y <= t["box"][3]]
        hits.sort(key=lambda t: (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]))
        r["truth"] = hits[0] if hits else None


def judge_agreement(r, tol):
    """Why an agreeing region is wrong, or None when it is right."""
    t = r["truth"]
    if t is None:
        return "agreed on a region that is no room on the drawing"
    # A size that agrees on a region reaching through a wall agrees by coincidence: the
    # area it would hand on is not the room's.
    if r.get("outside"):
        return "region reaches outside the room: %s" % ", ".join(
            "%s %.2f ft" % kv for kv in sorted(r["outside"].items()))
    for i, side in enumerate(("w", "d")):
        if t[side] is None:
            return "the drawing prints no %s for this room" % ("width", "depth")[i]
        if r["stated"] and abs(r["stated"][i] - t[side]) > STATED_SLACK_FT:
            return "stated %s %.2f ft, printed %.2f ft" % (("width", "depth")[i], r["stated"][i], t[side])
        m = r["measured"][i]
        if abs(m - t[side]) / t[side] > tol:
            return "measured %s %.2f ft, printed %.2f ft" % (("width", "depth")[i], m, t[side])
    return None


def score_plan(plan, regions, truth, tol, ppf=None):
    assign(regions, truth)
    for r in regions:
        r["outside"] = containment(r, ppf) if ppf else {}
    by_room = {t["room"]: [r for r in regions if r["truth"] is t] for t in truth}
    found = [t["room"] for t in truth if by_room[t["room"]]]
    primary = {name: max(rs, key=lambda r: r["area"]) for name, rs in by_room.items() if rs}

    errs = {"printed": {"width": [], "depth": []}, "band": {"width": [], "depth": []}}
    within = {tol: 0, 0.04: 0}
    scored_rooms = 0
    within_contained = []
    for t in truth:
        r = primary.get(t["room"])
        if r is None or not r["measured"]:
            continue
        sides = []
        for i, (side, name) in enumerate((("w", "width"), ("d", "depth"))):
            src = t[side + "_source"]
            if src in errs:
                e = abs(r["measured"][i] - t[side]) / t[side]
                errs[src][name].append((e, t["room"]))
                if src == "printed":
                    sides.append(e)
        if sides:
            scored_rooms += 1
            for lim in within:
                within[lim] += all(e <= lim for e in sides)
            if all(e <= tol for e in sides) and not r["outside"]:
                within_contained.append(t["room"])

    def summary(pairs):
        if not pairs:
            return None
        worst = max(pairs)
        return {"n": len(pairs), "median_pct": round(statistics.median(e for e, _ in pairs) * 100, 1),
                "worst_pct": round(worst[0] * 100, 1), "worst_room": worst[1]}

    correct, wrong, not_agreed = [], [], []
    for r in regions:
        if not r["agreed"]:
            not_agreed.append(r)
            continue
        why = judge_agreement(r, tol)
        (wrong if why else correct).append((r, why))

    return {
        "plan": plan,
        "rooms_on_drawing": len(truth),
        "rooms_found": len(found),
        "rooms_missing": [t["room"] for t in truth if not by_room[t["room"]]],
        "split_rooms": {n: [r["label"] for r in rs] for n, rs in by_room.items() if len(rs) > 1},
        "accuracy": {src: {side: summary(v) for side, v in sides.items()} for src, sides in errs.items()},
        "rooms_with_a_printed_side": scored_rooms,
        "regions_within_tol": {str(k): v for k, v in within.items()},
        "invented": [{"id": r["id"], "label": r["label"], "seed_px": [round(v) for v in r["seed"]],
                      "measured_ft": r["measured"]} for r in regions if r["truth"] is None],
        "agreed_correct": [{"label": r["label"], "room": r["truth"]["room"]} for r, _ in correct],
        "agreed_wrong": [{"label": r["label"], "room": r["truth"]["room"] if r["truth"] else None,
                          "why": why} for r, why in wrong],
        "not_agreed": len(not_agreed),
        "regions": [{"id": r["id"], "label": r["label"],
                     "room": r["truth"]["room"] if r["truth"] else None,
                     "measured_ft": r["measured"], "agreed": r["agreed"]} for r in regions],
        "outside": [{"label": r["label"], "room": r["truth"]["room"], "reach_ft": r["outside"]}
                    for r in regions if r.get("outside")],
        "within_tol_and_contained": within_contained,
    }


def print_score(s, tol):
    print("plan  %s" % s["plan"])
    print("  rooms found        %d of %d" % (s["rooms_found"], s["rooms_on_drawing"]))
    if s["rooms_missing"]:
        print("    no region:       %s" % ", ".join(s["rooms_missing"]))
    for name, labels in s["split_rooms"].items():
        print("    split:           %s holds %d regions (%s)" % (name, len(labels), ", ".join(labels)))
    for src in ("printed", "band"):
        for side in ("width", "depth"):
            a = s["accuracy"][src][side]
            if a:
                print("  %-7s %-6s     n=%d  median %.1f%%  worst %.1f%% (%s)"
                      % (src, side, a["n"], a["median_pct"], a["worst_pct"], a["worst_room"]))
            else:
                print("  %-7s %-6s     none" % (src, side))
    w = s["regions_within_tol"]
    print("  regions within tolerance on every printed side: %s of %d at %.2f, %s at 0.04"
          % (w[str(tol)], s["rooms_with_a_printed_side"], tol, w["0.04"]))
    print("  invented regions   %d" % len(s["invented"]))
    for r in s["invented"]:
        print("    %s at %s px" % (r["label"], r["seed_px"]))
    print("  outside their room %d" % len(s["outside"]))
    for r in s["outside"]:
        print("    %s (%s): %s" % (r["label"], r["room"], ", ".join(
            "%s %.2f ft" % kv for kv in sorted(r["reach_ft"].items()))))
    print("  within tolerance and inside the room: %d (%s)"
          % (len(s["within_tol_and_contained"]), ", ".join(s["within_tol_and_contained"]) or "none"))
    print("  agreed, correct    %d" % len(s["agreed_correct"]))
    print("  agreed, WRONG      %d" % len(s["agreed_wrong"]))
    for r in s["agreed_wrong"]:
        print("    FALSE AGREEMENT: %s (%s): %s" % (r["label"], r["room"] or "no room", r["why"]))
    print("  not agreed         %d" % s["not_agreed"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="a results.json; its traces are read beside it")
    ap.add_argument("--trace", default=None, help="score a single trace instead")
    ap.add_argument("--truth", default="truth.csv")
    ap.add_argument("--json", default=None, help="also write the numbers to this file")
    a = ap.parse_args(argv)

    tol = ruleset.rule("matching.side_tol_rel")
    truth = load_truth(a.truth)
    runs = []
    if a.trace:
        tr = json.loads(Path(a.trace).read_text())
        runs.append((tr["plan"], tr, None))
    else:
        path = Path(a.results or "viewer/results.json")
        data = json.loads(path.read_text())
        for p in data["plans"]:
            tr = json.loads((path.parent / p["trace_file"]).read_text())
            runs.append((p["plan"], tr, p["rooms"]))

    scores = []
    for plan, tr, rooms in runs:
        rows = [t for t in truth if t["plan"] == plan]
        if not rows:
            print("plan  %s: no rows in %s, not scored" % (plan, a.truth))
            continue
        s = score_plan(plan, regions_from(tr, rooms), rows, tol, ppf=sheet_ppf(tr))
        s["trace_id"] = tr.get("trace_id")
        s["version"] = tr.get("version")
        print_score(s, tol)
        scores.append(s)
    if a.json:
        Path(a.json).write_text(json.dumps(scores, indent=2))
    return 1 if any(s["agreed_wrong"] for s in scores) else 0


if __name__ == "__main__":
    sys.exit(main())
