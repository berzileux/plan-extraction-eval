"""Read a run trace and say what happened, in one screen.

A trace holds everything a layer decided and why, which is the point of it, but it
is several thousand lines of JSON and nobody reads that. This prints the parts you
look at when a run disappoints you.

  python trace.py                          # the newest trace in viewer/
  python trace.py viewer/trace_2026*.json  # one or several
  python trace.py --rooms                  # just the room table
  python trace.py --chains --warnings      # just those sections
  python trace.py --all                    # every section, nothing trimmed

With no section flags it prints a summary, the warnings, the room table and the
chains. Reads only; it changes nothing.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

SECTIONS = ("summary", "warnings", "stages", "seeds", "dims", "rooms", "chains")


def ft(v):
    """Feet as architects write them. None becomes a dash."""
    if v is None:
        return "-"
    n = round(float(v) * 12)
    return "%d'%d\"" % (n // 12, n % 12)


def newest(viewer="viewer"):
    files = sorted(Path(viewer).glob("trace_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit("no trace_*.json in %s/. Run run_layers.sh first." % viewer)
    return [files[-1]]


def section_summary(t):
    print("plan     %s" % t.get("plan", "?"))
    print("trace    %s    group %s    version %s"
          % (t.get("trace_id", "?"), t.get("group_id", "?"), t.get("version", "?")))
    print("started  %s    took %s s" % (t.get("started", "?"), t.get("total_seconds", "?")))
    p = t.get("params") or {}
    if p:
        keep = [(k, v) for k, v in p.items() if not isinstance(v, (dict, list))]
        print("params   " + "  ".join("%s=%s" % (k, v) for k, v in keep))
        for k, v in p.items():
            if isinstance(v, (dict, list)):
                print("         %s=%s" % (k, v))
    counts = [(k, len(v)) for k, v in t.items() if isinstance(v, list) and v]
    if counts:
        print("holds    " + "  ".join("%s=%d" % kv for kv in counts))


def section_warnings(t):
    w = t.get("warnings") or []
    if not w:
        print("none")
        return
    for m in w:
        print("  - %s" % m)


def section_stages(t):
    for st in t.get("stages") or []:
        extra = {k: v for k, v in st.items() if k not in ("stage", "seconds")}
        print("  %6.2f s  %-26s %s" % (st.get("seconds", 0), st.get("stage", "?"),
                                       "  ".join("%s=%s" % kv for kv in extra.items())))


def section_seeds(t):
    for s in t.get("seeds") or []:
        print("  %-22s (%5s,%5s)  %s" % (s.get("name"), s.get("x"), s.get("y"),
                                         s.get("decision", "")))


def section_dims(t):
    rows = t.get("dims") or []
    if not rows:
        print("none read")
        return
    for d in rows:
        print("  %-9s %s  at (%5s,%5s)   %r"
              % (ft(d.get("ft")), d.get("orient"), d.get("x"), d.get("y"),
                 (d.get("ocr_text") or "")[:46]))


def section_rooms(t):
    rooms = t.get("rooms") or []
    if not rooms:
        print("none")
        return
    print("  %-16s %8s %-15s %-15s %-22s %s"
          % ("room", "sq ft", "measured", "printed", "read so far", "verdict"))
    ok = 0
    for r in rooms:
        m = r.get("measured_ft") or []
        meas = "%s x %s" % (ft(m[0]), ft(m[1])) if len(m) == 2 else "-"
        sd = r.get("stated_dims_ft")
        printed = "%s x %s" % (ft(sd[0]), ft(sd[1])) if sd else "-"
        sides = r.get("chain_sides") or {}
        got = [k + " " + ft(v) for k, v in (("w", sides.get("width")),
                                            ("d", sides.get("depth"))) if v]
        partial = ", ".join(got) if (got and not sd) else ("both" if sd else "-")
        good = not r.get("needs_review", True)
        ok += good
        verdict = "ok" if good else (r.get("review_reason") or "needs a human")
        print("  %-16s %8.1f %-15s %-15s %-22s %s"
              % ((r.get("label") or "?")[:16], r.get("area_sqft", 0), meas, printed,
                 partial, verdict[:60]))
    print("\n  %d of %d agree with a printed size" % (ok, len(rooms)))
    if any(r.get("match_log") for r in rooms):
        print("\n  -- what each room tried to match --")
        for r in rooms:
            log = r.get("match_log")
            if not log:
                continue
            print("  %s" % (r.get("label") or "?"))
            for line in log:
                print("      %s" % line)


def section_chains(t):
    chains = t.get("chains") or []
    if not chains:
        print("none (this is not a dimension-layer trace, or no chain survived)")
        return
    for c in chains:
        print("  %s chain at %-6s  span %-9s overall %-9s  read %d, solved %d, unknown %d"
              % (c.get("axis"), c.get("y"), ft(c.get("span_ft")), ft(c.get("overall_ft")),
                 c.get("readings", 0), c.get("solved", 0), c.get("unknown", 0)))
        for i, s in enumerate(c.get("segments") or []):
            print("      seg%02d  %-9s measured %-9s  %-7s %s"
                  % (i, ft(s.get("value_ft")), ft(s.get("measured_ft")),
                     s.get("source") or "-", ("" if not s.get("raw_text")
                                              else repr(s["raw_text"])[:34])))
        for rj in c.get("rejected") or []:
            print("      rejected: %s on a run measuring %s (%s%% out)"
                  % (ft(rj.get("stated_ft")), ft(rj.get("measured_ft")), rj.get("off_pct")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="trace files; default is the newest in viewer/")
    ap.add_argument("--viewer", default="viewer")
    for name in SECTIONS:
        ap.add_argument("--" + name, action="store_true", help="print the %s section" % name)
    ap.add_argument("--all", action="store_true", help="every section")
    a = ap.parse_args(argv)

    want = [s for s in SECTIONS if getattr(a, s)]
    if a.all:
        want = list(SECTIONS)
    if not want:
        want = ["summary", "warnings", "rooms", "chains"]

    paths = []
    for f in a.files or []:
        paths += [Path(p) for p in sorted(glob.glob(f))] or [Path(f)]
    paths = paths or newest(a.viewer)

    for n, path in enumerate(paths):
        if not path.is_file():
            print("not a file, skipped: %s" % path, file=sys.stderr)
            continue
        try:
            t = json.loads(path.read_text())
        except ValueError as exc:
            print("not readable as JSON, skipped: %s (%s)" % (path, exc), file=sys.stderr)
            continue
        if n:
            print()
        print("=" * 78)
        print(path)
        print("=" * 78)
        for s in want:
            print("\n-- %s --" % s)
            globals()["section_" + s](t)


if __name__ == "__main__":
    main()
