"""Pack one run's outputs into a single zip you can send for review.

  python collect-run.py                    # the latest run (newest viewer/group_<id>.json)
  python collect-run.py 20261011-024724    # a specific group id

Contents: the group index, every trace file of the group, results.json, the overlay images of its
plans, the dimension-chain output (<plan>_dimlines.json and its overlay) and manifest.csv.

Only files belonging to this group go in. Traces name the group they came from, and anything from an
earlier run is left out and listed at the end so a stale file is never mistaken for a fresh result.

The plan images themselves are NOT included. Check that the overlays do not show anything you are not
allowed to share before sending the zip.
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", nargs="?", help="group id; default: the newest run in the viewer folder")
    ap.add_argument("--viewer", default="viewer")
    ap.add_argument("--manifest", default="manifest.csv")
    args = ap.parse_args()

    v = Path(args.viewer)
    indexes = sorted(v.glob("group_*.json"), key=lambda p: p.stat().st_mtime)
    if args.group:
        if not SAFE.match(args.group):
            sys.exit("group id may only contain letters, digits, '-', '_' and '.'")
        idx = v / ("group_%s.json" % args.group)
        if not idx.is_file():
            sys.exit("no %s; groups found: %s" % (idx, ", ".join(p.stem[6:] for p in indexes) or "none"))
    elif indexes:
        idx = indexes[-1]
    else:
        sys.exit("no group_<id>.json in %s/. Run seeded.py or run_layers.sh first." % args.viewer)
    info = json.loads(idx.read_text())
    gid = info["group_id"]

    files = {idx.name: idx}
    skipped = []

    def take(arc, path):
        if path.is_file():
            files[arc] = path
        else:
            skipped.append(str(path))

    for p in info["plans"]:
        tf = p["trace_file"]
        if not SAFE.match(tf):
            sys.exit("unexpected trace file name %r" % tf)
        take(tf, v / tf)

    # Every trace this run wrote, including layers the group index does not list,
    # such as the dimension-chain layer. A trace from an earlier run is skipped.
    for t in v.glob("trace_%s-*.json" % gid):
        take(t.name, t)

    # The dimension-chain layer writes per plan, outside results.json. Each file
    # carries its own group id, so a leftover from a previous run is left behind.
    for d in (v / "out").glob("*_dimlines.json"):
        try:
            got = json.loads(d.read_text()).get("group_id")
        except (OSError, ValueError):
            continue
        if got != gid:
            skipped.append("%s (group %s)" % (d, got))
            continue
        take("out__" + d.name, d)
        png = d.with_name(d.name.replace("_dimlines.json", "_dimlines_overlay.png"))
        take("out__" + png.name, png)

    results = v / "results.json"
    if results.is_file():
        rj = json.loads(results.read_text())
        if rj.get("group_id") != gid:
            print("note: results.json belongs to group %s, not %s; it is included anyway, so tell the "
                  "reviewer" % (rj.get("group_id"), gid), file=sys.stderr)
        files["results.json"] = results
        for pl in rj.get("plans", []):
            if pl.get("group_id") != gid:
                continue
            for key in ("overlay", "sam_overlay", "layout_overlay"):
                rel = pl.get(key)
                if rel and ".." not in rel and not rel.startswith("/"):
                    take(rel.replace("/", "__"), v / rel)
    take("manifest.csv", Path(args.manifest))

    out = Path("run_%s.zip" % gid)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, path in files.items():
            z.write(path, arc)
    print("group %s: %d file(s) -> %s" % (gid, len(files), out))
    for arc in sorted(files):
        print("  ", arc)
    for s_ in skipped:
        print("  not from this run, left out:", s_, file=sys.stderr)


if __name__ == "__main__":
    main()
