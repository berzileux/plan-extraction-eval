"""Add or update one plan's scale in manifest.csv without touching the other lines.

  python manifest-update.py plans/Putnam_House_-_floor_plans.jpg
  python manifest-update.py plans/house.png --ppf 24.0            # you measured the scale yourself
  python manifest-update.py plans/other.png --ref-width 1000 --ref-ppf 9.5

Default scale: the value measured on the 1280 px Putnam preview (11.7 pixels per foot), scaled to
this image's width. That only holds if the image is the same drawing with the same crop. For any
other plan pass --ppf with a scale you measured (pixel distance across a printed length / feet).
The old file is saved as manifest.csv.bak first.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2

HEADER = ["plan", "pixels_per_foot"]


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [(r["plan"].strip(), r["pixels_per_foot"].strip()) for r in csv.DictReader(f) if r.get("plan")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="path to the plan image, inside the plans folder")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--plans", default="plans", help="plans folder (seeded.py only reads files inside it)")
    ap.add_argument("--ppf", type=float, help="pixels per foot you measured; skips the preview formula")
    ap.add_argument("--ref-width", type=float, default=1280.0, help="width of the reference preview, px")
    ap.add_argument("--ref-ppf", type=float, default=11.7, help="pixels per foot measured on the reference")
    args = ap.parse_args()

    img_path = Path(args.image)
    plans = Path(args.plans).resolve()
    if not img_path.is_file():
        sys.exit("cannot find %s" % img_path)
    if img_path.resolve().parent != plans:
        sys.exit("%s is not directly inside %s/. Copy it there first; the pipeline refuses other paths."
                 % (img_path, args.plans))
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit("cannot read %s as an image" % img_path)
    w = img.shape[1]

    if args.ppf is not None:
        ppf, how = args.ppf, "given with --ppf"
    else:
        ppf = args.ref_ppf * w / args.ref_width
        how = "%.1f x %d / %d (preview formula; valid only for the same drawing and crop)" % (
            args.ref_ppf, w, args.ref_width)
    if not (0.5 <= ppf <= 500):
        sys.exit("pixels_per_foot %.2f is outside 0.5 to 500; check the numbers" % ppf)

    manifest = Path(args.manifest)
    rows = read_rows(manifest)
    if manifest.exists():
        shutil.copyfile(manifest, str(manifest) + ".bak")
    name = img_path.name
    new = "%.2f" % ppf
    rows = [(n, new if n == name else v) for n, v in rows]
    if name not in [n for n, _ in rows]:
        rows.append((name, new))
    with open(manifest, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(HEADER)
        wr.writerows(rows)
    print("image width: %d px" % w)
    print("scale: %s pixels per foot, %s" % (new, how))
    print("wrote %s:" % manifest)
    print(manifest.read_text())


if __name__ == "__main__":
    main()
