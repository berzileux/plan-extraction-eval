"""Optional layout splitter: a YOLO detector that boxes the title block, legend and (if the model
has the class) tables, plus the main drawing area, so those parts of the sheet are kept apart from
the floor plan. This is the stream-splitter layer.

What it changes in seeded.py:
  - text inside a title block, legend or table box is not used for room names or printed lengths
    (a legend that lists "KITCHEN" or a scale note like 1/4" = 1'-0" is not a room or a length);
  - those boxes are planted as background, so no room region can grow into them;
  - a room label that falls outside every drawing-area box is reported in the trace as a warning.
    It is not dropped: a wrong box must not delete a real room.
Everything it does is written to the trace under "layout" so a person can check it.

Install (kept out of requirements.txt on purpose, see licence note):
  pip install ultralytics huggingface_hub
Weights: pass a local .pt file, or a Hugging Face repo id such as
  GreenMap/yolo11x-blueprint-layout-detector
optionally as repo:filename. Without a filename the repo must hold one .pt file, or one called
best.pt, otherwise the script lists the files and stops.

Licence: Ultralytics code and models trained with it are AGPL-3.0 by default, and a model card may
state something else. Read both before any use beyond your own machine. This repo does not bundle
the code or the weights.

Safety: a .pt file is a Python pickle and can run code when loaded. Pin --layout-revision to a
commit you have looked at, and do not load weights from a source you do not trust.

Not run by the repo author: needs a machine that can reach Hugging Face.
"""
import cv2
import numpy as np

DEFAULT_MODEL = "GreenMap/yolo11x-blueprint-layout-detector"


def _norm(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def role_of(name):
    """block: text and pixels inside are kept apart from the plan. area: the drawing itself."""
    n = _norm(name)
    if any(k in n for k in ("title", "legend", "table", "schedule", "notes")):
        return "block"
    if "drawing" in n or n in ("plan", "floor_plan", "plan_view"):
        return "area"
    return None


def load_layout(spec=DEFAULT_MODEL, revision=None):
    """Return an Ultralytics YOLO model. spec: local .pt path, or repo id[:filename]."""
    import os
    from ultralytics import YOLO
    if os.path.isfile(spec):
        return YOLO(spec)
    from huggingface_hub import hf_hub_download, list_repo_files
    repo, _, fname = spec.partition(":")
    if not fname:
        pts = [f for f in list_repo_files(repo, revision=revision) if f.endswith(".pt")]
        if len(pts) == 1:
            fname = pts[0]
        else:
            best = [f for f in pts if f.split("/")[-1] == "best.pt"]
            if len(best) == 1:
                fname = best[0]
            else:
                raise SystemExit("cannot choose a weights file in %s; pass repo:filename. .pt files: %s"
                                 % (repo, pts or "none"))
    return YOLO(hf_hub_download(repo, fname, revision=revision))


def _np(t):
    if hasattr(t, "cpu"):
        t = t.cpu()
    return np.asarray(t)


def detect(model, bgr, conf=0.25, imgsz=1472):
    """Run the detector on CPU. Returns [{cls, role, conf, box:(x0,y0,x1,y1)}] in image pixels.
    model needs predict(img, imgsz=, conf=, device=, verbose=) returning results whose .boxes has
    xyxy / cls / conf and whose .names maps class ids to names."""
    res = model.predict(bgr, imgsz=imgsz, conf=conf, device="cpu", verbose=False)
    out = []
    h, w = bgr.shape[:2]
    for r in res:
        names = r.names
        xyxy, cls, cf = _np(r.boxes.xyxy), _np(r.boxes.cls), _np(r.boxes.conf)
        for b, c, p in zip(xyxy, cls, cf):
            name = names[int(c)]
            x0, y0, x1, y1 = (float(v) for v in b)
            x0, x1 = max(0.0, min(x0, w)), max(0.0, min(x1, w))
            y0, y1 = max(0.0, min(y0, h)), max(0.0, min(y1, h))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            out.append({"cls": _norm(name), "role": role_of(name), "conf": round(float(p), 3),
                        "box": (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))})
    out.sort(key=lambda d: -d["conf"])
    return out


def block_boxes(dets):
    return [d["box"] for d in dets if d["role"] == "block"]


def area_boxes(dets):
    return [d["box"] for d in dets if d["role"] == "area"]


def inside(box, x, y):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def split_items(items, boxes):
    """(kept, dropped): dropped are items whose centre is inside any box."""
    kept, dropped = [], []
    for it in items:
        (dropped if any(inside(b, it["cx"], it["cy"]) for b in boxes) else kept).append(it)
    return kept, dropped


def scale_boxes(boxes, s):
    return [tuple(v * s for v in b) for b in boxes]


def draw_layout_overlay(bgr, dets):
    out = bgr.copy()
    colours = {"block": (0, 0, 220), "area": (0, 160, 0), None: (160, 160, 160)}
    for d in dets:
        x0, y0, x1, y1 = (int(v) for v in d["box"])
        c = colours[d["role"]]
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 3)
        cv2.putText(out, "%s %.2f" % (d["cls"], d["conf"]), (x0 + 4, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
    return out
