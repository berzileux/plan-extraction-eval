"""Optional second opinion on every room region, from Segment Anything 2 (SAM 2).

Not a replacement for the label-seeded regions. SAM 2 is a general "cut out the object under
this point" model that was not trained on floor plans. Each room label gives one positive point
and every other label a negative point; the mask SAM 2 returns is compared with the watershed
region. A low overlap does not say which one is right. It says the two methods disagree, so a
person should look. Agreement never marks a room ok on its own: only a printed size does that.

Install (CPU works, tiny model): follow the SAM 2 repository instructions, for example
  pip install git+https://github.com/facebookresearch/sam2.git
The first run downloads the checkpoint from Hugging Face. Check the model licence yourself.
"""
import cv2
import numpy as np

DEFAULT_MODEL = "facebook/sam2-hiera-tiny"


def load_sam2(model_id=DEFAULT_MODEL):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor.from_pretrained(model_id)


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def sam_masks(predictor, bgr, seeds_xy, ppf, min_sqft=20.0, max_sqft=900.0):
    """One boolean mask per seed. predictor needs set_image(rgb) and
    predict(point_coords, point_labels, multimask_output) -> (masks, scores, logits)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    out = []
    for i, (x, y) in enumerate(seeds_xy):
        pts = [[x, y]] + [[ox, oy] for j, (ox, oy) in enumerate(seeds_xy) if j != i]
        labels = [1] + [0] * (len(pts) - 1)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(pts, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            multimask_output=True)
        masks = np.asarray(masks).astype(bool)
        order = np.argsort(-np.asarray(scores))
        pick = None
        for k in order:  # best-scoring mask whose size is plausible for a room
            sqft = masks[k].sum() / (ppf * ppf)
            if min_sqft <= sqft <= max_sqft:
                pick = masks[k]
                break
        out.append(pick if pick is not None else masks[order[0]])
    return out


def compare(rooms, region_masks, sam_masks_, ppf, agree=0.5):
    """Attach sam_iou / sam_area_sqft to each room and say so when the methods disagree."""
    for rm, ws, sm in zip(rooms, region_masks, sam_masks_):
        if ws is None or not ws.any():
            continue
        v = iou(ws, sm)
        rm["sam_iou"] = round(v, 2)
        rm["sam_area_sqft"] = round(float(sm.sum()) / (ppf * ppf), 1)
        if v < agree:
            note = "second method (SAM 2) disagrees with this region (overlap %.0f%%)" % (v * 100)
            if rm.get("needs_review"):
                rm["review_reason"] = (rm.get("review_reason", "") + "; " + note).lstrip("; ")
            else:
                rm["sam_note"] = note
    return rooms


def draw_sam_overlay(bgr, sam_masks_, rooms):
    out = bgr.copy()
    for rm, m in zip(rooms, sam_masks_):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
        x, y = rm["centroid"]
        cv2.putText(out, "#%d %s" % (rm["id"], rm.get("sam_iou", "?")), (max(0, x - 30), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return out
