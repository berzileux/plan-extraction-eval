"""Synthetic sheet for seeded.py: thin walls, room names with NO printed size beside them, and
dimension strings in chains outside the building (horizontal above, vertical on the left),
like a survey drawing. Scale: 20 pixels per foot. Writes plans/synthetic_open.png."""
import sys

import cv2
import numpy as np

PPF = 20.0
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text_img(txt, scale=0.8, thick=2):
    (w, h), base = cv2.getTextSize(txt, FONT, scale, thick)
    t = np.full((h + base + 8, w + 8, 3), 255, np.uint8)
    cv2.putText(t, txt, (4, h + 2), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return t


def paste(img, t, cx, cy):
    h, w = t.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    img[y0:y0 + h, x0:x0 + w] = np.minimum(img[y0:y0 + h, x0:x0 + w], t)


def build(right_width="20'-0\"", omit=()):
    img = np.full((720, 960, 3), 255, np.uint8)
    T = 8
    k = (0, 0, 0)
    cv2.rectangle(img, (60, 80), (860, 640), k, T)
    cv2.line(img, (460, 80), (460, 200), k, T)
    cv2.line(img, (460, 250), (460, 640), k, T)
    cv2.line(img, (60, 360), (300, 360), k, T)
    cv2.line(img, (350, 360), (860, 360), k, T)
    for txt, cx, cy in (("LIVING", 260, 220), ("KITCHEN", 660, 220),
                        ("BEDROOM 1", 260, 500), ("BEDROOM 2", 660, 500)):
        if txt not in omit:
            paste(img, text_img(txt), cx, cy)
    for txt, cx in (("20'-0\"", 260), (right_width, 660)):
        paste(img, text_img(txt, 0.7), cx, 48)
        paste(img, text_img(txt, 0.7), cx, 676)
    for txt, cy in (("14'-6\"", 220), ("14'-6\"", 500)):
        paste(img, np.rot90(text_img(txt, 0.7), 1).copy(), 28, cy)
    return img


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "plans/synthetic_open.png"
    cv2.imwrite(out, build())
    print("wrote", out, "at 20 pixels per foot; rooms are about 20 x 14 ft; printed lengths say 20 ft across and 14 ft 6 in down")
