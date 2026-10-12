"""Synthetic plan WITH printed room names and sizes, to smoke-test the OCR step."""
import cv2
import numpy as np

img = np.full((840, 1240, 3), 255, np.uint8)
T = 12
cv2.rectangle(img, (20, 20), (1220, 820), (0, 0, 0), T)
cv2.line(img, (620, 20), (620, 180), (0, 0, 0), T)
cv2.line(img, (620, 240), (620, 820), (0, 0, 0), T)
cv2.line(img, (20, 420), (900, 420), (0, 0, 0), T)
cv2.line(img, (960, 420), (1220, 420), (0, 0, 0), T)

def label(cx, cy, name, dims):
    for dy, txt in ((-15, name), (35, dims)):
        (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.putText(img, txt, (cx - w // 2, cy + dy), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)

label(320, 220, "LIVING", "29'-5\" x 19'-5\"")
label(920, 220, "KITCHEN", "29'-5\" x 19'-5\"")
label(920, 620, "BEDROOM 1", "29'-5\" x 19'-5\"")
label(320, 620, "BEDROOM 2", "12' x 12'")
cv2.imwrite("plans/synthetic_text.png", img)
