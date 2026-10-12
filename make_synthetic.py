"""Draws a simple test plan with known areas, for checking extract.py works."""
import cv2
import numpy as np

PPF = 10.0
img = np.full((420, 620, 3), 255, np.uint8)
T = 6
cv2.rectangle(img, (10, 10), (610, 410), (0, 0, 0), T)
# interior walls drawn as segments, leaving 3 ft (30 px) door gaps
cv2.line(img, (310, 10), (310, 90), (0, 0, 0), T, cv2.LINE_8)
cv2.line(img, (310, 120), (310, 410), (0, 0, 0), T, cv2.LINE_8)
cv2.line(img, (10, 210), (450, 210), (0, 0, 0), T, cv2.LINE_8)
cv2.line(img, (480, 210), (610, 210), (0, 0, 0), T, cv2.LINE_8)
cv2.imwrite("plans/synthetic.png", img)
print("rooms ~ 4 of about 29x19 ft and 29x19 ft; expect each near 540 sq ft")
