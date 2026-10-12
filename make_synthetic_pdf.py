"""Vector PDF plan at 1/4 in = 1 ft with real text, to test the PDF path (no OCR)."""
import pymupdf

# 40 ft x 24 ft house at 1/4"=1' -> 10 in x 6 in -> 720 x 432 pt
doc = pymupdf.open()
page = doc.new_page(width=800, height=520)
ox, oy = 40, 40
s = 18.0  # points per foot (0.25 in = 18 pt)
def L(x0, y0, x1, y1, w=5):
    page.draw_line((ox + x0 * s, oy + y0 * s), (ox + x1 * s, oy + y1 * s), width=w)
L(0, 0, 40, 0); L(40, 0, 40, 24); L(40, 24, 0, 24); L(0, 24, 0, 0)
L(20, 0, 20, 8); L(20, 11, 20, 24)      # vertical wall, 3 ft door gap
L(0, 12, 14, 12); L(17, 12, 40, 12)     # horizontal wall, 3 ft door gap
def T(x, y, txt):
    page.insert_text((ox + x * s, oy + y * s), txt, fontsize=9)
T(5, 5, "LIVING"); T(3.5, 7, "20'-0\" x 12'-0\"")
T(25, 5, "KITCHEN"); T(23.5, 7, "20'-0\" x 12'-0\"")
T(5, 17, "BEDROOM 2"); T(3.5, 19, "12' x 12'")
T(25, 17, "BEDROOM 1"); T(23.5, 19, "20'-0\" x 12'-0\"")
doc.save("plans_pdf/synthetic_plan.pdf")
print("saved plans_pdf/synthetic_plan.pdf")
