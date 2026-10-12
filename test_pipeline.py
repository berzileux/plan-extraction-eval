import json
import os
import subprocess
import sys
import tempfile
import unittest

from dims import parse_dims, strip_dims, normalize


class TestDims(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_dims("12'-6\" x 14'-0\""), (12.5, 14.0))
        self.assertEqual(parse_dims("12' x 14'"), (12.0, 14.0))
        self.assertEqual(parse_dims("12x14"), (12.0, 14.0))
        self.assertEqual(parse_dims("BEDROOM 11'6\" X 13'2\""), (11.5, round(13 + 2 / 12, 2)))

    def test_ocr_misreads(self):
        self.assertEqual(parse_dims("1Z'-6\" x 14'-0\""), (12.5, 14.0))
        self.assertEqual(parse_dims("l2' x 14'"), (12.0, 14.0))
        self.assertEqual(parse_dims("12\u2019-6\u201d x 14\u2019"), (12.5, 14.0))
        self.assertEqual(normalize("BEDROOM 1"), "BEDROOM 1")
        self.assertEqual(normalize("KITCHEN"), "KITCHEN")

    def test_label_not_polluted(self):
        self.assertEqual(parse_dims("BEDROOM 1 29'-5\" x 19'-5\""), (29.42, 19.42))
        self.assertEqual(strip_dims("BEDROOM 1 29'-5\" x 19'-5\""), "BEDROOM 1")
        self.assertEqual(strip_dims("KITCHEN"), "KITCHEN")

    def test_rejects(self):
        self.assertIsNone(parse_dims("KITCHEN"))
        self.assertIsNone(parse_dims("1x1"))
        self.assertIsNone(parse_dims("12'-13\" x 14'"))
        self.assertIsNone(parse_dims(""))


class TestEndToEnd(unittest.TestCase):
    def test_fake_ocr_flow(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ)
            run = lambda *a: subprocess.run([sys.executable] + list(a), cwd=here, check=True,
                                            capture_output=True, text=True, env=env)
            run("make_synthetic.py")
            man = os.path.join(d, "manifest.csv")
            open(man, "w").write("plan,pixels_per_foot\nsynthetic.png,10.0\n")
            run("extract.py", "--manifest", man, "--out", d)
            fake = os.path.join(d, "fake.json")
            json.dump({"synthetic.png": [
                {"text": "LIVING", "cx": 150, "cy": 100},
                {"text": "29'-5\" x 19'-5\"", "cx": 150, "cy": 130},
                {"text": "BED", "cx": 450, "cy": 100},
                {"text": "10' x 10'", "cx": 450, "cy": 130},
            ]}, open(fake, "w"))
            run("read_labels.py", "--results", os.path.join(d, "results.json"), "--fake-ocr", fake)
            data = json.load(open(os.path.join(d, "results.json")))
            rooms = {r["id"]: r for r in data["plans"][0]["rooms"]}
            ok = [r for r in rooms.values() if r.get("label") == "LIVING"][0]
            bad = [r for r in rooms.values() if r.get("label") == "BED"][0]
            self.assertFalse(ok["needs_review"])
            self.assertTrue(bad["needs_review"])
            self.assertGreater(bad["agreement_gap_pct"], 100)
            unlabeled = [r for r in rooms.values() if r.get("label") == ""]
            self.assertTrue(all(r["needs_review"] for r in unlabeled))



class TestInterpret(unittest.TestCase):
    def room(self, raw, rules=None, area=144.0):
        rm = {"id": 1, "area_sqft": area, "raw_text": raw, "label": "", "needs_review": True}
        if rules:
            rm["stated_dims_ft"] = list(rules)
        return rm

    def test_model_fills_size_rules_missed(self):
        import interpret
        rm = self.room(["BEDROOM", "12 by 12"])
        interpret.interpret_room(rm, lambda t: {"name": "BEDROOM", "width_ft": 12, "depth_ft": 12})
        self.assertEqual(rm["stated_dims_ft"], [12.0, 12.0])
        self.assertEqual(rm["dims_source"], "model")
        self.assertFalse(rm["needs_review"])

    def test_invented_size_is_ignored(self):
        import interpret
        rm = self.room(["BED"])
        interpret.interpret_room(rm, lambda t: {"name": "BED", "width_ft": 20, "depth_ft": 20})
        self.assertNotIn("stated_dims_ft", rm)
        self.assertTrue(rm["needs_review"])
        self.assertIn("not in the text", rm["model_note"])

    def test_disagreement_is_flagged(self):
        import interpret
        rm = self.room(["12' x 12'", "14 x 14"], rules=(12.0, 12.0))
        interpret.interpret_room(rm, lambda t: {"name": None, "width_ft": 14, "depth_ft": 14})
        self.assertTrue(rm["needs_review"])
        self.assertEqual(rm["review_reason"], "rules and model read different printed sizes")

    def test_model_failure_does_not_crash(self):
        import interpret
        rm = self.room(["KITCHEN"])
        def boom(t):
            raise RuntimeError("server down")
        interpret.interpret_room(rm, boom)
        self.assertIn("server down", rm["model_error"])


    def test_stray_digits_do_not_ground_a_model_size(self):
        import interpret
        rm = self.room(["BEDROOM", "ROOM C", "L 4", "_ Lal 5"])
        interpret.interpret_room(rm, lambda t: {"name": "BEDROOM", "width_ft": 4, "depth_ft": 5})
        self.assertNotIn("stated_dims_ft", rm)
        self.assertIn("not in the text", rm["model_note"])

    def test_pair_in_one_item_grounds_a_model_size(self):
        import interpret
        rm = self.room(["BEDROOM", "14 by 12"])
        interpret.interpret_room(rm, lambda t: {"name": "BEDROOM", "width_ft": 12, "depth_ft": 14})
        self.assertEqual(rm["stated_dims_ft"], [12.0, 14.0])

    def test_remote_server_refused(self):
        import interpret
        self.assertTrue(interpret.local_only("http://127.0.0.1:11434"))
        self.assertFalse(interpret.local_only("http://example.com:11434"))

    def test_garbage_values_rejected(self):
        import interpret
        rm = self.room(["12 x 12"])
        interpret.interpret_room(rm, lambda t: {"name": "<script>x</script>", "width_ft": "12", "depth_ft": True})
        self.assertNotIn("stated_dims_ft", rm)
        self.assertNotIn("<", rm.get("label", ""))


class TestPdfPath(unittest.TestCase):
    def test_pdf_text_layer_end_to_end(self):
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            self.skipTest("pymupdf not installed")
        here = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(here, "plans_pdf"), exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            run = lambda *a: subprocess.run([sys.executable] + list(a), cwd=here, check=True,
                                            capture_output=True, text=True)
            run("make_synthetic_pdf.py")
            run("ingest.py", "plans_pdf/synthetic_plan.pdf", "--plans", d, "--text-db", d)
            man = os.path.join(d, "manifest.csv")
            with open(man, "w") as f:
                f.write("plan,pixels_per_foot\nsynthetic_plan_p1.png,50.0\n")
            run("extract.py", "--plans", d, "--manifest", man, "--out", d)
            run("read_labels.py", "--plans", d, "--results", os.path.join(d, "results.json"),
                "--text-db", d)
            with open(os.path.join(d, "results.json")) as f:
                rooms = json.load(f)["plans"][0]["rooms"]
            by = {r["label"]: r for r in rooms}
            self.assertEqual(set(by), {"LIVING", "KITCHEN", "BEDROOM 1", "BEDROOM 2"})
            self.assertEqual(by["LIVING"]["stated_dims_ft"], [20.0, 12.0])
            self.assertFalse(by["LIVING"]["needs_review"])
            self.assertTrue(by["BEDROOM 2"]["needs_review"])


class TestLabelCleanup(unittest.TestCase):
    def test_stray_marks_removed(self):
        from dims import clean_label
        self.assertEqual(clean_label("LIVING '"), "LIVING")
        self.assertEqual(clean_label("BEDROOM 1 *"), "BEDROOM 1")
        self.assertEqual(clean_label("  W/C  "), "W/C")
        self.assertEqual(strip_dims("LIVING 29'-5\" x 19'-5\" '"), "LIVING")


class TestQualityAndMerge(unittest.TestCase):
    def test_merge_words_keeps_x_in_order(self):
        from ingest import merge_words
        words = [
            {"text": "29'-5", "x0": 100, "y0": 100, "x1": 150, "y1": 120},
            {"text": "x", "x0": 156, "y0": 106, "x1": 166, "y1": 120},
            {"text": "19'-5", "x0": 172, "y0": 100, "x1": 222, "y1": 120},
        ]
        out = merge_words(words)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "29'-5 x 19'-5")

    def test_plan_quality_flags_edge_and_fragments(self):
        from extract import plan_quality
        shape = (1000, 1000, 3)
        good = [{"id": i, "area_sqft": 200.0, "bbox": [100 + i, 100, 200 + i, 200]} for i in range(4)]
        self.assertTrue(plan_quality(good, shape)["geometry_reliable"])
        edge = good + [{"id": 9, "area_sqft": 600.0, "bbox": [2, 3, 500, 300]}]
        q = plan_quality(edge, shape)
        self.assertFalse(q["geometry_reliable"])
        frag = [{"id": i, "area_sqft": 20.0, "bbox": [100, 100, 150, 150]} for i in range(10)]
        self.assertFalse(plan_quality(frag, shape)["geometry_reliable"])


class TestSeeded(unittest.TestCase):
    def test_read_name(self):
        from seeded import read_name
        self.assertEqual(read_name("WEST*FARLOR"), "WEST PARLOR")
        self.assertEqual(read_name("BED'ROOM:A"), "BED ROOM A")
        self.assertEqual(read_name("MEMORIAL: ROOM"), "MEMORIAL ROOM")
        self.assertEqual(read_name("BEDROOM1"), "BEDROOM 1")
        self.assertEqual(read_name("PARLOI2"), "PARLOR")
        self.assertIsNone(read_name("DOOR SCHEDULE"))
        self.assertIsNone(read_name("CLOS."))
        self.assertIsNone(read_name("OFFICE OF NATIONAL PARKS"))
        self.assertEqual(read_name("OFFICE", ["OFFICE"]), "OFFICE")

    def test_all_singles(self):
        from dims import all_singles
        self.assertEqual(all_singles("13'-6\""), [13.5])
        self.assertEqual(all_singles("8'7\" 4'"), [8.58, 4.0])
        self.assertEqual(all_singles("12' x 14'"), [])
        self.assertEqual(all_singles("KITCHEN"), [])

    def test_seed_uses_name_words_not_line_centre(self):
        from seeded import find_seeds
        items = [{"text": "BE by DINING", "cx": 349, "cy": 403, "parts": [
                    {"text": "BE", "cx": 290, "cy": 403}, {"text": "by", "cx": 330, "cy": 403},
                    {"text": "DINING", "cx": 426, "cy": 403}]},
                 {"text": "DINING", "cx": 428, "cy": 404, "parts": [{"text": "DINING", "cx": 428, "cy": 404}]}]
        seeds = find_seeds(items, 11.7)
        self.assertEqual(len(seeds), 1)
        self.assertAlmostEqual(seeds[0][1], 426, delta=3)

    def test_clean_raw_texts(self):
        from seeded import clean_raw_texts
        out = clean_raw_texts(["KITCHEN", "kitchen", "-", "_ _", "x", "KITCHEN.", "12' x 14'", "POST"])
        self.assertEqual(out, ["KITCHEN", "12' x 14'", "POST"])
        self.assertLessEqual(len(clean_raw_texts(["WORD%d" % i for i in range(500)])), 40)

    def test_verify_region(self):
        from seeded import verify_region
        dims = [{"ft": 20.0, "cx": 260, "cy": 48, "orient": "h"},
                {"ft": 14.0, "cx": 28, "cy": 220, "orient": "v"},
                {"ft": 12.0, "cx": 660, "cy": 48, "orient": "h"}]
        got = verify_region([70, 90, 450, 350], dims, 20.0)
        self.assertEqual(got, {"width": 20.0, "depth": 14.0})
        miss = verify_region([470, 90, 850, 350], dims, 20.0)
        self.assertIsNone(miss["width"])

    def test_verify_region_ignores_far_lengths(self):
        from seeded import verify_region
        # 20 ft width printed 60 ft above the room (another plan on the sheet), and a 14 ft
        # depth printed 80 ft to the left: same columns / rows, but not near the room.
        dims = [{"ft": 20.0, "cx": 260, "cy": 90 - 60 * 20, "orient": "h"},
                {"ft": 14.0, "cx": 70 - 80 * 20, "cy": 220, "orient": "v"}]
        got = verify_region([70, 90, 450, 350], dims, 20.0)
        self.assertEqual(got, {"width": None, "depth": None})

    def test_verify_region_rejects_a_length_a_room_away(self):
        from seeded import verify_region
        # The sample sheet's defect: a 14'10" width printed in the room above was taken
        # as this room's width because the old cross-axis radius was 12 ft, wider than
        # the rooms themselves. 7 ft clear of the region is another room's dimension.
        dims = [{"ft": 14.83, "cx": 260, "cy": 90 - 7 * 20, "orient": "h"}]
        self.assertIsNone(verify_region([70, 90, 450, 350], dims, 20.0)["width"])

    def test_lexicon_carries_attic_and_bay(self):
        import ruleset
        ruleset.reload()
        words = ruleset.lexicon()
        # Both were missing, and each cost the sample sheet a whole region: no word,
        # no seed, no room. The attic's printed width was then matched elsewhere.
        for w in ("ATTIC", "BAY"):
            self.assertIn(w, words)

    def test_segment_separates_rooms(self):
        import cv2
        import seeded
        import make_synthetic_open as m
        img = m.build()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        seeds = [(260, 220), (660, 220), (260, 500), (660, 500)]
        lab, ext = seeded.segment(gray, 20.0, seeds)
        for i, s in enumerate(seeds, start=1):
            mask = seeded.region_mask(lab, i, s, 20.0)
            sqft = mask.sum() / 400.0
            self.assertTrue(240 < sqft < 290, sqft)
        self.assertTrue((lab[5, 5] == ext))

    def test_ocr_to_verified_rooms(self):
        import shutil
        if not shutil.which("tesseract"):
            self.skipTest("tesseract not installed")
        import cv2
        import seeded
        import make_synthetic_open as m
        quiet = lambda *a, **k: None
        rooms, *_ = seeded.read_plan(m.build(), 20.0, fast=True, log=quiet)
        by = {r["label"]: r for r in rooms}
        self.assertEqual({"LIVING", "KITCHEN", "BEDROOM 1"} - set(by), set())
        self.assertFalse(by["LIVING"]["needs_review"])
        self.assertEqual(by["LIVING"]["stated_dims_ft"], [20.0, 14.5])
        rooms, *_ = seeded.read_plan(m.build("12'-0\""), 20.0, fast=True, log=quiet)
        by = {r["label"]: r for r in rooms}
        self.assertFalse(by["LIVING"]["needs_review"])
        self.assertTrue(by["KITCHEN"]["needs_review"])
        self.assertTrue(rooms and all("review_reason" in r for r in rooms if r["needs_review"]))


    def test_trace_records_decisions(self):
        import json
        import os
        import tempfile
        import shutil
        if not shutil.which("tesseract"):
            self.skipTest("tesseract not installed")
        import seeded
        import make_synthetic_open as m
        from runtrace import Trace
        tr = Trace("synthetic_open.png", {"engine": "tesseract"}, group="20261011-024724")
        rooms, *_ = seeded.read_plan(m.build(), 20.0, fast=True, log=lambda *a: None, trace=tr)
        with tempfile.TemporaryDirectory() as d:
            path = tr.write(os.path.join(d, "t.json"))
            data = json.load(open(path))
        self.assertRegex(data["trace_id"], r"^20261011-024724-[0-9a-f]{6}$")
        self.assertEqual(data["group_id"], "20261011-024724")
        self.assertRegex(data["started"], r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$")
        stages = {s["stage"] for s in data["stages"]}
        self.assertTrue({"ocr horizontal", "ocr rotated", "segment"} <= stages)
        self.assertTrue(any(s["decision"] == "kept" for s in data["seeds"]))
        self.assertTrue(data["dims"])
        living = [r for r in data["rooms"] if r["label"] == "LIVING"][0]
        self.assertTrue(living["verified"])
        self.assertIn("aligned_printed_lengths", living)
        self.assertTrue(living["aligned_printed_lengths"]["width"])
        self.assertTrue(all(r["trace_ref"].startswith(data["trace_id"]) for r in rooms))

    def test_sam_check_flags_disagreement(self):
        import numpy as np
        import cv2
        import sam_check
        import seeded
        import make_synthetic_open as m
        img = m.build()
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        seeds = [(260, 220), (660, 220), (260, 500), (660, 500)]
        lab, ext = seeded.segment(gray, 20.0, seeds)
        ws = [seeded.region_mask(lab, i, s, 20.0) for i, s in enumerate(seeds, start=1)]

        class Fake:
            # agrees with the watershed region for room 1, and returns a far-off box for the rest
            def set_image(self, rgb):
                self.calls = 0

            def predict(self, point_coords, point_labels, multimask_output):
                i = self.calls
                self.calls += 1
                good = ws[0].copy()
                bad = np.zeros((h, w), bool)
                bad[10:60, 10:60] = True
                pick = good if i == 0 else bad
                return np.stack([pick, pick, pick]), np.array([0.9, 0.5, 0.1]), None

        sms = sam_check.sam_masks(Fake(), img, seeds, 20.0, min_sqft=1.0, max_sqft=9999.0)
        rooms = [{"id": i + 1, "needs_review": True, "review_reason": "x", "centroid": [0, 0]}
                 for i in range(4)]
        sam_check.compare(rooms, ws, sms, 20.0)
        self.assertGreater(rooms[0]["sam_iou"], 0.95)
        self.assertLess(rooms[1]["sam_iou"], 0.1)
        self.assertIn("SAM 2", rooms[1]["review_reason"])
        self.assertNotIn("SAM 2", rooms[0]["review_reason"])


    def test_dimension_reader_reads_vertical_strings(self):
        import shutil
        if not shutil.which("tesseract"):
            self.skipTest("tesseract not installed")
        import seeded
        from dims import all_lengths
        import make_synthetic_open as m
        layer = seeded.text_layer(m.build(), 20.0, keep_crossings=True)
        got = seeded.read_dimension_strings(layer, 20.0, "v")
        self.assertEqual(len(got), 2)
        for g in got:
            self.assertAlmostEqual(g["cx"], 27, delta=6)
            self.assertEqual(all_lengths(g["text"]), [14.5])
        self.assertEqual(sorted(round(g["cy"]) for g in got)[0] // 100, 2)

    def test_doubled_foot_mark_is_one_mark(self):
        from dims import all_lengths
        self.assertEqual(all_lengths("14\u2019'-9\u201d"), [14.75])
        self.assertEqual(all_lengths("14'-6\""), [14.5])

    def test_schedule_finder(self):
        import seeded
        rows = ["2'-10\" x 6'-7\" x 1 1/2\" SIX PANEL", "2'-8\" x 6'-6\" x 1 1/2\" FOUR PANEL",
                "2'-6\" x 6'-2\" x 7/8\" FOUR", "2'-4\" x 6'-1\" x 1 1/8\" SHEATHING"]
        items = [{"text": "%d %s" % (i + 1, t), "cx": 600 + 4 * i, "cy": 60 + 14 * i} for i, t in enumerate(rows)]
        items += [{"text": "LIVING", "cx": 200, "cy": 300}, {"text": "20' x 14'", "cx": 200, "cy": 330}]
        found = seeded.find_schedules(items, 11.7)
        self.assertEqual(len(found), 1)
        self.assertEqual(len(found[0]["rows"]), 4)
        self.assertEqual(found[0]["rows"][0]["mark"], "1")
        self.assertEqual(found[0]["rows"][0]["lengths_ft"][:2], [2.83, 6.58])
        self.assertEqual(seeded.find_schedules(items[-2:], 11.7), [])

    def test_florence_adapter_offsets_and_pipeline(self):
        import cv2
        import seeded
        import make_synthetic_open as m
        img = m.build()
        calls = []

        def fake_run(pil):
            calls.append(pil.size)
            w, h = pil.size
            out = []
            # names at their true page positions, expressed in this tile's pixels
            for txt, cx, cy in (("LIVING", 260, 220), ("KITCHEN", 660, 220),
                                ("BEDROOM 1", 260, 500), ("BEDROOM 2", 660, 500),
                                ("20'-0\"", 260, 48), ("20'-0\"", 660, 48),
                                ("20'-0\"", 260, 676), ("20'-0\"", 660, 676)):
                ox, oy = fake_run.origin
                x, y = cx - ox, cy - oy
                if 0 <= x < w and 0 <= y < h:
                    out.append((txt, [x - 20, y - 8, x + 20, y - 8, x + 20, y + 8, x - 20, y + 8]))
            return out

        # drive florence_items tile by tile so each fake call knows its own origin
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = list(seeded._tile_boxes(gray.shape[1], gray.shape[0], 20.0))
        it = iter(boxes)

        def run(pil):
            fake_run.origin = next(it)[:2]
            return fake_run(pil)

        items = seeded.florence_items(gray, 20.0, run)
        seeds = seeded.find_seeds(items, 20.0)
        names = sorted(s[0] for s in seeds)
        self.assertEqual(names, ["BEDROOM 1", "BEDROOM 2", "KITCHEN", "LIVING"])
        for name, x, y in seeds:
            if name == "LIVING":
                self.assertAlmostEqual(x, 260, delta=2)
                self.assertAlmostEqual(y, 220, delta=2)

    # ---- layout splitter (fake detector; the real YOLO model is never run in these tests)

    @staticmethod
    def _fake_layout(boxes):
        import numpy as np

        class B:
            def __init__(self, rows):
                self.xyxy = np.array([r[1] for r in rows], dtype=np.float32)
                self.cls = np.array([r[0] for r in rows], dtype=np.float32)
                self.conf = np.array([r[2] for r in rows], dtype=np.float32)

        class R:
            names = {0: "drawing_area", 1: "legend_block", 2: "title_block"}

            def __init__(self, rows):
                self.boxes = B(rows)

        class M:
            def predict(self, img, **kw):
                M.kw = kw
                return [R(boxes)]
        return M()

    def test_manifest_rejects_placeholder_text(self):
        import tempfile
        import os
        import seeded
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "m.csv")
            open(f, "w").write("plan,pixels_per_foot\nputnam.png,<11.7 * w / 1280\n")
            with self.assertRaises(SystemExit) as cm:
                seeded.load_manifest(f)
            self.assertIn("plain number", str(cm.exception))
            open(f, "w").write("plan,pixels_per_foot\nputnam.png,11.7\n")
            self.assertEqual(seeded.load_manifest(f), {"putnam.png": 11.7})

    def test_manifest_update_script(self):
        import tempfile
        import os
        import cv2
        import numpy as np
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest-update.py")
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "plans"))
            cv2.imwrite(os.path.join(d, "plans", "a.png"), np.full((50, 2560, 3), 255, np.uint8))
            open(os.path.join(d, "manifest.csv"), "w").write("plan,pixels_per_foot\nb.png,9.0\n")
            run = lambda *a: subprocess.run([sys.executable, script, *a], cwd=d, capture_output=True, text=True)
            r = run("plans/a.png")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(open(os.path.join(d, "manifest.csv")).read().split(), ["plan,pixels_per_foot", "b.png,9.0", "a.png,23.40"])
            self.assertTrue(os.path.exists(os.path.join(d, "manifest.csv.bak")))
            run("plans/a.png", "--ppf", "30")
            self.assertIn("a.png,30.00", open(os.path.join(d, "manifest.csv")).read())
            self.assertEqual(open(os.path.join(d, "manifest.csv")).read().count("a.png"), 1)
            self.assertNotEqual(run("../x.png").returncode, 0)
            self.assertNotEqual(run("plans/a.png", "--ppf", "9999").returncode, 0)

    def test_dimension_chain_string_is_not_given_one_position(self):
        import seeded
        items = [{"text": "4'1\"2\"4\"5\"4'5\"6\"7'8\"8\"5'4\"6'7\"", "cx": 977, "cy": 862},
                 {"text": "12'6\"", "cx": 1086, "cy": 487}]
        skipped = []
        dims = seeded.collect_dims(items, [], skipped=skipped)
        self.assertEqual([d["ft"] for d in dims], [12.5])
        self.assertEqual(len(skipped), 1)
        self.assertGreater(skipped[0]["n_lengths"], 3)

    def test_two_names_joined_by_marks_become_two_seeds(self):
        import seeded
        def part(text, x0, x1):
            return {"text": text, "x0": x0, "x1": x1, "cx": (x0 + x1) / 2, "cy": 100}
        joined = {"text": "POST HALL ---- BATH", "cx": 1033, "cy": 100,
                  "parts": [part("POST", 940, 970), part("HALL", 975, 1010), part("----", 1015, 1060),
                            part("BATH", 1075, 1105)]}
        names = sorted(n for n, _, _ in seeded.find_seeds([joined], 11.7))
        self.assertEqual(names, ["BATH", "HALL"])
        one = {"text": "BED ROOM D", "cx": 100, "cy": 50,
               "parts": [part("BED", 80, 100), part("ROOM", 104, 135), part("D", 140, 150)]}
        self.assertEqual([n for n, _, _ in seeded.find_seeds([one], 11.7)], ["BED ROOM D"])

    def test_seed_on_a_wall_still_gets_a_region(self):
        import cv2
        import seeded
        import make_synthetic_open as m
        gray = cv2.cvtColor(m.build(), cv2.COLOR_BGR2GRAY)
        lines = seeded.line_map(gray, 20.0)
        self.assertTrue(lines[500, 460] > 0)  # the vertical wall between the two lower rooms
        lab, ext = seeded.segment(gray, 20.0, [(460, 500), (260, 500)])
        self.assertGreater((lab == 1).sum() / 400.0, 20.0)

    def test_group_ids(self):
        import os
        from runtrace import Trace, check_label, new_group, stamp
        a, b = Trace("one.png", group="20261011-024724"), Trace("two.png", group="20261011-024724")
        self.assertRegex(a.id, r"^20261011-024724-[0-9a-f]{6}$")
        self.assertNotEqual(a.id, b.id)  # same group, different plan
        self.assertEqual(a.data["group_id"], b.data["group_id"])
        self.assertRegex(Trace("x.png").id, r"^\d{8}-\d{6}-[0-9a-f]{6}$")
        old = os.environ.pop("GROUP_ID", None)
        try:
            os.environ["GROUP_ID"] = "20261011-111111"
            self.assertEqual(new_group(), "20261011-111111")
            self.assertEqual(new_group("mine-1"), "mine-1")  # explicit beats the environment
            del os.environ["GROUP_ID"]
            self.assertRegex(new_group(), r"^\d{8}-\d{6}$")
        finally:
            os.environ.pop("GROUP_ID", None)
            if old is not None:
                os.environ["GROUP_ID"] = old
        for bad in ("../x", "a b", "", "x" * 40):
            with self.assertRaises(ValueError):
                check_label(bad)
        data = {}
        stamp(data, "extract", "20261011-024724")
        stamp(data, "interpret", "20261011-030000")
        self.assertEqual(data["group_id"], "20261011-024724")  # the first layer sets it
        self.assertEqual(data["layers"]["interpret"]["group_id"], "20261011-030000")

    def test_layout_roles_and_detect(self):
        import numpy as np
        import layout_check as lc
        self.assertEqual(lc.role_of("title_block"), "block")
        self.assertEqual(lc.role_of("Legend Block"), "block")
        self.assertEqual(lc.role_of("table"), "block")
        self.assertEqual(lc.role_of("drawing_area"), "area")
        self.assertIsNone(lc.role_of("door"))
        img = np.zeros((100, 200, 3), np.uint8)
        m = self._fake_layout([(2, (150, 60, 260, 130), 0.9), (0, (0, 0, 140, 100), 0.8), (1, (5, 5, 6, 6), 0.7)])
        dets = lc.detect(m, img, conf=0.3, imgsz=640)
        self.assertEqual([d["cls"] for d in dets], ["title_block", "drawing_area"])  # tiny box dropped
        self.assertEqual(dets[0]["box"], (150.0, 60.0, 200.0, 100.0))  # clamped to the image
        self.assertEqual(m.kw["device"], "cpu")
        kept, dropped = lc.split_items([{"cx": 160, "cy": 70}, {"cx": 20, "cy": 20}], lc.block_boxes(dets))
        self.assertEqual((len(kept), len(dropped)), (1, 1))

    def test_segment_plants_background_boxes(self):
        import cv2
        import seeded
        import make_synthetic_open as m
        gray = cv2.cvtColor(m.build(), cv2.COLOR_BGR2GRAY)
        seeds = [(260, 220), (660, 220), (260, 500), (660, 500)]
        lab0, ext0 = seeded.segment(gray, 20.0, seeds)
        self.assertEqual(lab0[220, 740], 2)  # right strip of the top-right room
        lab, ext = seeded.segment(gray, 20.0, seeds, bg_boxes=[(720, 150, 780, 320)])
        self.assertEqual(lab[220, 740], ext)
        self.assertEqual(lab[220, 660], 2)  # the seed itself is never overwritten

    def test_layout_sets_aside_room_name_and_warns(self):
        import shutil
        if not shutil.which("tesseract"):
            self.skipTest("tesseract not installed")
        import seeded
        import make_synthetic_open as m
        from runtrace import Trace
        base, _, _, _, wppf, wseeds = seeded.read_plan(m.build(), 20.0, fast=True, log=lambda *a: None)
        k = 20.0 / wppf  # seeds come back in working-image pixels; the detector works in image pixels
        x, y = [(x * k, y * k) for n, x, y in wseeds if n == "KITCHEN"][0]
        model = self._fake_layout([(1, (x - 90, y - 30, x + 90, y + 30), 0.9)])
        tr = Trace("synthetic_open.png", {})
        rooms, *_ = seeded.read_plan(m.build(), 20.0, fast=True, log=lambda *a: None, trace=tr, layout=model)
        self.assertNotIn("KITCHEN", [r["label"] for r in rooms])
        self.assertTrue(any("KITCHEN" in w for w in tr.data["warnings"]))
        self.assertEqual(tr.data["layout"][0]["cls"], "legend_block")
        self.assertIn("layout detector", {s["stage"] for s in tr.data["stages"]})
        # with no detections nothing changes and the trace says so
        tr2 = Trace("synthetic_open.png", {})
        rooms2, *_ = seeded.read_plan(m.build(), 20.0, fast=True, log=lambda *a: None, trace=tr2,
                                      layout=self._fake_layout([]))
        self.assertEqual(sorted(r["label"] for r in rooms2), sorted(r["label"] for r in base))
        self.assertTrue(any("no boxes" in w for w in tr2.data["warnings"]))


if __name__ == "__main__":
    unittest.main()


class TestDimensions(unittest.TestCase):
    def test_parse_length(self):
        from dimensions import parse_length as P
        self.assertAlmostEqual(P("4'8\""), 4 + 8 / 12)
        self.assertAlmostEqual(P(". 4'8\" \""), 4 + 8 / 12)       # reader adds junk
        self.assertAlmostEqual(P("7'10 1/2\""), 7.875)
        self.assertAlmostEqual(P("7'10½\""), 7.875)
        self.assertAlmostEqual(P("8'0"), 8.0)
        for bad in ("", "0 1", "References", "15'3\" 7'10\" 14'10\"", "4'15\""):
            self.assertIsNone(P(bad), bad)

    def test_believe_rejects_the_dangerous_misread(self):
        from dimensions import believe
        # a reader returned 12'0" for a run that measures 12.52 ft: must not be believed
        self.assertFalse(believe(12.0, 12.52))
        # but a hand-drawn short segment is allowed to be well off its printed value
        self.assertTrue(believe(2 + 8 / 12, 2.24))
        self.assertTrue(believe(4 + 8 / 12, 4.39))
        self.assertFalse(believe(33 + 5 / 12, 3.35))              # stray leading digit

    def test_solve_chain_recovers_repeated_gaps(self):
        """The Putnam second-floor chain: six readings, five unread 1'11" piers."""
        from dimensions import solve_chain
        meas = [4.39, 2.18, 2.44, 2.19, 8.14, 2.24, 6.78, 2.18, 2.24, 2.10, 3.35]
        stated = [4 + 8 / 12, None, 2 + 8 / 12, None, 7 + 10.5 / 12, None,
                  7 + 0.5 / 12, None, 2 + 8 / 12, None, 3 + 5 / 12]
        truth = [4 + 8 / 12, 1 + 11 / 12, 2 + 8 / 12, 1 + 11 / 12, 7 + 10.5 / 12,
                 1 + 11 / 12, 7 + 0.5 / 12, 1 + 11 / 12, 2 + 8 / 12, 1 + 11 / 12, 3 + 5 / 12]
        vals, info = solve_chain(meas, stated, 37 + 11.5 / 12)
        for v, t in zip(vals, truth):
            self.assertLess(abs(v - t) * 12, 0.6)
        self.assertEqual(info["warnings"], [])
        self.assertEqual(sum(s == "solved" for s in info["source"]), 5)

    def test_solve_chain_without_an_overall_leaves_gaps_unknown(self):
        from dimensions import solve_chain
        vals, info = solve_chain([4.0, 2.0], [4.0, None], None)
        self.assertIsNone(vals[1])
        self.assertTrue(info["warnings"])

    def test_scale_from_chain(self):
        from dimensions import scale_from_chain
        segs = [{"stated_ft": 10.0, "length_px": 750}, {"stated_ft": 20.0, "length_px": 1500},
                {"stated_ft": 8.0, "length_px": 596}]
        ppf, n = scale_from_chain(segs, None)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(ppf, 75.0, delta=1.0)

    def test_fmt(self):
        from dimensions import fmt
        self.assertEqual(fmt(15.25), "15'3\"")
        self.assertEqual(fmt(7.875), "7'10-1/2\"")
        self.assertEqual(fmt(None), "?")


# ANCHOR BEFORE 20261011_193301
def _chain_image(ppf=20.0, ticks=(100, 200, 240, 400, 700), y=200, overall_y=250, marks=False):
    """A bare dimension chain: one line, joints, and an overall line below it.

    No text. The readers are faked in these tests, so what matters is whether
    the geometry is found and the plumbing routes it correctly. With marks, each
    segment gets a glyph-sized blot above the line, because a piece with nothing
    text-shaped on it is never shown to a reader.
    """
    # ANCHOR AFTER 20261011_193301
    import numpy as np
    img = np.full((360, 820), 255, np.uint8)
    img[y - 1:y + 1, ticks[0]:ticks[-1]] = 0
    for x in ticks:
        img[y - 18:y + 18, x:x + 2] = 0
    img[overall_y - 1:overall_y + 1, ticks[0]:ticks[-1]] = 0
    for x in (ticks[0], ticks[-1]):
        img[overall_y - 16:overall_y + 16, x:x + 2] = 0
    # ANCHOR BEFORE 20261011_193302
    if marks:
        for a, b in zip(ticks, ticks[1:]):
            mid = (a + b) // 2
            img[y - 14:y - 6, mid - 6:mid + 6] = 0
    # ANCHOR AFTER 20261011_193302
    return img, ppf


class _FakeReader:
    """Returns whatever it was told to, and counts how often it was asked."""

    def __init__(self, kind, answers):
        self.kind, self.answers, self.calls, self.asked = kind, answers, 0, []
        self.widths = []

    def __call__(self, crop):
        self.calls += 1
        self.widths.append(crop.shape[1])
        return self.answers.pop(0) if self.answers else ""


class TestDimlines(unittest.TestCase):
    def test_detect_finds_the_joints(self):
        import dimlines
        img, ppf = _chain_image()
        chains = [c for c in dimlines.detect(img, ppf, "h")[0] if len(c["ticks"]) >= 4]
        self.assertTrue(chains)
        got = chains[0]["ticks"]
        for want, g in zip((100, 200, 240, 400, 700), got):
            self.assertLess(abs(g - want), 3, "joint at %d, found %d" % (want, g))

    def test_detect_pairs_the_overall_line(self):
        import dimlines
        img, ppf = _chain_image()
        chain = [c for c in dimlines.detect(img, ppf, "h")[0] if len(c["ticks"]) >= 4][0]
        cands = chain["overall_candidates"]
        self.assertTrue(cands, "the line below should be offered as the overall")
        self.assertLess(len(cands[0]["ticks"]), len(chain["ticks"]))

    def test_vertical_chains_are_found(self):
        import cv2
        import dimlines
        img, ppf = _chain_image()
        chains = dimlines.detect(cv2.transpose(img), ppf, "v")[0]
        self.assertTrue([c for c in chains if len(c["ticks"]) >= 4])

    def test_expensive_reader_only_sees_unresolved_singletons(self):
        """The whole point of the two tiers: repeated gaps are solved by
        arithmetic, so the slow reader is never asked about them."""
        import dimlines
        # ANCHOR BEFORE 20261011_193303
        img, ppf = _chain_image(ticks=(100, 200, 240, 400, 440, 700), marks=True)
        # ANCHOR AFTER 20261011_193303
        chain = [c for c in dimlines.detect(img, ppf, "h")[0] if len(c["ticks"]) >= 5][0]
        # segments measure 5, 2, 8, 2, 13 ft; the reader manages only the first
        cheap = _FakeReader("tesseract", ["5'0\""] + [""] * 20)
        pricey = _FakeReader("ollama", ["13'0\""] * 5)
        raws = dimlines.read_chain_raw(img, chain, ppf, cheap)
        out = dimlines.assemble(chain, raws, ppf, pricey, gray=img)
        self.assertEqual(out["segments"][0]["source"], "read")
        # the two 2 ft gaps are a cluster: arithmetic handles them, so the slow
        # reader is never asked about either one
        for i in (1, 3):
            self.assertNotEqual(out["segments"][i]["reader"], "ollama")
        # two singleton gaps plus at most one try at the overall
        self.assertLessEqual(pricey.calls, 3)

    def test_a_line_with_no_numbers_is_not_a_chain(self):
        import dimlines
        img, ppf = _chain_image()
        chain = [c for c in dimlines.detect(img, ppf, "h")[0] if len(c["ticks"]) >= 4][0]
        raws = dimlines.read_chain_raw(img, chain, ppf, _FakeReader("tesseract", []))
        out = dimlines.assemble(chain, raws, ppf, gray=img)
        self.assertFalse(dimlines.is_real_chain(out))
        self.assertEqual(out["readings"], 0)

    def test_solved_segments_are_never_labelled_read(self):
        import dimlines
        img, ppf = _chain_image()
        chain = [c for c in dimlines.detect(img, ppf, "h")[0] if len(c["ticks"]) >= 4][0]
        cheap = _FakeReader("tesseract", ["5'0\"", "2'0\"", "8'0\"", "", "30'0\""])
        raws = dimlines.read_chain_raw(img, chain, ppf, cheap)
        out = dimlines.assemble(chain, raws, ppf, gray=img)
        for s in out["segments"]:
            if s["source"] == "solved":
                self.assertIsNone(s["reader"])
                self.assertIsNone(s["raw_text"])
            if s["source"] == "read":
                self.assertIsNotNone(s["raw_text"])


class TestChainsFeedRooms(unittest.TestCase):
    """The chains are only worth finding if the rooms layer uses them."""

    def _data(self):
        # one horizontal and one vertical chain, each three solved segments of
        # 10 ft, on a sheet 400 px tall at 10 px/ft
        def chain(axis, y):
            segs, a = [], 100
            for _ in range(3):
                segs.append({"a": a, "b": a + 100, "length_px": 100, "measured_ft": 10.0,
                             "value_ft": 10.0, "printed": "10'0\"", "source": "read",
                             "reader": "tesseract", "raw_text": "10'0\"", "stated_ft": 10.0})
                a += 100
            return {"axis": axis, "y": y, "x0": 100, "x1": 400, "span_ft": 30.0,
                    "overall_ft": 30.0, "segments": segs}
        return {"chains": [chain("h", 380), chain("v", 20)]}

    def test_span_between_reads_off_the_chain(self):
        import dimlines
        idx = dimlines.chain_index(self._data(), height=400)
        got = dimlines.span_between(idx, "h", 100, 300, tol_px=15)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got[0], 20.0, places=3)

    def test_span_between_refuses_edges_that_miss_the_joints(self):
        import dimlines
        idx = dimlines.chain_index(self._data(), height=400)
        self.assertIsNone(dimlines.span_between(idx, "h", 150, 260, tol_px=5))

    def test_a_gap_spoils_only_the_distances_that_cross_it(self):
        import dimlines
        data = self._data()
        data["chains"][0]["segments"][1]["value_ft"] = None   # middle 10 ft unread
        idx = dimlines.chain_index(data, height=400)
        self.assertTrue(idx["h"], "the rest of the chain is still usable")
        # 100..200 sits entirely before the hole
        self.assertAlmostEqual(dimlines.span_between(idx, "h", 100, 200, 15)[0], 10.0, places=3)
        # 100..300 would have to cross it
        self.assertIsNone(dimlines.span_between(idx, "h", 100, 300, 15))

    def test_chain_lookup_scales_the_room_box_back_up(self):
        """Geometry runs on a shrunken copy; the joints are on the full sheet."""
        import seeded
        ask = seeded.chain_lookup(self._data(), full_height=400, scale=0.5, wppf=5.0)
        self.assertIsNotNone(ask)
        got = ask([50, 60, 150, 140])          # half-scale box = 100..300 full size
        self.assertAlmostEqual(got["width"][0], 20.0, places=3)

    def test_build_rooms_prefers_the_chain_over_a_printed_pair(self):
        import numpy as np
        import seeded
        lab = np.zeros((40, 40), np.int32)
        lab[10:30, 10:30] = 1
        def ask(bbox, seed=None):
            return {"width": (20.0, "h chain at 380", 0.0),
                    "depth": (20.0, "v chain at 20", 0.0)}
        rooms = seeded.build_rooms(lab, [("KITCHEN", 20, 20)], 1.0, [], [], chains=ask)
        self.assertEqual(rooms[0]["dims_source"], "dimension chain")
        self.assertEqual(rooms[0]["chain_refs"], ["h chain at 380", "v chain at 20"])
        self.assertFalse(rooms[0]["needs_review"])

    def test_one_side_on_a_chain_is_not_a_confirmation(self):
        import numpy as np
        import seeded
        lab = np.zeros((40, 40), np.int32)
        lab[10:30, 10:30] = 1
        def ask(bbox, seed=None):
            return {"width": (20.0, "h chain at 380", 0.0), "depth": None}
        rooms = seeded.build_rooms(lab, [("KITCHEN", 20, 20)], 1.0, [], [], chains=ask)
        self.assertTrue(rooms[0]["needs_review"])
        self.assertIn("only the width", rooms[0]["review_reason"])


class TestSelfCalibration(unittest.TestCase):
    """The scale must come from the drawing before any reading is judged against it."""

    def test_votes_ignore_short_segments(self):
        import dimlines
        raws = [[{"length_px": 150, "text": "2'0\""},     # 2 ft: crammed, no vote
                 {"length_px": 705, "text": "10'0\""},    # 70.5 px/ft
                 {"length_px": 1410, "text": "20'0\""},   # 70.5 px/ft
                 {"length_px": 1057, "text": "15'0\""},   # 70.5 px/ft
                 {"length_px": 846, "text": "12'0\""}]]
        votes = dimlines.chain_scale_votes(raws)
        self.assertEqual(len(votes), 4)
        got, used = dimlines.scale_from_votes(votes, 74.94)
        self.assertAlmostEqual(got, 70.5, delta=0.2)
        self.assertEqual(used, 4)

    def test_a_wild_misread_does_not_move_the_scale(self):
        import dimlines
        raws = [[{"length_px": 705, "text": "10'0\""}, {"length_px": 1410, "text": "20'0\""},
                 {"length_px": 1057, "text": "15'0\""}, {"length_px": 846, "text": "12'0\""},
                 {"length_px": 705, "text": "99'0\""}]]   # nonsense: 7 px/ft
        got, _ = dimlines.scale_from_votes(dimlines.chain_scale_votes(raws), 74.94)
        self.assertAlmostEqual(got, 70.5, delta=0.2)

    def test_too_few_votes_keeps_the_manifest(self):
        import dimlines
        raws = [[{"length_px": 705, "text": "10'0\""}, {"length_px": 1410, "text": "20'0\""}]]
        got, used = dimlines.scale_from_votes(dimlines.chain_scale_votes(raws), 74.94)
        self.assertIsNone(got, "two readings is not enough to overrule the manifest")

    def test_reading_is_separate_from_judging(self):
        """The same readings must be judgeable at more than one scale, without re-reading."""
        import dimlines
        img, _ = _chain_image()
        chain = [c for c in dimlines.detect(img, 20.0, "h")[0] if len(c["ticks"]) >= 4][0]
        cheap = _FakeReader("tesseract", ["5'0\"", "2'0\"", "8'0\"", "15'0\""])
        raws = dimlines.read_chain_raw(img, chain, 20.0, cheap)
        before = cheap.calls
        dimlines.assemble(chain, raws, 20.0, gray=img)
        dimlines.assemble(chain, raws, 18.0, gray=img)
        self.assertEqual(cheap.calls, before,
                         "judging again must not cost another reading")


class TestVersion(unittest.TestCase):
    def test_version_is_read_and_stamped(self):
        import runtrace
        v = runtrace.version()
        self.assertRegex(v, r"^\d+\.\d+\.\d+$")
        d = {}
        runtrace.stamp(d, "layer-a", "20260101-000000")
        self.assertEqual(d["version"], v)
        self.assertEqual(d["layers"]["layer-a"]["version"], v)
        self.assertEqual(runtrace.Trace("p.png", group="20260101-000000").data["version"], v)


class TestDimensionInBox(unittest.TestCase):
    """A length written inside a room is that room's dimension."""

    def _points(self):
        # kitchen spans x 100..500, y 200..600
        def h(ft, lo, hi, cross):
            return {"ft": ft, "axis": "h", "lo": lo, "hi": hi, "cross": cross,
                    "source": "read", "chain": "h@%d" % cross}

        def v(ft, lo, hi, cross):
            return {"ft": ft, "axis": "v", "lo": lo, "hi": hi, "cross": cross,
                    "source": "read", "chain": "v@%d" % cross}
        return [h(15.25, 100, 500, 580),        # the kitchen's own width
                h(2.0, 260, 300, 380),          # a closet jamb inside it
                v(23.58, 200, 600, 150),        # the kitchen's depth
                h(14.83, 700, 1100, 580)]       # the next room along

    def test_picks_the_length_written_in_the_room(self):
        import dimlines
        got = dimlines.dimension_in_box(self._points(), "h", (100, 200, 500, 600))
        self.assertAlmostEqual(got[0], 15.25)

    def test_ignores_a_length_written_in_the_next_room(self):
        import dimlines
        got = dimlines.dimension_in_box(self._points(), "h", (700, 200, 1100, 600))
        self.assertAlmostEqual(got[0], 14.83)

    def test_a_room_with_nothing_written_in_it_gets_nothing(self):
        import dimlines
        self.assertIsNone(dimlines.dimension_in_box(self._points(), "h", (1200, 200, 1500, 600)))

    def test_a_narrow_hall_cannot_claim_the_run_passing_over_it(self):
        """The 15'7" along the bottom of a bedroom is level with the hall beside it."""
        import dimlines
        hall = (240, 520, 420, 620)          # narrow, and the 15'3" run crosses it
        self.assertIsNone(dimlines.dimension_in_box(self._points(), "h", hall),
                          "a run that reaches past a room is not that room's dimension")

    def test_a_closet_jamb_is_not_the_room(self):
        import dimlines
        got = dimlines.dimension_in_box(self._points(), "h", (100, 200, 500, 600))
        self.assertAlmostEqual(got[0], 15.25, msg="must not pick the 2 ft jamb")

    def test_depth_comes_from_the_vertical_run(self):
        import dimlines
        got = dimlines.dimension_in_box(self._points(), "v", (100, 200, 500, 600))
        self.assertAlmostEqual(got[0], 23.58)

    def test_dimension_points_maps_vertical_chains_back_to_the_sheet(self):
        import dimlines
        data = {"chains": [{"axis": "v", "y": 150, "segments": [
            {"a": 100, "b": 300, "value_ft": 23.58, "source": "read"}]}]}
        pts = dimlines.dimension_points(data, height=1000)
        self.assertEqual(pts[0]["cross"], 150)
        self.assertEqual(pts[0]["lo"], 1000 - 1 - 300)
        self.assertEqual(pts[0]["hi"], 1000 - 1 - 100)


class TestShareAcrossBands(unittest.TestCase):
    """The bath knows its width, the bedroom beside it knows the depth they share."""

    def _rooms(self):
        # same band: both run from y=100 to y=300. BATH has a width, BEDROOM B a depth.
        # one band, 9 ft deep at 20 px/ft: y 100..280. The bath is 12'6" wide.
        return [
            {"label": "BEDROOM B", "bbox": [100, 100, 458, 280], "area_sqft": 160.0,
             "chain_sides": {"width": None, "depth": 9.0}, "needs_review": True,
             "review_reason": "only the depth of this room lands on a dimension chain"},
            {"label": "BATH", "bbox": [480, 100, 730, 280], "area_sqft": 108.0,
             "chain_sides": {"width": 12.5, "depth": None}, "needs_review": True,
             "review_reason": "only the width of this room lands on a dimension chain"},
        ]

    def test_the_bath_borrows_the_depth_of_its_band(self):
        from seeded import share_across_bands
        rooms = self._rooms()
        share_across_bands(rooms, ppf=20.0)
        bath = rooms[1]
        self.assertEqual(bath["stated_dims_ft"], [12.5, 9.0])
        self.assertEqual(bath["chain_borrowed"], {"depth": "BEDROOM B"})
        self.assertIn("borrowed", bath["dims_source"])
        self.assertFalse(bath["needs_review"])

    def test_a_room_in_another_band_does_not_lend(self):
        from seeded import share_across_bands
        rooms = self._rooms()
        rooms[0]["bbox"] = [100, 600, 458, 780]        # different band entirely
        share_across_bands(rooms, ppf=20.0)
        self.assertNotIn("stated_dims_ft", rooms[1])
        self.assertTrue(rooms[1]["needs_review"])

    def test_a_borrowed_size_that_disagrees_still_needs_review(self):
        from seeded import share_across_bands
        rooms = self._rooms()
        rooms[1]["bbox"] = [480, 100, 600, 280]        # region is far narrower than 12'6"
        share_across_bands(rooms, ppf=20.0)
        self.assertTrue(rooms[1]["needs_review"])
        self.assertIn("measures", rooms[1]["review_reason"])

    def test_a_room_that_is_not_a_rectangle_still_passes(self):
        """A stair or closet cut out of a room lowers its area, not its width or depth."""
        from seeded import share_across_bands
        rooms = self._rooms()
        rooms[1]["area_sqft"] = 70.0                   # 62% of the envelope: notched, not wrong
        share_across_bands(rooms, ppf=20.0)
        self.assertFalse(rooms[1]["needs_review"])
        self.assertAlmostEqual(rooms[1]["fill_pct"], 62.2, delta=1.0)

    def test_a_room_that_already_knows_both_sides_is_untouched(self):
        from seeded import share_across_bands
        rooms = self._rooms()
        rooms[1]["chain_sides"] = {"width": 12.5, "depth": 9.0}
        rooms[1]["dims_source"] = "dimension chain"
        share_across_bands(rooms, ppf=20.0)
        self.assertEqual(rooms[1]["dims_source"], "dimension chain")
        self.assertNotIn("chain_borrowed", rooms[1])


class TestSharedWallDimension(unittest.TestCase):
    """A dimension drawn on the wall between two rooms can serve either of them."""

    def test_either_room_may_claim_a_run_on_their_shared_wall(self):
        import dimlines
        # Bed Room A sits above Bed Room B; the 17'11" is drawn on the wall between
        # them, so it lies on A's bottom edge and B's top edge at once.
        run = [{"ft": 17.92, "axis": "h", "lo": 100, "hi": 900, "cross": 500,
                "source": "read", "chain": "h@500"}]
        a = (100, 100, 900, 500)
        b = (100, 500, 900, 900)
        pad = 30
        self.assertAlmostEqual(dimlines.dimension_in_box(run, "h", a, pad)[0], 17.92)
        self.assertAlmostEqual(dimlines.dimension_in_box(run, "h", b, pad)[0], 17.92)

    def test_a_column_neighbour_lends_its_width(self):
        from seeded import share_across_bands
        rooms = [
            {"label": "BED ROOM A", "bbox": [100, 100, 458, 400], "area_sqft": 268.0,
             "chain_sides": {"width": None, "depth": 15.0}, "needs_review": True},
            {"label": "BED ROOM B", "bbox": [100, 420, 458, 700], "area_sqft": 160.0,
             "chain_sides": {"width": 17.92, "depth": None}, "needs_review": True},
        ]
        share_across_bands(rooms, ppf=20.0)
        self.assertEqual(rooms[0]["chain_borrowed"], {"width": "BED ROOM B"})
        self.assertEqual(rooms[0]["stated_dims_ft"], [17.92, 15.0])

    def test_a_neighbour_of_a_different_width_lends_nothing(self):
        from seeded import share_across_bands
        rooms = [
            {"label": "BED ROOM A", "bbox": [100, 100, 700, 400], "area_sqft": 200.0,
             "chain_sides": {"width": None, "depth": 15.0}, "needs_review": True},
            {"label": "BED ROOM B", "bbox": [100, 420, 458, 700], "area_sqft": 160.0,
             "chain_sides": {"width": 17.92, "depth": None}, "needs_review": True},
        ]
        share_across_bands(rooms, ppf=20.0)
        self.assertNotIn("stated_dims_ft", rooms[0],
                         "rooms that do not run between the same walls have different widths")


class TestTraceTool(unittest.TestCase):
    """The trace reader must survive whatever a run threw into the file."""

    def _run(self, args, data=None, tmp=None):
        import io
        import json as j
        import runpy
        import sys
        import tempfile
        from contextlib import redirect_stdout
        d = tmp or tempfile.mkdtemp()
        p = os.path.join(d, "trace_20260101-000000-abc.json")
        with open(p, "w") as f:
            j.dump(data if data is not None else {}, f)
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace.py")
        buf, argv = io.StringIO(), sys.argv
        try:
            sys.argv = ["trace.py", p] + args
            with redirect_stdout(buf):
                runpy.run_path(script, run_name="__main__")
        finally:
            sys.argv = argv
        return buf.getvalue()

    def test_an_empty_trace_does_not_crash(self):
        out = self._run(["--all"])
        self.assertIn("summary", out)

    def test_rooms_are_tabulated_with_a_count(self):
        data = {"plan": "p.png", "rooms": [
            {"label": "KITCHEN", "area_sqft": 166.6, "measured_ft": [12.0, 15.9],
             "stated_dims_ft": [15.25, 23.58], "needs_review": False},
            {"label": "HALL", "area_sqft": 35.9, "measured_ft": [6.4, 7.5],
             "chain_sides": {"width": None, "depth": 7.5}, "needs_review": True,
             "review_reason": "only the depth was found"}]}
        out = self._run(["--rooms"], data)
        self.assertIn("15'3\" x 23'7\"", out)
        self.assertIn("d 7'6\"", out)
        self.assertIn("1 of 2 agree", out)

    def test_chains_show_each_segment_and_what_was_rejected(self):
        data = {"plan": "p.png", "chains": [
            {"axis": "h", "y": 100, "span_ft": 38.0, "overall_ft": 37.96,
             "readings": 1, "solved": 1, "unknown": 0,
             "segments": [{"value_ft": 4.667, "measured_ft": 4.39, "source": "read",
                           "raw_text": "4'8\""},
                          {"value_ft": 1.917, "measured_ft": 2.18, "source": "solved"}],
             "rejected": [{"stated_ft": 12.0, "measured_ft": 12.52, "off_pct": 4.2}]}]}
        out = self._run(["--chains"], data)
        self.assertIn("overall 38'0\"", out)   # 37.96 ft rounds to 38'0"
        self.assertIn("seg00", out)
        self.assertIn("rejected", out)

    def test_a_file_that_is_not_json_is_skipped_not_fatal(self):
        import tempfile
        d = tempfile.mkdtemp()
        open(os.path.join(d, "trace_bad.json"), "w").write("{not json")
        out = self._run(["--rooms"], {"rooms": []}, tmp=d)
        self.assertIn("rooms", out)    # the good file still printed


class TestProgress(unittest.TestCase):
    def test_it_reports_without_a_terminal(self):
        import io
        from progress import Progress
        buf = io.StringIO()
        bar = Progress(20, "rooms", every=10, stream=buf)
        for _ in range(20):
            bar.tick()
        took = bar.finish()
        out = buf.getvalue()
        self.assertIn("10/20", out)
        self.assertIn("ETA", out)
        self.assertIn("done", out)
        self.assertGreaterEqual(took, 0)

    def test_quiet_says_nothing(self):
        import io
        from progress import Progress
        buf = io.StringIO()
        bar = Progress(5, "x", stream=buf, quiet=True)
        bar.tick()
        bar.finish()
        self.assertEqual(buf.getvalue(), "")

    def test_zero_work_does_not_divide_by_zero(self):
        import io
        from progress import Progress
        Progress(0, "none", stream=io.StringIO()).finish()

    def test_clock(self):
        from progress import Progress
        self.assertEqual(Progress._clock(59), "0:59")
        self.assertEqual(Progress._clock(1598), "26:38")
        self.assertEqual(Progress._clock(3700), "1:01:40")


class TestRoomFieldsReachBothViews(unittest.TestCase):
    """results.json and the trace must tell the same story about one run."""

    def test_measured_and_chain_sides_are_on_the_room_and_in_the_trace(self):
        import numpy as np
        import seeded
        from runtrace import Trace
        lab = np.zeros((40, 40), np.int32)
        lab[10:30, 10:30] = 1

        def ask(bbox, seed=None):
            return {"width": (20.0, "h chain at 1", 0.0), "depth": None}

        tr = Trace("p.png", group="20260101-000000")
        rooms = seeded.build_rooms(lab, [("KITCHEN", 20, 20)], 1.0, [], [],
                                   trace=tr, chains=ask)
        rm = rooms[0]
        self.assertEqual(len(rm["measured_ft"]), 2, "the room carries its own size")
        self.assertEqual(rm["chain_sides"]["width"], 20.0)
        ent = tr.data["rooms"][0]
        self.assertEqual(ent["measured_ft"], rm["measured_ft"])
        self.assertEqual(ent["chain_sides"], rm["chain_sides"])


class TestThreeJointChains(unittest.TestCase):
    """An interior chain runs wall to wall, and a wall is not a tick, so its
    outermost joint is invisible. Three joints must be enough."""

    def test_a_three_joint_chain_is_a_chain(self):
        import dimlines
        img, ppf = _chain_image(ticks=(100, 400, 700))      # two segments, 15 ft each
        chains = dimlines.detect(img, ppf, "h")[0]
        self.assertTrue([c for c in chains if len(c["ticks"]) == 3],
                        "a wall-to-wall interior chain has three visible joints")

    def test_both_segments_must_read_for_it_to_count(self):
        """With only two segments there is no room for a silent one."""
        import dimlines
        # ANCHOR BEFORE 20261011_193304
        img, ppf = _chain_image(ticks=(100, 400, 700), marks=True)
        # ANCHOR AFTER 20261011_193304
        chain = [c for c in dimlines.detect(img, ppf, "h")[0] if len(c["ticks"]) == 3][0]
        one = _FakeReader("tesseract", ["15'0\"", ""])
        raws = dimlines.read_chain_raw(img, chain, ppf, one)
        self.assertFalse(dimlines.is_real_chain(dimlines.assemble(chain, raws, ppf, gray=img)))
        both = _FakeReader("tesseract", ["15'0\"", "15'0\""])
        raws = dimlines.read_chain_raw(img, chain, ppf, both)
        self.assertTrue(dimlines.is_real_chain(dimlines.assemble(chain, raws, ppf, gray=img)))


class TestAnchorOnTheRoomName(unittest.TestCase):
    """When the traced outline is wrong, the name is still in the right place."""

    def _points(self):
        def h(ft, lo, hi, cross):
            return {"ft": ft, "axis": "h", "lo": lo, "hi": hi, "cross": cross,
                    "source": "read", "chain": "h@%d" % cross}

        def v(ft, lo, hi, cross):
            return {"ft": ft, "axis": "v", "lo": lo, "hi": hi, "cross": cross,
                    "source": "read", "chain": "v@%d" % cross}
        # the kitchen's own 15'3" and 23'7", plus a longer run two rooms away
        return [h(15.25, 180, 560, 900), v(23.58, 330, 900, 150), h(27.33, 900, 1580, 900)]

    def test_a_run_crossing_the_name_is_the_room_s_dimension(self):
        import dimlines
        got = dimlines.dimension_over_point(self._points(), "h", 300, 870, near_px=400)
        self.assertAlmostEqual(got[0], 15.25)

    def test_nearest_wins_not_longest(self):
        """With no outline to bound it, the longer run elsewhere must not be taken."""
        import dimlines
        got = dimlines.dimension_over_point(self._points(), "h", 300, 870, near_px=400)
        self.assertNotAlmostEqual(got[0], 27.33)

    def test_a_run_that_misses_the_name_is_not_taken(self):
        import dimlines
        self.assertIsNone(
            dimlines.dimension_over_point(self._points(), "h", 700, 870, near_px=400))

    def test_a_run_too_far_from_the_name_is_not_taken(self):
        import dimlines
        self.assertIsNone(
            dimlines.dimension_over_point(self._points(), "h", 300, 200, near_px=100))

    def test_depth_uses_the_other_axis(self):
        import dimlines
        got = dimlines.dimension_over_point(self._points(), "v", 300, 600, near_px=400)
        self.assertAlmostEqual(got[0], 23.58)


class TestLoneDimensions(unittest.TestCase):
    """Most room dimensions are one line between two arrowheads, not a chain."""

    def test_a_bare_line_costs_no_reader_call(self):
        """Most lines on a drawing carry no number; none of them should be read."""
        import dimlines
        img, ppf = _chain_image(ticks=(100, 700))
        line = dimlines.detect(img, ppf, "h")[1][0]
        r = _FakeReader("t", ["30'0\""])
        self.assertIsNone(dimlines.read_single(img, line, ppf, r, "h", 360))
        self.assertEqual(r.calls, 0)

    def test_a_two_tick_line_is_offered_as_a_single(self):
        import dimlines
        img, ppf = _chain_image(ticks=(100, 700))     # one span, 30 ft, no joints between
        chains, singles = dimlines.detect(img, ppf, "h")
        self.assertFalse([c for c in chains if len(c["ticks"]) >= 3])
        self.assertTrue(singles, "a lone dimension line must still be a candidate")

    def test_it_is_believed_only_if_it_measures_what_it_says(self):
        import cv2
        import dimlines
        img, ppf = _chain_image(ticks=(100, 700))     # 600 px at 20 px/ft = 30 ft
        # a dimension line carries its number; a bare line is skipped before any
        # reader is called, which is deliberate on a sheet full of unlabelled lines
        cv2.putText(img, "30'0\"", (360, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        line = dimlines.detect(img, ppf, "h")[1][0]
        right = dimlines.read_single(img, line, ppf, _FakeReader("t", ["30'0\""]), "h", 360)
        self.assertIsNotNone(right)
        self.assertAlmostEqual(right["ft"], 30.0)
        wrong = dimlines.read_single(img, line, ppf, _FakeReader("t", ["12'0\""]), "h", 360)
        self.assertIsNone(wrong, "a line that does not measure what it says is not a dimension")

    def test_lone_dimensions_reach_the_matching_rules(self):
        import dimlines
        data = {"single_dimensions": [
            {"ft": 23.58, "axis": "v", "lo": 300, "hi": 855, "cross": 265,
             "source": "read", "chain": "v dimension at 265"}], "chains": []}
        pts = dimlines.dimension_points(data, height=2076)
        self.assertEqual(len(pts), 1)
        got = dimlines.dimension_over_point(pts, "v", 300, 500, near_px=200)
        self.assertAlmostEqual(got[0], 23.58)


class TestRuleset(unittest.TestCase):
    """Tuning lives in rules.json. The engine holds no constants of its own."""

    def test_every_rule_has_a_description(self):
        import ruleset
        missing = [p for p, _, why in ruleset.all_rules() if not why]
        self.assertEqual(missing, [], "a rule nobody can explain is a magic number")

    def test_values_come_from_the_file(self):
        import ruleset
        self.assertAlmostEqual(ruleset.rule("matching.side_tol_rel"), 0.08)
        self.assertEqual(ruleset.rule("detection.min_chain_ticks"), 3)

    def test_a_nonsense_value_is_refused_not_used(self):
        """A tolerance typed as 8 instead of 0.08 would accept every room."""
        import json
        import tempfile
        from pathlib import Path
        import ruleset
        d = tempfile.mkdtemp()
        bad = Path(d) / "rules.json"
        bad.write_text(json.dumps({"matching": {"side_tol_rel": {"value": 8}}}))
        old = ruleset.RULES_FILE
        try:
            ruleset.RULES_FILE = bad
            ruleset.reload()
            with self.assertRaises(ValueError):
                ruleset.rule("matching.side_tol_rel")
        finally:
            ruleset.RULES_FILE = old
            ruleset.reload()

    def test_a_missing_file_falls_back_to_defaults(self):
        from pathlib import Path
        import ruleset
        old = ruleset.RULES_FILE
        try:
            ruleset.RULES_FILE = Path("/nonexistent/rules.json")
            ruleset.reload()
            self.assertAlmostEqual(ruleset.rule("matching.side_tol_rel"), 0.08)
        finally:
            ruleset.RULES_FILE = old
            ruleset.reload()

    def test_an_unknown_rule_is_an_error_not_a_silent_none(self):
        import ruleset
        with self.assertRaises(KeyError):
            ruleset.rule("matching.no_such_thing")

    def test_the_lexicon_marks_what_it_was_fitted_to(self):
        """MEMORIAL is not a room type; it is on the sample sheet. That must be visible."""
        import json
        from pathlib import Path
        lex = json.loads((Path("lexicon.json")).read_text())
        self.assertIn("MEMORIAL", lex["period_or_specific"])
        self.assertNotIn("MEMORIAL", lex["rooms"])


class TestSingleOutcomes(unittest.TestCase):
    """Why a lone dimension was not believed has to be visible: unreadable and
    disagreed have opposite fixes."""

    def test_each_outcome_is_counted(self):
        import cv2
        import dimlines
        img, ppf = _chain_image(ticks=(100, 700))
        cv2.putText(img, "30'0\"", (360, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        line = dimlines.detect(img, ppf, "h")[1][0]

        why = {}
        dimlines.read_single(img, line, ppf, _FakeReader("t", ["30'0\""]), "h", 360, why)
        self.assertEqual(why.get("believed"), 1)

        why = {}
        dimlines.read_single(img, line, ppf, _FakeReader("t", ["12'0\""]), "h", 360, why)
        self.assertEqual(why.get("disagreed"), 1)

        why = {}
        dimlines.read_single(img, line, ppf, _FakeReader("t", ["POST"]), "h", 360, why)
        self.assertEqual(why.get("unreadable"), 1)

        why = {}
        bare, _ = _chain_image(ticks=(100, 700))
        bareline = dimlines.detect(bare, ppf, "h")[1][0]
        dimlines.read_single(bare, bareline, ppf, _FakeReader("t", []), "h", 360, why)
        self.assertEqual(why.get("nothing written on it"), 1)


class TestMatchLog(unittest.TestCase):
    """A count with no reasons is not traceable: every room must say what it tried."""

    def test_each_rule_reports_why_it_did_not_answer(self):
        import dimlines
        why = []
        dimlines.span_between({"h": []}, "h", 0, 100, 10, why)
        dimlines.dimension_in_box([], "h", (0, 0, 100, 100), 0, reasons=why)
        dimlines.dimension_over_point([], "h", 50, 50, 10, why)
        self.assertEqual(len(why), 3, "all three rules must account for themselves")
        self.assertTrue(any("joint" in w for w in why))
        self.assertTrue(any("in the room" in w for w in why))
        self.assertTrue(any("at the name" in w for w in why))

    def test_the_reason_says_how_near_the_run_got(self):
        import dimlines
        pts = [{"ft": 15.0, "axis": "h", "lo": 0, "hi": 300, "cross": 900,
                "source": "read", "chain": "h@900"}]
        why = []
        dimlines.dimension_over_point(pts, "h", 150, 100, near_px=50, reasons=why)
        self.assertIn("800", why[0], "it must say how far away the nearest run was")

    def test_a_room_carries_its_match_log(self):
        import numpy as np
        import seeded
        lab = np.zeros((40, 40), np.int32)
        lab[10:30, 10:30] = 1
        ask = seeded.chain_lookup(
            {"chains": [], "single_dimensions": [
                {"ft": 9.9, "axis": "h", "lo": 0, "hi": 400, "cross": 900,
                 "source": "read", "chain": "h@900"}]},
            full_height=1000, scale=1.0, wppf=1.0)
        rooms = seeded.build_rooms(lab, [("KITCHEN", 20, 20)], 1.0, [], [], chains=ask)
        self.assertTrue(rooms[0]["match_log"])
        self.assertTrue(any("width" in line for line in rooms[0]["match_log"]))


class TestSingleCandidateFiltering(unittest.TestCase):
    """A sheet is full of long thin lines that are not dimensions."""

    def _line(self, img, y, x0, x1, ticks=()):
        img[y:y + 2, x0:x1] = 0
        for x in ticks:
            img[y - 18:y + 18, x:x + 2] = 0

    def test_a_rule_with_no_marked_ends_is_not_a_dimension(self):
        """A title-block rule is the same length and thickness and has bare ends."""
        import numpy as np
        import dimlines
        img = np.full((400, 900), 255, np.uint8)
        self._line(img, 100, 100, 700)                      # bare rule
        self._line(img, 250, 100, 700, ticks=(100, 698))    # dimension line
        singles = dimlines.detect(img, 20.0, "h")[1]
        rows = sorted(l["y"] for l in singles)
        self.assertEqual(len(rows), 1, "only the line with both ends marked counts")
        self.assertAlmostEqual(rows[0], 250, delta=2)   # the row is the line's centre

    def test_prose_on_a_line_is_rejected_before_any_reader_runs(self):
        import cv2
        import numpy as np
        import dimlines
        img = np.full((400, 900), 255, np.uint8)
        self._line(img, 250, 100, 700, ticks=(100, 698))
        cv2.putText(img, "DEPARTMENT OF THE INTERIOR", (110, 246),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        line = dimlines.detect(img, 20.0, "h")[1][0]
        r, why = _FakeReader("t", ["30'0\""]), {}
        self.assertIsNone(dimlines.read_single(img, line, 20.0, r, "h", 400, why))
        self.assertEqual(r.calls, 0, "prose must cost no reader call")
        self.assertIn("prose", " ".join(why))


class TestDebugDoesNotClobber(unittest.TestCase):
    def test_a_debug_run_writes_beside_the_real_output(self):
        """A four-minute diagnostic must not destroy a twenty-five-minute result."""
        import dimlines
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "dimlines.py")).read()
        self.assertIn('suffix = "_dimlines_debug" if DEBUG["on"] else "_dimlines"', src)
        self.assertIn('suffix + ".json"', src,
                      "the written name must go through the suffix, not be hardcoded")


# ANCHOR BEFORE 20261011_121319
class TestScore(unittest.TestCase):
    """The scorer is where a false agreement gets caught, so it is tested on one."""

    def _run(self, rooms):
        import score
        d = tempfile.mkdtemp()
        truth = os.path.join(d, "truth.csv")
        with open(truth, "w") as f:
            f.write("plan,room,printed_w_ft,printed_d_ft,w_source,d_source,notes,"
                    "box_x0,box_y0,box_x1,box_y1\n"
                    "p.png,DINING ROOM,14.833,24.333,printed,printed,,0,0,1000,1000\n"
                    "p.png,BATH,12.5,9.0,printed,band,,1000,0,2000,1000\n"
                    "p.png,LATER KITCHEN,,,none,none,,0,1000,1000,2000\n")
        trace = {"trace_id": "t", "plan": "p.png", "params": {"pixels_per_foot_given": 70.0},
                 "stages": [{"stage": "segment", "working_ppf": 14.0}], "warnings": [],
                 "rooms": [dict(room=i + 1, area_sqft=100.0, **r) for i, r in enumerate(rooms)]}
        results = {"plans": [{"plan": "p.png", "trace_file": "trace_t.json",
                              "rooms": [{"id": i + 1, "needs_review": not r.get("verified")}
                                        for i, r in enumerate(rooms)]}]}
        with open(os.path.join(d, "trace_t.json"), "w") as f:
            json.dump(trace, f)
        with open(os.path.join(d, "results.json"), "w") as f:
            json.dump(results, f)
        out = os.path.join(d, "s.json")
        code = score.main(["--results", os.path.join(d, "results.json"),
                           "--truth", truth, "--json", out])
        return code, json.load(open(out))[0]

    def test_matches_by_seed_not_by_label(self):
        # The seed is in working pixels; at 70/14 it lands at (500, 500), inside DINING ROOM.
        code, s = self._run([{"label": "LIVING 7", "seed": [100, 100],
                              "measured_ft": [14.79, 24.5]}])
        self.assertEqual(code, 0)
        self.assertEqual(s["regions"][0]["room"], "DINING ROOM")
        self.assertEqual(s["rooms_found"], 1)
        self.assertEqual(s["accuracy"]["printed"]["width"]["n"], 1)

    def test_band_sides_are_never_pooled_with_printed(self):
        code, s = self._run([{"label": "BATH", "seed": [300, 100], "measured_ft": [12.86, 9.43]}])
        self.assertEqual(s["accuracy"]["printed"]["depth"], None)
        self.assertEqual(s["accuracy"]["band"]["depth"]["n"], 1)

    def test_false_agreement_is_named_and_fails_the_run(self):
        # Agreeing on a room the drawing prints no size for is a false agreement.
        code, s = self._run([{"label": "LATER KITCHEN", "seed": [100, 300], "verified": True,
                              "measured_ft": [27.4, 15.3]}])
        self.assertEqual(code, 1)
        self.assertEqual(len(s["agreed_wrong"]), 1)
        self.assertEqual(len(s["agreed_correct"]), 0)

    def test_region_outside_every_room_is_invented(self):
        code, s = self._run([{"label": "(unlabelled 1)", "seed": [900, 900],
                              "measured_ft": [5.0, 5.0], "verified": True}])
        self.assertEqual(len(s["invented"]), 1)
        self.assertEqual(code, 1, "an agreeing invented region is a false agreement")

    def test_repo_truth_has_a_source_for_every_side(self):
        import score
        here = os.path.dirname(os.path.abspath(__file__))
        rows = score.load_truth(os.path.join(here, "truth.csv"))
        self.assertTrue(rows)
        for r in rows:
            self.assertIn(r["w_source"], score.SOURCES)
            self.assertIn(r["d_source"], score.SOURCES)
# ANCHOR AFTER 20261011_121319


# ANCHOR BEFORE 20261011_122046
class TestEmptyDimensionInputIsSaid(unittest.TestCase):
    """A run with no dimension data looked exactly like a run where matching was tried
    and failed. Both the missing file and the empty one must leave a warning."""

    def _warnings(self, dim_file):
        import cv2
        import numpy as np
        here = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as d:
            plans = os.path.join(d, "plans")
            os.makedirs(os.path.join(d, "out", "out"))
            os.makedirs(plans)
            cv2.imwrite(os.path.join(plans, "blank.png"), np.full((200, 200, 3), 255, np.uint8))
            man = os.path.join(d, "manifest.csv")
            open(man, "w").write("plan,pixels_per_foot\nblank.png,10.0\n")
            if dim_file is not None:
                json.dump(dim_file, open(os.path.join(d, "out", "out", "blank_dimlines.json"), "w"))
            subprocess.run([sys.executable, "seeded.py", "--plans", plans, "--manifest", man,
                            "--out", os.path.join(d, "out"), "--fast", "--no-vertical",
                            "--no-dimension-reader"],
                           cwd=here, check=True, capture_output=True, text=True)
            trace = [f for f in os.listdir(os.path.join(d, "out")) if f.startswith("trace_")][0]
            return json.load(open(os.path.join(d, "out", trace)))["warnings"]

    def test_missing_file_is_warned(self):
        w = self._warnings(None)
        self.assertTrue(any("no dimension file" in m for m in w), w)

    def test_empty_file_is_warned(self):
        w = self._warnings({"plan": "blank.png", "chains": [], "single_dimensions": []})
        self.assertTrue(any("holds no chains" in m and "segmenter only" in m for m in w), w)
# ANCHOR AFTER 20261011_122046


# ANCHOR BEFORE 20261011_130047
class TestDoorwaysClosed(unittest.TestCase):
    """Rooms on the sample sheet flowed into the next space through doorways wider than
    the 2 ft gap the wall dilation closes. A door-sized gap in a straight wall must be
    shut in the mask; an opening wider than any door must stay open."""

    def _wall_with_gap(self, gap_ft, ppf=20.0):
        import numpy as np
        img = np.full((400, 900), 255, np.uint8)
        left_end = 300
        img[200:203, 50:left_end] = 0
        img[200:203, left_end + int(gap_ft * ppf):850] = 0
        return img, left_end + int(gap_ft * ppf / 2)

    def test_a_door_sized_gap_is_bridged(self):
        import seeded
        img, mid = self._wall_with_gap(2.5)
        self.assertTrue(seeded.line_map(img, 20.0)[201, mid] > 0)

    def test_an_opening_wider_than_a_door_stays_open(self):
        import seeded
        img, mid = self._wall_with_gap(5.0)
        self.assertEqual(seeded.line_map(img, 20.0)[201, mid], 0)
# ANCHOR AFTER 20261011_130047


# ANCHOR BEFORE 20261011_135305
class TestDerivedScaleIsReportedNotUsed(unittest.TestCase):
    """A run measured everything 2.4% small because it used 73.14 px/ft derived from its
    own readings instead of the manifest's 74.94, which matches the printed overall."""

    def test_choose_scale_keeps_the_manifest_and_says_why(self):
        import dimlines
        ppf, note = dimlines.choose_scale(74.94, 73.14, 25)
        self.assertEqual(ppf, 74.94)
        self.assertIn("73.14", note)
        self.assertIn("manifest was used", note)

    def test_choose_scale_uses_the_drawing_only_when_asked(self):
        import dimlines
        self.assertEqual(dimlines.choose_scale(74.94, 73.14, 25, use_drawing=True)[0], 73.14)
        self.assertEqual(dimlines.choose_scale(74.94, 74.5, 25), (74.94, None))

    def _run(self, *flags):
        import cv2
        import numpy as np
        here = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as d:
            plans = os.path.join(d, "plans")
            os.makedirs(os.path.join(d, "out", "out"))
            os.makedirs(plans)
            cv2.imwrite(os.path.join(plans, "blank.png"), np.full((200, 200, 3), 255, np.uint8))
            man = os.path.join(d, "manifest.csv")
            open(man, "w").write("plan,pixels_per_foot\nblank.png,10.0\n")
            json.dump({"plan": "blank.png", "pixels_per_foot": 9.0, "pixels_per_foot_from_drawing": 9.0,
                       "readings_used_for_scale": 12, "chains": [], "single_dimensions": []},
                      open(os.path.join(d, "out", "out", "blank_dimlines.json"), "w"))
            subprocess.run([sys.executable, "seeded.py", "--plans", plans, "--manifest", man,
                            "--out", os.path.join(d, "out"), "--fast", "--no-vertical",
                            "--no-dimension-reader"] + list(flags),
                           cwd=here, check=True, capture_output=True, text=True)
            trace = [f for f in os.listdir(os.path.join(d, "out")) if f.startswith("trace_")][0]
            return json.load(open(os.path.join(d, "out", trace)))

    def test_rooms_layer_keeps_the_manifest_scale_and_still_reports(self):
        t = self._run()
        self.assertEqual(t["params"]["pixels_per_foot_used"], 10.0)
        self.assertTrue(any("imply 9.00" in w and "manifest was used" in w for w in t["warnings"]),
                        t["warnings"])

    def test_rooms_layer_opts_in(self):
        self.assertEqual(self._run("--use-drawing-scale")["params"]["pixels_per_foot_used"], 9.0)
        self.assertEqual(self._run("--use-drawing-scale", "--trust-manifest")
                         ["params"]["pixels_per_foot_used"], 10.0)
# ANCHOR AFTER 20261011_135305


# ANCHOR BEFORE 20261011_135536
class TestScoreContainment(unittest.TestCase):
    """The attic merged with the roof through a broken wall and scored better, because its
    depth is not printed. A region reaching past its room's box must be flagged, and if it
    is marked agreeing, that is a false agreement."""

    def _run(self, bbox, verified=False):
        import score
        d = tempfile.mkdtemp()
        truth = os.path.join(d, "truth.csv")
        # ATTIC: width printed, depth not; box on the inner wall faces, in sheet pixels.
        with open(truth, "w") as f:
            f.write("plan,room,printed_w_ft,printed_d_ft,w_source,d_source,notes,"
                    "box_x0,box_y0,box_x1,box_y1\n"
                    "p.png,ATTIC,27.667,,printed,none,,5357,1456,7376,1860\n")
        k = 74.94 / 14.0
        work = [round(v / k) for v in bbox]
        w_ft = (work[2] - work[0] + 1) / 14.0
        trace = {"trace_id": "t", "plan": "p.png",
                 "params": {"pixels_per_foot_given": 74.94, "pixels_per_foot_used": 74.94},
                 "stages": [{"stage": "segment", "working_ppf": 14.0}], "warnings": [],
                 "rooms": [{"room": 1, "label": "ATTIC", "area_sqft": 400.0,
                            "seed": [round(6300 / k), round(1700 / k)], "bbox": work,
                            "measured_ft": [w_ft, (work[3] - work[1] + 1) / 14.0],
                            "verified": verified}]}
        with open(os.path.join(d, "trace_t.json"), "w") as f:
            json.dump(trace, f)
        with open(os.path.join(d, "results.json"), "w") as f:
            json.dump({"plans": [{"plan": "p.png", "trace_file": "trace_t.json",
                                  "rooms": [{"id": 1, "needs_review": not verified}]}]}, f)
        out = os.path.join(d, "s.json")
        code = score.main(["--results", os.path.join(d, "results.json"), "--truth", truth,
                           "--json", out])
        return code, json.load(open(out))[0]

    def test_attic_merged_with_the_roof_is_flagged(self):
        # The merged region: right width, but 10 ft of roof above the attic's top wall.
        code, s = self._run([5289, 674, 7430, 1895])
        self.assertEqual(s["accuracy"]["printed"]["width"]["n"], 1)
        self.assertLess(s["accuracy"]["printed"]["width"]["median_pct"], 8.0)
        self.assertEqual([r["room"] for r in s["outside"]], ["ATTIC"])
        self.assertGreater(s["outside"][0]["reach_ft"]["top"], 10.0)
        self.assertEqual(s["within_tol_and_contained"], [])

    def test_flagged_region_marked_agreeing_is_a_false_agreement(self):
        code, s = self._run([5289, 674, 7430, 1895], verified=True)
        self.assertEqual(code, 1)
        self.assertIn("outside the room", s["agreed_wrong"][0]["why"])

    def test_a_region_on_its_walls_is_not_flagged(self):
        code, s = self._run([5358, 1452, 7380, 1864])
        self.assertEqual(s["outside"], [])
        self.assertEqual(s["within_tol_and_contained"], ["ATTIC"])
# ANCHOR AFTER 20261011_135536


# ANCHOR BEFORE 20261011_140528
class TestCornerDoorClosed(unittest.TestCase):
    """Door 13 on the sample sheet sits at a corner with a 0.88 ft jamb above it. The wall
    map dropped the jamb, so there was nothing to close the door from and both rear
    bedrooms flowed onto the landing."""

    def _corner(self, gap_ft, ppf=20.0):
        import numpy as np
        img = np.full((500, 500), 255, np.uint8)
        img[99:102, 50:451] = 0                        # wall across the top
        jamb_end = 100 + int(0.9 * ppf)
        img[100:jamb_end, 399:402] = 0                 # short jamb at the corner
        img[jamb_end + int(gap_ft * ppf):450, 399:402] = 0
        return img, jamb_end + int(gap_ft * ppf / 2)

    def test_a_door_at_a_corner_is_closed(self):
        import seeded
        img, mid = self._corner(2.5)
        self.assertTrue(seeded.line_map(img, 20.0)[mid, 400] > 0)

    def test_an_opening_wider_than_a_door_stays_open_at_a_corner(self):
        import seeded
        img, mid = self._corner(4.0)
        self.assertEqual(seeded.line_map(img, 20.0)[mid, 400], 0)
# ANCHOR AFTER 20261011_140528


# ANCHOR BEFORE 20261011_142247
class TestMatchOnlyWhereADimensionCanBelong(unittest.TestCase):
    """A first-floor chain was matched to a second-floor bedroom, and the attic's width
    to the room below it. Two plans side by side at 10 px/ft: plan A x 50..350,
    plan B x 500..850, both y 100..300 on a sheet 400 px tall."""

    def _sheet(self):
        import numpy as np
        lab = np.zeros((400, 900), np.int32)
        lab[100:300, 50:350] = 1
        lab[100:300, 500:850] = 2
        return lab, 3, np.zeros((400, 900), np.uint8)

    def _vertical_chain(self, x):
        # one read 20 ft segment from sheet y 100 to 300; vertical chains live in the
        # rotated view, where sheet y = 399 - a
        seg = {"a": 99, "b": 299, "length_px": 200, "measured_ft": 20.0, "value_ft": 20.0,
               "printed": "20'0\"", "source": "read", "reader": "trocr",
               "raw_text": "20'0\"", "stated_ft": 20.0}
        return {"chains": [{"axis": "v", "y": x, "x0": 99, "x1": 299, "span_ft": 20.0,
                            "overall_ft": None, "segments": [seg]}]}

    def test_a_chain_beside_the_other_plan_is_refused(self):
        import seeded
        data = self._vertical_chain(380)        # just outside plan A's east wall
        bare = seeded.chain_lookup(data, full_height=400, scale=1.0, wppf=10.0)
        self.assertAlmostEqual(bare([500, 100, 850, 300])["depth"][0], 20.0,
                               msg="without the plans this is the old cross-plan match")
        lab, ext, lines = self._sheet()
        ask = seeded.chain_lookup(data, full_height=400, scale=1.0, wppf=10.0,
                                  lab=lab, ext=ext, lines=lines)
        got = ask([500, 100, 850, 300])
        self.assertIsNone(got["depth"])
        self.assertTrue(any("on another plan" in line for line in got["log"]))

    def test_a_run_behind_an_interior_wall_is_refused(self):
        import seeded
        lab, ext, lines = self._sheet()
        lab[100:300, 50:350] = 0
        lab[100:200, 50:350] = 1                 # room 1 on top, room 2 below
        lab[200:300, 50:350] = 4
        lines[199:202, 50:350] = 255             # the wall between them
        data = {"chains": [], "single_dimensions": [
            {"ft": 30.0, "axis": "h", "lo": 50, "hi": 350, "cross": 150, "source": "read",
             "chain": "h dimension at 150"}]}
        ask = seeded.chain_lookup(data, full_height=400, scale=1.0, wppf=10.0,
                                  lab=lab, ext=ext, lines=lines)
        self.assertAlmostEqual(ask([50, 100, 349, 199], seed=(200, 150))["width"][0], 30.0)
        below = ask([50, 202, 349, 299], seed=(200, 250))
        self.assertIsNone(below["width"], "the run is in the room above, behind a wall")

    def test_a_value_found_at_the_name_is_not_lent(self):
        from seeded import share_across_bands
        rooms = [
            {"label": "MEMORIAL ROOM", "bbox": [100, 100, 458, 280], "area_sqft": 160.0,
             "chain_sides": {"width": None, "depth": 9.0}, "chain_rules": {"depth": "at the name"},
             "needs_review": True},
            {"label": "BATH", "bbox": [480, 100, 730, 280], "area_sqft": 108.0,
             "chain_sides": {"width": 12.5, "depth": None}, "chain_rules": {"width": "chain joints"},
             "needs_review": True},
        ]
        share_across_bands(rooms, ppf=20.0)
        self.assertNotIn("chain_borrowed", rooms[1])
        self.assertIsNone(rooms[1]["chain_sides"]["depth"])

    def test_a_lent_value_says_where_it_came_from(self):
        from seeded import share_across_bands
        rooms = [
            {"label": "BEDROOM B", "bbox": [100, 100, 458, 280], "area_sqft": 160.0,
             "chain_sides": {"width": None, "depth": 9.0}, "chain_rules": {"depth": "chain joints"},
             "chain_side_refs": {"depth": "v chain at 40"}, "needs_review": True},
            {"label": "BATH", "bbox": [480, 100, 730, 280], "area_sqft": 108.0,
             "chain_sides": {"width": 12.5, "depth": None}, "needs_review": True},
        ]
        share_across_bands(rooms, ppf=20.0)
        self.assertEqual(rooms[1]["chain_borrowed_from"]["depth"],
                         "BEDROOM B (chain joints, v chain at 40)")
# ANCHOR AFTER 20261011_142247


# ANCHOR BEFORE 20261011_144059
class TestLoneOutcomesAreKept(unittest.TestCase):
    """The dimension file kept only counts, so nobody could say which candidate was
    unreadable or look at what the reader had been shown."""

    def _candidate(self):
        import cv2
        import dimlines
        img, ppf = _chain_image(ticks=(100, 700))     # 600 px at 20 px/ft = 30 ft
        cv2.putText(img, "30'0\"", (360, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        return img, ppf, dimlines.detect(img, ppf, "h")[1][0]

    def test_every_candidate_is_recorded_with_its_outcome(self):
        import dimlines
        img, ppf, line = self._candidate()
        record = []
        dimlines.read_single(img, line, ppf, _FakeReader("t", ["30'0\""]), "h", 360, record=record)
        dimlines.read_single(img, line, ppf, _FakeReader("t", ["12'0\""]), "h", 360, record=record)
        self.assertEqual([r["outcome"] for r in record], ["believed", "disagreed"])
        self.assertAlmostEqual(record[1]["stated_ft"], 12.0)
        self.assertAlmostEqual(record[1]["drawn_ft"], 30.0, delta=0.5)
        self.assertEqual(record[1]["raw_text"], "12'0\"")

    def test_the_crops_of_a_failure_are_saved(self):
        import dimlines
        img, ppf, line = self._candidate()
        record = []
        with tempfile.TemporaryDirectory() as d:
            dimlines.read_single(img, line, ppf, _FakeReader("t", []), "h", 360,
                                 record=record, crops_dir=d)
            self.assertEqual(record[0]["outcome"], "unreadable")
            self.assertTrue(record[0]["crops"])
            self.assertEqual(sorted(os.listdir(d)), sorted(record[0]["crops"]))

    def test_a_reader_s_slash_two_is_a_half(self):
        from dimensions import parse_length
        # what the handwriting reader returned for 7'0 1/2" on the sample sheet
        self.assertAlmostEqual(parse_length("7'0/2\""), 7 + 0.5 / 12)
        self.assertAlmostEqual(parse_length("7'10 1/2\""), 7.875)
        self.assertAlmostEqual(parse_length("4'2/ \""), 4 + 2 / 12.0)
# ANCHOR AFTER 20261011_144059


# ANCHOR BEFORE 20261011_152314
class TestDashedFeetAndInches(unittest.TestCase):
    """7'-10 1/2" parsed as 7'0": the dash stopped the inches, silently, ten and a half
    inches short. Most CAD sheets write lengths that way."""

    def test_the_dashed_forms(self):
        from dimensions import parse_length as P
        self.assertAlmostEqual(P("7'-10 1/2\""), 7.875)
        self.assertAlmostEqual(P("7'-10\""), 7 + 10 / 12)
        self.assertAlmostEqual(P("7'-0\""), 7.0)
        self.assertAlmostEqual(P("14'-9\""), 14.75)

    def test_the_existing_forms_are_unchanged(self):
        from dimensions import parse_length as P
        for text, ft in (("7'10 1/2\"", 7.875), ("7'10½\"", 7.875), ("4'8\"", 4 + 8 / 12),
                         ("8'0", 8.0), ("4'", 4.0), ("12' 4\"", 12 + 4 / 12),
                         ("7'0/2\"", 7 + 0.5 / 12), ("- 12'3, \"", 12.25)):
            self.assertAlmostEqual(P(text), ft, msg=text)
        self.assertIsNone(P("2'-10\" x 6'-7\" x 1 1/2\""), "several lengths stay ambiguous")
# ANCHOR AFTER 20261011_152314


# ANCHOR BEFORE 20261011_154150
class TestEraseOnlyWhatAReadingProves(unittest.TestCase):
    """Two guards built on shape broke walls. Only a line carrying a believed reading
    may leave the wall mask; a line nothing read stays, whatever it looks like."""

    def test_a_line_with_a_believed_reading_goes_and_one_without_stays(self):
        import numpy as np
        import dimlines
        img = np.full((400, 900), 255, np.uint8)
        for y in (100, 250):                      # two identical lines, 30 ft at 20 px/ft
            img[y:y + 2, 100:700] = 0
            for x in (100, 698):
                img[y - 18:y + 18, x:x + 2] = 0
        data = {"chains": [], "single_dimensions": [
            {"ft": 30.0, "axis": "h", "lo": 100, "hi": 700, "cross": 250, "source": "read",
             "chain": "h dimension at 250"}]}
        geo = dimlines.line_geometry(img, 20.0, data)
        erased = [s["line"][1] for s in geo["stretches"]]
        kept = [s["line"][1] for s in geo["kept"]]
        self.assertEqual(len(erased), 1)
        self.assertAlmostEqual(erased[0], 250, delta=3)
        self.assertTrue(any(abs(y - 100) <= 3 for y in kept), "nothing read it, so it stays")
        self.assertEqual(geo["unmatched"], [])


class TestReadingsOnABrokenUpLine(unittest.TestCase):
    """A beam line crossing 23'7" cut it into pieces, and the number disagreed with the
    piece it sat on. The run it measures is found from the number."""

    def _chain(self, text, pieces=(0, 100, 120, 400)):
        import dimlines
        chain = {"axis": "h", "y": 50, "x0": pieces[0], "x1": pieces[-1], "ticks": list(pieces)}
        raws = [{"a": a, "b": b, "length_px": b - a, "text": "", "reader": None}
                for a, b in zip(pieces, pieces[1:])]
        raws[2]["text"], raws[2]["reader"] = text, "trocr"
        result = dimlines.assemble(chain, raws, 10.0)
        return chain, raws, result

    def test_the_number_is_placed_on_the_run_it_measures(self):
        import dimlines
        chain, raws, result = self._chain("40'0\"")     # pieces of 10, 2 and 28 ft
        self.assertFalse(dimlines.is_real_chain(result))
        got = dimlines.readings_as_singles(chain, raws, result, 10.0, 200)
        self.assertEqual(len(got), 1)
        self.assertEqual((got[0]["lo"], got[0]["hi"], got[0]["pieces"]), (0, 400, 3))

    def test_a_number_that_fits_two_runs_equally_is_left_out(self):
        import dimlines
        # 29 ft sits exactly between the 28 ft piece it is on and the 30 ft run of it and
        # its neighbour, and both are within the slack
        chain, raws, result = self._chain("29'0\"")
        self.assertEqual(dimlines.readings_as_singles(chain, raws, result, 10.0, 200), [])

    def test_a_dropped_chain_keeps_what_was_read(self):
        import dimlines
        chain, raws, result = self._chain("40'0\"")
        rec = dimlines.chain_record(chain, raws, result, 10.0, 200)
        self.assertFalse(rec["kept"])
        self.assertEqual(rec["segments"][2]["raw_text"], "40'0\"")
        self.assertEqual(rec["segments"][2]["outcome"], "disagreed")
        self.assertEqual(rec["segments"][0]["outcome"], "unreadable")
# ANCHOR AFTER 20261011_154150


# ANCHOR BEFORE 20261011_171543
class TestErasureLeavesCrossingWalls(unittest.TestCase):
    """Erasing a horizontal dimension line cut the bath's west wall where the line
    crossed it; the two halves fell under the 2.5 ft wall minimum and the wall left the
    mask. A dimension line taken out must not shorten a wall it crosses."""

    def test_a_crossing_wall_keeps_its_length(self):
        import numpy as np
        import seeded
        img = np.full((300, 800), 255, np.uint8)
        img[100:160, 399:402] = 0                 # a 3 ft wall at 20 px/ft
        img[129:131, 100:700] = 0                 # a dimension line crossing it
        lines = seeded.line_map(img, 20.0, erase=[((100, 130), (700, 130))])
        self.assertTrue(lines[105, 400] > 0 and lines[155, 400] > 0, "the wall must survive")
        self.assertEqual(int(lines[130, 600]), 0, "the dimension line itself is gone")
# ANCHOR AFTER 20261011_171543


# ANCHOR BEFORE 20261011_173131
class TestChainEndArrowhead(unittest.TestCase):
    """A chain whose line ran on to the wall it measures to had no joint there, because
    the wall is too long to be a tick, so its last stretch (18'7" at the bottom of BED
    ROOM C) was never cut into a piece or read. The arrowhead drawn against that wall is
    the joint."""

    PPF = 75.0

    def _sheet(self, mark):
        import cv2
        import numpy as np
        img = np.full((200, 1400), 255, np.uint8)
        img[98:101, 100:1330] = 0                 # the dimension line, running into the wall
        img[:, 1300:1330] = 0                     # the wall it measures to
        if mark == "arrow":
            for s in (-1, 1):
                cv2.line(img, (1300, 99), (1282, 99 + s * 13), 0, 3)
        if mark == "cross":
            cv2.line(img, (1250, 80), (1290, 118), 0, 3)
            cv2.line(img, (1250, 118), (1290, 80), 0, 3)
        line = {"y": 99, "x0": 100, "x1": 1330, "ticks": [400, 700, 900]}
        return img, line

    def test_the_arrowhead_at_the_wall_is_a_joint(self):
        import dimlines
        img, line = self._sheet("arrow")
        dimlines.add_end_joints(img, line, self.PPF)
        self.assertEqual(len(line["end_marks"]), 1)
        self.assertAlmostEqual(line["end_marks"][0], 1300, delta=4)
        self.assertEqual(len(line["ticks"]), 4)

    def test_no_mark_no_joint(self):
        import dimlines
        img, line = self._sheet(None)
        dimlines.add_end_joints(img, line, self.PPF)
        self.assertNotIn("end_marks", line)
        self.assertEqual(line["ticks"], [400, 700, 900])

    def test_two_lines_crossing_are_not_an_arrowhead(self):
        import dimlines
        img, line = self._sheet("cross")
        dimlines.add_end_joints(img, line, self.PPF)
        self.assertNotIn("end_marks", line)
# ANCHOR AFTER 20261011_173131


# ANCHOR BEFORE 20261011_174013
class TestDoubleWallCavityIsOutside(unittest.TestCase):
    """BATH and BED ROOM D are parted by two walls back to back with a cavity between
    them. A region that filled the cavity measured BATH's depth across a wall and the
    cavity; the margin stays one wall thick, so that region is outside its room."""

    BATH = {"box": [6423, 3023, 7371, 3675]}

    def test_a_region_in_the_cavity_is_outside(self):
        import score
        r = {"truth": self.BATH, "box": [6423, 3023, 7371, 3675 + 0.68 * 74.94]}
        self.assertEqual(score.containment(r, 74.94), {"bottom": 0.68})

    def test_a_region_on_its_own_wall_is_inside(self):
        import score
        r = {"truth": self.BATH, "box": [6423, 3023, 7371, 3675 + 0.45 * 74.94]}
        self.assertEqual(score.containment(r, 74.94), {})
# ANCHOR AFTER 20261011_174013


# ANCHOR BEFORE 20261011_175231
class TestBandSharingStaysOnOnePlan(unittest.TestCase):
    """The sample sheet carries the first and second floors side by side. BED ROOM C on
    the second floor spans the same y-band as WEST PARLOR on the first, and took the
    parlour's 19'4" as its depth: a false agreement. A value is lent only within a plan."""

    def _rooms(self):
        # 20 px/ft. WEST PARLOR on plan 0 knows its depth; BED ROOM C on plan 1 knows
        # its width and spans the same y-band 100..480.
        return [
            {"label": "WEST PARLOR", "plan": 0, "bbox": [100, 100, 400, 480], "area_sqft": 280.0,
             "chain_sides": {"width": None, "depth": 19.333}, "chain_rules": {"depth": "in the room"},
             "needs_review": True},
            {"label": "BED ROOM C", "plan": 1, "bbox": [900, 100, 1211, 480], "area_sqft": 290.0,
             "chain_sides": {"width": 15.583, "depth": None}, "chain_rules": {"width": "in the room"},
             "needs_review": True},
        ]

    def test_a_room_on_another_plan_does_not_lend(self):
        from seeded import share_across_bands
        rooms = self._rooms()
        share_across_bands(rooms, ppf=20.0)
        self.assertNotIn("chain_borrowed", rooms[1])
        self.assertNotIn("stated_dims_ft", rooms[1])
        self.assertTrue(rooms[1]["needs_review"])

    def test_a_neighbour_on_the_same_plan_still_lends(self):
        from seeded import share_across_bands
        rooms = self._rooms()
        # BED ROOM D, beside C on plan 1 between the same top and bottom walls
        rooms.append({"label": "BED ROOM D", "plan": 1, "bbox": [1230, 100, 1500, 480],
                      "area_sqft": 250.0, "chain_sides": {"width": None, "depth": 19.0},
                      "chain_rules": {"depth": "in the room"}, "needs_review": True})
        share_across_bands(rooms, ppf=20.0)
        self.assertEqual(rooms[1]["chain_borrowed"], {"depth": "BED ROOM D"})
        self.assertEqual(rooms[1]["stated_dims_ft"], [15.583, 19.0])

    def test_a_room_not_clearly_on_one_plan_neither_lends_nor_borrows(self):
        from seeded import share_across_bands
        rooms = self._rooms()
        rooms[0]["plan"] = None
        rooms[1]["plan"] = None
        share_across_bands(rooms, ppf=20.0)
        self.assertNotIn("chain_borrowed", rooms[1])
# ANCHOR AFTER 20261011_175231


# ANCHOR BEFORE 20261011_180856
class TestChainLineLeavesTheCrop(unittest.TestCase):
    """The dimension line ran through the numerals into the reader, which took it for part
    of a glyph: 13'6" came back as 43'6". The line's own rows are cleared from a piece's
    crop; a stroke that crosses the line stays."""

    def test_line_gone_crossing_stroke_kept(self):
        import numpy as np
        import dimlines
        ppf = 75.0
        chain = {"y": 300, "axis": "h"}
        gray = np.full((600, 900), 255, np.uint8)
        gray[299:302, 100:800] = 0                # the chain line
        gray[270:330, 400:404] = 0                # a numeral's stroke crossing it
        crop = dimlines.crop_segment(gray, chain, 100, 800, ppf)
        out = dimlines.remove_chain_line(crop, chain, ppf)
        row = 300 - (300 - max(10, int(0.8 * ppf)))
        x_line = 600 - (100 + max(2, int(700 * 0.06)))
        x_stroke = 401 - (100 + max(2, int(700 * 0.06)))
        self.assertEqual(int(crop[row, x_line]), 0, "the line is in the raw crop")
        self.assertEqual(int(out[row, x_line]), 255, "and gone from what is read")
        self.assertEqual(int(out[row, x_stroke]), 0, "the crossing stroke survives")
# ANCHOR AFTER 20261011_180856


# ANCHOR BEFORE 20261011_183319
class TestLeakedRegionCannotAgree(unittest.TestCase):
    """WEST PARLOR read 15'3" x 19'4" correctly and agreed, while its region ran 1.20 ft
    past its east wall through an opening. The pipeline has no truth box, so it judges by
    the walls the room's own label sees: a region past them cannot agree."""

    PPF = 20.0

    def _sheet(self):
        import numpy as np
        lines = np.zeros((500, 700), np.uint8)
        lines[100, 100:601] = 255                  # top
        lines[400, 100:601] = 255                  # bottom
        lines[100:401, 100] = 255                  # west wall of room A
        lines[100:401, 600] = 255                  # east wall of room B
        lines[100:401, 300] = 255                  # the partition between them...
        lines[120:188, 300] = 0                    # ...with a 3.4 ft opening, away from A's label
        return lines

    def _room(self, bbox):
        return {"id": 1, "label": "A", "area_px": 1000, "bbox": bbox, "needs_review": False}

    def test_a_region_through_the_opening_is_refused(self):
        import seeded
        rooms = [self._room([101, 101, 599, 399])]          # spans A and B
        seeded.refuse_leaks(rooms, [("A", 200, 250)], self._sheet(), self.PPF)
        self.assertTrue(rooms[0]["needs_review"])
        self.assertIn("right", rooms[0]["region_leak_ft"])
        self.assertIn("past this room's own walls", rooms[0]["review_reason"])

    def test_a_region_filling_its_room_stands(self):
        import seeded
        rooms = [self._room([101, 101, 299, 399])]          # room A exactly
        seeded.refuse_leaks(rooms, [("A", 200, 250)], self._sheet(), self.PPF)
        self.assertFalse(rooms[0]["needs_review"])
        self.assertNotIn("region_leak_ft", rooms[0])
# ANCHOR AFTER 20261011_183319


# ANCHOR BEFORE 20261011_184007
class TestReadingPlacedByItsArrowheads(unittest.TestCase):
    """18'7" at the bottom of BED ROOM C is drawn 0.46 ft short, so the run across the
    partition above it is the closer fit. Its arrowheads bound the room, and they place it."""

    def _sheet(self):
        import cv2
        import numpy as np
        img = np.full((200, 1800), 255, np.uint8)
        img[98:101, 150:1640] = 0                       # the dimension line
        for x, pointing in ((252, -1), (1610, 1)):      # arrowheads at the run's two ends
            for s in (-1, 1):
                cv2.line(img, (x, 99), (x - pointing * 18, 99 + s * 13), 0, 3)
        chain = {"y": 99, "axis": "h", "ticks": [200, 252, 1610]}
        raws = [{"a": 200, "b": 252, "text": ""},
                {"a": 252, "b": 1610, "text": "18'7\"", "reader": "t"}]
        return img, chain, raws

    def test_the_marked_run_wins_over_the_closer_fit(self):
        import dimlines
        img, chain, raws = self._sheet()
        got = dimlines.readings_as_singles(chain, raws, {"readings": 1}, 75.0, 200, gray=img)
        self.assertEqual([(g["lo"], g["hi"], g["placed_by"]) for g in got], [(252, 1610, "end marks")])

    def test_without_the_image_closest_fit_still_decides(self):
        import dimlines
        img, chain, raws = self._sheet()
        got = dimlines.readings_as_singles(chain, raws, {"readings": 1}, 75.0, 200)
        self.assertEqual([(g["lo"], g["hi"], g["placed_by"]) for g in got], [(200, 1610, "closest fit")])
# ANCHOR AFTER 20261011_184007


# ANCHOR BEFORE 20261011_185022
class TestBlankPieceIsNotRead(unittest.TestCase):
    """With the line taken out, a 0.84 ft stub held only paper and a few specks, and the
    reader answered 1'. It was believed, and turned the line into a kept chain. A piece
    with nothing text-shaped on it is not shown to the reader."""

    def test_reader_never_sees_blank_paper(self):
        import numpy as np
        import dimlines
        calls = []

        class Says1Foot:
            kind = "fake"

            def __call__(self, crop):
                calls.append(crop.shape)
                return "1'"

        gray = np.full((300, 900), 255, np.uint8)
        gray[149:152, 50:850] = 0                       # the chain line, nothing written on it
        chain = {"y": 150, "axis": "h", "ticks": [100, 163, 400, 800]}
        raws = dimlines.read_chain_raw(gray, chain, 75.0, Says1Foot())
        self.assertEqual(calls, [])
        self.assertEqual([r["text"] for r in raws], ["", "", ""])
# ANCHOR AFTER 20261011_185022


# ANCHOR BEFORE 20261011_190101
class TestSteppedCasingIsAJamb(unittest.TestCase):
    """EAST PARLOR's doors are 2'-1" and 2'-4", but their moulded jambs are drawn as steps,
    so the straight wall stopped short of each door edge and the gap measured over the
    3 ft doorway rule. A wall carries on through the casing attached to its end."""

    def _wall(self, casing):
        import numpy as np
        img = np.full((300, 300), 255, np.uint8)
        img[0:100, 199:202] = 0                    # wall above the door, 5 ft at 20 px/ft
        img[168:300, 199:202] = 0                  # wall below it; 3.4 ft of straight gap
        if casing:                                 # a stepped casing hanging off the upper wall
            img[100:108, 202:205] = 0
            img[106:109, 199:205] = 0
            img[108:116, 204:207] = 0
        return img

    def test_the_door_with_a_stepped_casing_closes(self):
        import seeded
        self.assertTrue(seeded.line_map(self._wall(True), 20.0)[140, 200] > 0)

    def test_a_straight_gap_over_a_door_stays_open(self):
        import seeded
        self.assertEqual(int(seeded.line_map(self._wall(False), 20.0)[140, 200]), 0)
# ANCHOR AFTER 20261011_190101


# ANCHOR BEFORE 20261011_214906
class TestRetryOnlyWhatReadNothing(unittest.TestCase):
    """Clearing the line's ragged edge recovered 15'7" and lost 17'11" and 12'6", the widths
    of both confirmations. Used only to retry a piece that read nothing, it cannot lose a
    reading: a piece that read anything is never read again."""

    def test_a_read_piece_is_never_retried(self):
        import dimlines
        img, ppf = _chain_image(ticks=(100, 400, 700), marks=True)
        chain = [c for c in dimlines.detect(img, ppf, "h")[0] if len(c["ticks"]) == 3][0]
        raws = dimlines.read_chain_raw(img, chain, ppf, _FakeReader("tesseract", ["15'0\""]))
        self.assertEqual(raws[0]["text"], "15'0\"")
        self.assertFalse([v for v in raws[0]["views"] if v["view"].startswith("retry")])
        self.assertTrue([v for v in raws[1]["views"] if v["view"].startswith("retry")])

    def test_the_retry_clears_the_ragged_edge(self):
        import numpy as np
        import dimlines
        ppf = 75.0
        chain = {"y": 300, "axis": "h"}
        gray = np.full((600, 900), 255, np.uint8)
        gray[299:302, 100:800] = 0                    # the line
        gray[298, 100:800:3] = 0                      # its ragged edge, a third inked
        crop = dimlines.crop_segment(gray, chain, 100, 800, ppf)
        row = 300 - (300 - max(10, int(0.8 * ppf))) - 2
        self.assertTrue((dimlines.remove_chain_line(crop, chain, ppf)[row] < 160).any())
        self.assertFalse((dimlines.remove_chain_line(crop, chain, ppf, edge=True)[row] < 160).any())
# ANCHOR AFTER 20261011_214906


# ANCHOR BEFORE 20261011_215006
class TestCornerDoorCasing(unittest.TestCase):
    """Door 4 on the sample sheet sits at a corner: its casing rises off the end of the
    south wall, which runs the other way, so there was no run on the door's axis to carry
    on and EAST PARLOR flowed into the entry hall. The casing at a perpendicular wall's end
    is the jamb."""

    def _corner(self, casing):
        import numpy as np
        img = np.full((360, 400), 255, np.uint8)
        img[0:230, 199:202] = 0                    # partition above the door, 11.5 ft at 20 px/ft
        img[299:302, 0:202] = 0                    # south wall, ending at the partition's line
        if casing:                                 # stepped casing rising off the south wall's end
            img[290:300, 199:202] = 0
            img[282:291, 197:200] = 0
        return img

    @staticmethod
    def _sealed(lines):
        # the door is shut when some column along the partition's line is wall all the way
        return bool((lines[230:282, 196:205] > 0).all(axis=0).any())

    def test_the_corner_door_closes(self):
        import seeded
        self.assertTrue(self._sealed(seeded.line_map(self._corner(True), 20.0)))

    def test_without_a_casing_the_corner_stays_open(self):
        import seeded
        self.assertFalse(self._sealed(seeded.line_map(self._corner(False), 20.0)))
# ANCHOR AFTER 20261011_215006


# ANCHOR BEFORE 20261011_220947
class TestChainConfirmedRoomInTheTrace(unittest.TestCase):
    """A room confirmed by its chains is never checked against loose printed lengths, so it
    has no list of them. The trace wrote one anyway, the previous room's, and crashed when
    the first room on the sheet (BED ROOM C, once 15'7" was read) was confirmed that way."""

    def test_first_room_confirmed_by_its_chains(self):
        import numpy as np
        import seeded
        from runtrace import Trace
        lab = np.zeros((40, 40), np.int32)
        lab[10:30, 10:30] = 1

        def ask(bbox, seed=None):
            return {"width": (20.0, "h chain at 380", 0.0), "depth": (20.0, "v chain at 20", 0.0)}

        tr = Trace("p.png", group="20260101-000000")
        rooms = seeded.build_rooms(lab, [("KITCHEN", 20, 20)], 1.0, [], [], trace=tr, chains=ask)
        self.assertFalse(rooms[0]["needs_review"])
        self.assertNotIn("aligned_printed_lengths", tr.data["rooms"][0])
# ANCHOR AFTER 20261011_220947


# ANCHOR BEFORE 20261012_033440
class TestDuplicateChainsShareTheirJoints(unittest.TestCase):
    """The second-floor room widths (15'7" and 13'6") were dropped as a copy of the window
    pier chain 3.7 ft below them: the same length, a nearby row. Two first-floor chains
    were dropped as copies of lines on the other plan. A copy has every joint on the
    other line's joints; two chains with joints of their own are two chains."""

    @staticmethod
    def _line(y, ticks):
        return {"y": y, "x0": ticks[0], "x1": ticks[-1], "ticks": list(ticks)}

    def test_two_chains_a_few_pixels_apart_both_survive(self):
        import dimlines
        rooms = self._line(500, (100, 300, 500, 700, 900))
        piers = self._line(504, (100, 200, 400, 600, 800, 900))
        self.assertEqual(len(dimlines.drop_duplicate_chains([rooms, piers], 20.0)), 2)

    def test_the_same_line_found_twice_is_kept_once(self):
        import dimlines
        a = self._line(500, (100, 300, 500, 700, 900))
        b = self._line(503, (101, 299, 502, 700, 899))
        self.assertEqual(len(dimlines.drop_duplicate_chains([a, b], 20.0)), 1)

    def test_an_overall_is_still_a_copy_of_its_chain(self):
        import dimlines
        chain = self._line(500, (100, 300, 500, 700, 900))
        overall = self._line(530, (100, 500, 900))
        kept = dimlines.drop_duplicate_chains([overall, chain], 20.0)
        self.assertEqual([len(c["ticks"]) for c in kept], [5])

    def test_equal_lines_on_two_plans_both_survive(self):
        import dimlines
        left = self._line(500, (100, 300, 500, 700, 900))
        right = self._line(520, (2100, 2400, 2600, 2900))
        self.assertEqual(len(dimlines.drop_duplicate_chains([left, right], 20.0)), 2)


class TestCapKeepsInteriorChains(unittest.TestCase):
    """With 16 chains read per axis, the long, regular perimeter chains took the slots and
    the interior ones, which measure rooms, were left over."""

    def _chains(self):
        height = 1000
        perim_top = {"axis": "h", "y": 50, "x0": 100, "x1": 900, "ticks": [100, 300, 600, 900]}
        perim_bottom = {"axis": "h", "y": 950, "x0": 100, "x1": 900, "ticks": [100, 400, 900]}
        interior = {"axis": "h", "y": 500, "x0": 300, "x1": 700, "ticks": [300, 500, 700]}
        # a vertical chain at x=400 with joints at sheet y 200, 450 and 800, stored in the
        # rotated view the vertical chains live in
        across = {"axis": "v", "y": 400, "x0": 199, "x1": 799,
                  "ticks": sorted(height - 1 - y for y in (200, 450, 800))}
        return height, perim_top, perim_bottom, interior, across

    def test_interior_and_perimeter_are_told_apart(self):
        import dimlines
        height, top, bottom, interior, across = self._chains()
        dimlines.mark_inside([top, bottom, interior, across], height, 20.0)
        self.assertTrue(interior["inside"])
        self.assertFalse(top["inside"])
        self.assertFalse(bottom["inside"])

    def test_a_cap_of_two_keeps_the_interior_chain(self):
        import dimlines
        height, top, bottom, interior, across = self._chains()
        chains = [top, bottom, interior, across]
        dimlines.mark_inside(chains, height, 20.0)
        chains.sort(key=lambda c: -dimlines.span_px(c))
        picked = [c for c in dimlines.by_axis(chains, 2) if c["axis"] == "h"]
        self.assertIn(interior, picked)
        self.assertEqual(len(picked), 2)


class TestLoneOutcomesAddUp(unittest.TestCase):
    """The log said 12 lone dimensions believed of 20 lines tried, with outcomes that
    accounted for all 20 as failures: the 12 were placed on chain lines, not tried."""

    def test_counts_on_chain_lines_are_kept_apart(self):
        import dimlines
        why = {"disagreed": 1, "nothing written on it": 4, "unreadable": 15}
        text = dimlines.lone_summary(12, 20, why)
        self.assertTrue(text.startswith("0 lone dimension(s) believed, of 20 line(s) tried"))
        self.assertIn("12 more believed on chain lines", text)

    def test_believed_lines_are_counted_once(self):
        import dimlines
        text = dimlines.lone_summary(3, 5, {"believed": 3, "unreadable": 2})
        self.assertTrue(text.startswith("3 lone dimension(s) believed, of 5 line(s) tried"))
        self.assertNotIn("more believed", text)


class TestFullRunErasesProvenLines(unittest.TestCase):
    """Every result was measured with the proven dimension lines taken out of the walls,
    and the pipeline never wrote the file that does it: a real run left them in."""

    def test_the_dimension_pass_is_followed_by_the_geometry_pass(self):
        script = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "run_layers.sh")).read()
        reading = script.index('python dimlines.py "${DIMS_ARGS[@]}"')
        geometry = script.index("python dimlines.py --lines-only")
        rooms = script.index("python seeded.py")
        self.assertTrue(reading < geometry < rooms)
# ANCHOR AFTER 20261012_033440
