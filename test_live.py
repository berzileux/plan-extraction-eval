"""
Opt-in tests that run the REAL models. Skipped unless RUN_LIVE=1.

  RUN_LIVE=1 python test_live.py                     # Florence-2 and Ollama
  RUN_LIVE=1 LIVE_SKIP_FLORENCE=1 python test_live.py  # Ollama only
  LIVE_MODEL=qwen2.5:3b  LIVE_OLLAMA_URL=http://127.0.0.1:11434   (defaults)

Model output varies, so these check broad properties, not exact text:
  - Florence-2 reads most of the labels and sizes on the synthetic plan, and the
    deliberately wrong room is flagged.
  - The language model step never accepts a size that is not in the room's text.
Passing here is evidence, not proof. Failing may mean a weak model, not a bug.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.environ.get("RUN_LIVE") == "1"
MODEL = os.environ.get("LIVE_MODEL", "qwen2.5:3b")
URL = os.environ.get("LIVE_OLLAMA_URL", "http://127.0.0.1:11434")
EXPECTED = ["LIVING", "KITCHEN", "BEDROOM 1", "BEDROOM 2"]


def run(args, cwd):
    return subprocess.run([sys.executable] + args, cwd=cwd, check=True,
                          capture_output=True, text=True)


def build_plan(d):
    os.makedirs(os.path.join(d, "plans"), exist_ok=True)
    run([os.path.join(HERE, "make_synthetic_text.py")], d)
    man = os.path.join(d, "manifest.csv")
    with open(man, "w") as f:
        f.write("plan,pixels_per_foot\nsynthetic_text.png,20.0\n")
    run([os.path.join(HERE, "extract.py"), "--plans", os.path.join(d, "plans"),
         "--manifest", man, "--out", os.path.join(d, "viewer")], d)
    return os.path.join(d, "viewer", "results.json")


def load(path):
    with open(path) as f:
        return json.load(f)["plans"][0]["rooms"]


@unittest.skipUnless(LIVE, "set RUN_LIVE=1 to run live model tests")
@unittest.skipIf(os.environ.get("LIVE_SKIP_FLORENCE") == "1", "LIVE_SKIP_FLORENCE=1")
class TestFlorenceLive(unittest.TestCase):
    def test_reads_labels_and_flags_the_bad_room(self):
        with tempfile.TemporaryDirectory() as d:
            results = build_plan(d)
            run([os.path.join(HERE, "read_labels.py"), "--plans", os.path.join(d, "plans"),
                 "--results", results, "--trust-remote-code"], d)
            rooms = load(results)
            self.assertEqual(len(rooms), 4)
            labels = [r.get("label", "").upper() for r in rooms]
            found = [e for e in EXPECTED if any(e in l for l in labels)]
            print("\nFlorence read labels:", labels)
            self.assertGreaterEqual(len(found), 3, "model read fewer than 3 of 4 labels")
            with_dims = [r for r in rooms if r.get("stated_dims_ft")]
            self.assertGreaterEqual(len(with_dims), 3, "model read fewer than 3 of 4 sizes")
            bad = [r for r in rooms if "BEDROOM 2" in r.get("label", "").upper()]
            if bad:
                self.assertTrue(bad[0]["needs_review"], "the wrong 12' x 12' room was not flagged")
            ok = [r for r in rooms if not r["needs_review"]]
            self.assertGreaterEqual(len(ok), 2, "fewer than 2 correct rooms passed")


def ollama_ready():
    try:
        with urllib.request.urlopen(URL.rstrip("/") + "/api/tags", timeout=5) as r:
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception:
        return False
    return MODEL in names or (":" not in MODEL and (MODEL + ":latest") in names)


@unittest.skipUnless(LIVE, "set RUN_LIVE=1 to run live model tests")
class TestOllamaLive(unittest.TestCase):
    def test_language_model_never_adds_ungrounded_sizes(self):
        if not ollama_ready():
            self.skipTest("Ollama not reachable at %s or model %s not pulled" % (URL, MODEL))
        with tempfile.TemporaryDirectory() as d:
            results = build_plan(d)
            fake = os.path.join(d, "fake.json")
            with open(fake, "w") as f:
                json.dump({"synthetic_text.png": [
                    {"text": "LIVING", "cx": 320, "cy": 183},
                    {"text": "29'-5\" x 19'-5\"", "cx": 320, "cy": 233},
                    {"text": "KITCHEN", "cx": 920, "cy": 183},
                    {"text": "BEDROOM 2", "cx": 320, "cy": 583},
                    {"text": "12 by 12", "cx": 320, "cy": 633}]}, f)
            run([os.path.join(HERE, "read_labels.py"), "--plans", os.path.join(d, "plans"),
                 "--results", results, "--fake-ocr", fake], d)
            run([os.path.join(HERE, "interpret.py"), "--results", results,
                 "--model", MODEL, "--url", URL], d)
            rooms = load(results)
            for r in rooms:
                print("room", r["id"], repr(r.get("label")), r.get("stated_dims_ft"),
                      r.get("dims_source"), r.get("model_error") or r.get("model_note") or "")
            self.assertFalse([r for r in rooms if r.get("model_error")], "model call failed")
            kitchen = [r for r in rooms if "KITCHEN" in (r.get("label") or "").upper()][0]
            self.assertNotIn("stated_dims_ft", kitchen,
                             "a size appeared for a room whose text has no size")
            for r in rooms:
                if r.get("dims_source") == "model":
                    raw = " ".join(r["raw_text"])
                    for v in r["stated_dims_ft"]:
                        self.assertIn(str(int(v)), raw, "model size not found in the room text")


if __name__ == "__main__":
    unittest.main()
