"""Load the tuning rules, so the engine stays fixed and the rules move.

Every threshold the pipeline can be adjusted by lives in rules.json, and the room
vocabulary in lexicon.json, each with a description of what it is for. Modules ask
for a value by path:

    from ruleset import rule
    slack = rule("reading.disagree_ft")

A missing file or a missing key falls back to the default recorded here, so the
pipeline runs from a bare checkout and a half-written rules file cannot silently
break it. A value of the wrong type, or outside its sane range, is refused loudly:
a tolerance typed as 8 instead of 0.08 would quietly accept every room on the sheet,
and that is worse than stopping.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULES_FILE = HERE / "rules.json"
LEXICON_FILE = HERE / "lexicon.json"

# path -> (default, low, high). The range is what the value can mean, not a
# preference: outside it the number is a typo, not a tuning choice.
DEFAULTS = {
    "detection.line_min_ft": (8.0, 1.0, 100.0),
    "detection.line_max_thick_ft": (0.25, 0.02, 3.0),
    "detection.tick_min_ft": (0.9, 0.1, 10.0),
    "detection.tick_max_ft": (12.0, 1.0, 200.0),
    "detection.bridge_ft": (0.4, 0.0, 5.0),
    "detection.min_chain_ticks": (3, 2, 20),
    "detection.single_min_ft": (4.0, 0.5, 100.0),
    "detection.single_max_ft": (60.0, 2.0, 500.0),
    "detection.single_min_ticks": (2, 0, 10),
    "detection.single_max_text_ft": (8.0, 0.5, 100.0),
    "detection.overall_near_ft": (4.0, 0.5, 40.0),
    "detection.overall_span_tol": (0.02, 0.001, 0.5),
    "detection.end_mark_window_ft": (1.0, 0.1, 10.0),
    "detection.arrow_stroke_min_ft": (0.08, 0.0, 2.0),
    "detection.arrow_stroke_max_ft": (0.2, 0.02, 5.0),
    "detection.arrow_stroke_width_ft": (0.07, 0.01, 2.0),
    "detection.arrow_angle_min_deg": (15, 1, 89),
    "detection.arrow_angle_max_deg": (60, 1, 89),
    "detection.arrow_stroke_ink": (0.8, 0.1, 1.0),
    "detection.arrow_flank_ink": (0.25, 0.0, 1.0),
    "reading.disagree_ft": (0.45, 0.0, 10.0),
    "reading.disagree_rel": (0.04, 0.0, 1.0),
    "reading.same_gap_rel": (0.04, 0.0, 1.0),
    "reading.min_readings": (2, 1, 20),
    "reading.scale_min_segment_ft": (3.0, 0.5, 100.0),
    "reading.scale_max_drift": (0.25, 0.01, 2.0),
    "reading.scale_min_votes": (4, 1, 100),
    "reading.merge_margin_rel": (0.5, 0.05, 1.0),
    "scoring.containment_margin_ft": (0.6, 0.0, 10.0),
    "segmentation.door_max_ft": (3.0, 0.0, 10.0),
    "segmentation.jamb_min_ft": (0.75, 0.1, 5.0),
    "segmentation.casing_depth_ft": (0.3, 0.0, 2.0),
    "segmentation.casing_reach_ft": (1.0, 0.0, 5.0),
    "matching.joint_tol_ft": (1.5, 0.1, 20.0),
    "matching.cover_rel": (0.7, 0.1, 1.0),
    "matching.band_tol_ft": (1.0, 0.05, 20.0),
    "matching.name_near_ft": (14.0, 1.0, 200.0),
    "matching.side_tol_rel": (0.08, 0.001, 1.0),
    "matching.text_cross_ft": (4.0, 0.0, 60.0),
    "matching.min_room_sqft": (25.0, 0.0, 10000.0),
    "matching.plan_gap_ft": (6.0, 1.0, 200.0),
    "matching.plan_margin_rel": (0.5, 0.05, 1.0),
    "matching.wall_faces_ft": (0.6, 0.05, 5.0),
    "matching.partition_min_ft": (3.0, 0.5, 50.0),
    "matching.box_offset_ft": (4.0, 0.0, 30.0),
    "matching.lend_max_rule": (2, 1, 3),
    "matching.leak_margin_ft": (0.6, 0.0, 10.0),
    "matching.leak_fan_ft": (2.0, 0.0, 20.0),
    "matching.leak_fan_step_ft": (0.25, 0.01, 5.0),
    "lexicon.room_cutoff": (0.82, 0.3, 1.0),
    "lexicon.modifier_cutoff": (0.85, 0.3, 1.0),
}

_cache = {}


def _load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _rules():
    if "rules" not in _cache:
        _cache["rules"] = _load(RULES_FILE)
        _cache["lexicon"] = _load(LEXICON_FILE)
    return _cache["rules"], _cache["lexicon"]


def reload():
    """Forget what was loaded. For tests, and for a long-running process."""
    _cache.clear()


def rule(path):
    """One tuning value, by dotted path. Falls back to the default; refuses nonsense."""
    if path not in DEFAULTS:
        raise KeyError("no such rule: %s" % path)
    default, low, high = DEFAULTS[path]
    rules, lex = _rules()
    section, _, key = path.partition(".")
    src = lex.get("thresholds", {}) if section == "lexicon" else rules.get(section, {})
    entry = src.get(key)
    if entry is None:
        return default
    value = entry.get("value") if isinstance(entry, dict) else entry
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s in %s must be a number, got %r"
                         % (path, RULES_FILE.name, value))
    if not low <= value <= high:
        raise ValueError("%s is %g, outside the sensible range %g to %g. A tolerance "
                         "written as 8 instead of 0.08 would accept everything."
                         % (path, value, low, high))
    return type(default)(value)


def describe(path):
    """What a rule is for, as written in the file."""
    rules, lex = _rules()
    section, _, key = path.partition(".")
    src = lex.get("thresholds", {}) if section == "lexicon" else rules.get(section, {})
    entry = src.get(key)
    return (entry or {}).get("description", "") if isinstance(entry, dict) else ""


def lexicon(extra=()):
    """Room words. The period and drawing-specific ones are included by default but
    are listed apart in the file, so it stays visible what the vocabulary has been
    fitted to."""
    _, lex = _rules()
    words = list(lex.get("rooms") or [])
    words += [w for w in (lex.get("period_or_specific") or {}) if not w.startswith("_")]
    words += [w.strip().upper() for w in extra if w and w.strip()]
    return sorted(set(words))


def modifiers():
    _, lex = _rules()
    return list(lex.get("modifiers") or [])


def all_rules():
    """Every rule with its value and description. For `python ruleset.py`."""
    return [(p, rule(p), describe(p)) for p in sorted(DEFAULTS)]


if __name__ == "__main__":
    print("rules from %s and %s\n" % (RULES_FILE.name, LEXICON_FILE.name))
    section = None
    for path, value, why in all_rules():
        head = path.split(".")[0]
        if head != section:
            section = head
            print("[%s]" % section)
        print("  %-34s %-8s %s" % (path.split(".", 1)[1], value, why[:88]))
    print("\n%d room words, %d modifiers" % (len(lexicon()), len(modifiers())))
