"""Layer 4a: turn a detected dimension chain into trusted lengths.

The reader is allowed to be bad. It is not allowed to be confidently wrong, and
everything here exists to enforce that split:

  - a stated value that disagrees with its own measured length is thrown away
  - whatever is left must sum to the chain's overall
  - unread gaps whose measured lengths agree with each other are the same
    dimension repeated, so the residual divides between the groups, not the gaps

On a 1930s hand-lettered sheet this recovered all eleven segments of a chain
from six readings. Nothing here is specific to pixels: a chain detected in an
image and a chain read out of a PDF's vector layer both arrive as the same
list, so both go through the same checks.

No image, no model, no network. Pure arithmetic, so it is cheap to test.
"""
import re

from ruleset import rule

# A stated length further than this from its own measured length is not believed.
# Two numbers, because a drawn chain is not uniformly accurate: a draughtsman
# places a long run carefully and crams the short ones, so on the Putnam sheet a
# 2'8" segment measures 19% short while a 12'6" run is within 0.2%. A flat
# percentage either throws away every short reading or lets a wrong long one
# through. Allow a fixed slack of about five inches, or 4% on longer runs,
# whichever is larger.
DISAGREE_FT = rule("reading.disagree_ft")
DISAGREE_REL = rule("reading.disagree_rel")
# unread gaps whose measured lengths agree this closely are the same dimension
SAME_GAP = rule("reading.same_gap_rel")
# how far a solved value may sit from its measured length before we flag it
SOLVE_WARN = 0.25

# A handwriting reader returns a superscript half as "/2" straight after the inches
# (7'0 1/2" comes back as 7'0/2"), so a bare "/2" there is a half too.
FEET_INCHES = re.compile(
    r"""(?P<ft>\d{1,3})\s*[''`]\s*          # 12'
        -?\s*                               # 12'-6", the usual way on CAD sheets
        (?:(?P<in>\d{1,2})\s*)?             # 6
        (?:(?P<num>\d)\s*/\s*(?P<den>\d)|(?P<half>1/2|½|/2))?  # 1/2
        \s*["”'']?""",
    re.X,
)


def parse_length(text):
    """Read one architectural length. Returns feet as a float, or None.

    Handles 12'6", 7'10 1/2", 7'10½", 8'0, 4'. Readers add stray quotes and
    periods, so the pattern is found inside the string rather than matched
    against the whole of it. A string with more than one length in it is
    ambiguous and gets rejected: that is a whole chain read as one blob, and
    its position means nothing.
    """
    if not text:
        return None
    hits = list(FEET_INCHES.finditer(text))
    if len(hits) != 1:
        return None
    m = hits[0]
    ft = int(m.group("ft"))
    inches = int(m.group("in") or 0)
    if m.group("half"):
        inches += 0.5
    elif m.group("num"):
        den = int(m.group("den"))
        if den:
            inches += int(m.group("num")) / den
    if inches >= 12:
        return None
    total = ft + inches / 12.0
    return total if 0.1 <= total <= 500 else None


def slack_ft(measured_ft):
    """How far a reading may sit from its drawn length before we stop believing it."""
    return max(DISAGREE_FT, DISAGREE_REL * measured_ft)


def believe(stated_ft, measured_ft):
    """Keep a reading only if it agrees with the length actually drawn.

    This is the guard against the dangerous failure. A reader that returns
    nothing costs one gap; a reader that returns 12'0" for a 12'6" dimension
    corrupts the whole chain, and only the measured length catches it. That
    particular misread is 0.52 ft off a 12.52 ft run, which clears the slack
    allowed at that length, so it is rejected and becomes a gap.
    """
    if stated_ft is None or measured_ft <= 0:
        return False
    return abs(stated_ft - measured_ft) <= slack_ft(measured_ft)


def group_gaps(gaps, measured, tol=SAME_GAP):
    """Cluster unread gaps whose measured lengths agree: the same dimension repeated.

    Drawn chains repeat wall thicknesses and window piers. Allocating the
    residual gap by gap lets rounding scatter them; allocating it group by
    group and splitting evenly inside the group does not.
    """
    out = []
    for i in sorted(gaps, key=lambda i: measured[i]):
        if out and measured[i] / measured[out[-1][0]] - 1 <= tol:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def solve_chain(measured, stated, overall_ft, round_to=12.0):
    """Fill the gaps in one chain.

    measured   length of each segment in feet, from the drawing
    stated     what the reader returned per segment, feet or None
    overall_ft the chain's own total, if it is known
    round_to   snap solved values to this many parts of a foot (12 = inches)

    Returns (values, info). values holds a float per segment, or None where
    nothing could be decided. info['source'] says per segment where its number
    came from: 'read', 'solved', or None.
    """
    n = len(measured)
    values = [None] * n
    source = [None] * n
    rejected = []
    for i, (m, s) in enumerate(zip(measured, stated)):
        if s is None:
            continue
        if believe(s, m):
            values[i] = s
            source[i] = "read"
        else:
            rejected.append({"segment": i, "stated_ft": s, "measured_ft": round(m, 2),
                             "off_pct": round(abs(s - m) / m * 100, 1)})

    info = {"source": source, "rejected": rejected, "overall_ft": overall_ft,
            "warnings": [], "residual_ft": None}
    gaps = [i for i in range(n) if values[i] is None]
    if not gaps:
        return values, info
    if overall_ft is None:
        info["warnings"].append(
            "chain total is unknown, so %d unread segment(s) stay unknown" % len(gaps))
        return values, info

    residual = overall_ft - sum(v for v in values if v is not None)
    info["residual_ft"] = round(residual, 3)
    pool = sum(measured[i] for i in gaps)
    if residual <= 0 or pool <= 0:
        info["warnings"].append(
            "the segments already read exceed the chain total, so the gaps cannot be solved")
        return values, info

    for grp in group_gaps(gaps, measured, SAME_GAP):
        share = residual * sum(measured[i] for i in grp) / pool
        each = round(share / len(grp) * round_to) / round_to
        for i in grp:
            values[i] = each
            source[i] = "solved"
            if measured[i] > 0 and abs(each - measured[i]) / measured[i] > SOLVE_WARN:
                info["warnings"].append(
                    "segment %d solved to %s but measures %.2f ft" % (i, fmt(each), measured[i]))
    got = sum(values)
    if abs(got - overall_ft) * 12 > 1.0:
        info["warnings"].append(
            "solved segments sum to %s against a chain total of %s" % (fmt(got), fmt(overall_ft)))
    return values, info


def fmt(ft):
    """Feet as architects write them. 15.25 -> 15'3\""""
    if ft is None:
        return "?"
    inches = round(ft * 24) / 2.0
    f, i = int(inches // 12), inches % 12
    whole, half = int(i), i - int(i)
    s = "%d'%d" % (f, whole)
    if half:
        s += "-1/2"
    return s + '"'


def scale_from_chain(segments, pixels_per_foot):
    """What the drawing says its own scale is.

    Each believed reading gives pixels divided by printed feet. The median of
    those is a scale derived from the sheet rather than taken on trust, and a
    reading far from the median is a misread rather than a bad scale.
    Returns (ppf or None, number of readings used).
    """
    got = []
    for seg in segments:
        s, px = seg.get("stated_ft"), seg.get("length_px")
        if s and px and s > 2.0:  # short segments are not drawn to scale
            got.append(px / s)
    if len(got) < 3:
        return None, len(got)
    got.sort()
    return got[len(got) // 2], len(got)
