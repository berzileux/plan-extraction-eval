"""Parse printed room dimensions from OCR text. Pure functions, no model."""
import re

_FT = r"(\d{1,3})\s*(?:'|ft|FT)(?:\s*-?\s*(\d{1,2})\s*(?:\"|in|IN)?)?"
_PAIR_FT = re.compile(_FT + r"\s*[xX×]\s*" + _FT)
_PAIR_PLAIN = re.compile(r"(?<!\d)(\d{1,3})\s*[xX×]\s*(\d{1,3})(?!\d)")

_QUOTES = (("’", "'"), ("‘", "'"), ("′", "'"), ("”", '"'),
           ("“", '"'), ("″", '"'), ("''", '"'), ("ʺ", '"'))
_LOOKALIKE = {"Z": "2", "z": "2", "O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"}


def normalize(text):
    """Fix common OCR misreads: curly quotes, and letters standing in for digits
    when they sit right next to a digit and a foot/inch mark (1Z'-6" -> 12'-6")."""
    if not text:
        return ""
    # OCR often doubles the foot mark before a dash ("14’'-9"): read it as one foot mark
    text = re.sub(r"(\d)\s*['’‘′`´]+\s*(?=-\s*\d)", r"\1'", text)
    for a, b in _QUOTES:
        text = text.replace(a, b)
    letters = "".join(_LOOKALIKE)
    pat = re.compile(r"(?<=\d)[" + letters + r"](?=[\d'\"\-])|(?<![A-Za-z])[" + letters + r"](?=\d{1,2}\s*['\"])")
    return pat.sub(lambda m: _LOOKALIKE[m.group(0)], text)


def _feet(whole, inches):
    f = float(whole)
    if inches is not None and inches != "":
        i = float(inches)
        if i >= 12:
            return None
        f += i / 12.0
    return f


def _find(text):
    """Return (width, depth, match) for the first valid dimension pair, or None."""
    text = normalize(text)
    for m in _PAIR_FT.finditer(text):
        w = _feet(m.group(1), m.group(2))
        d = _feet(m.group(3), m.group(4))
        if w is not None and d is not None and 3 <= w <= 60 and 3 <= d <= 60:
            return round(w, 2), round(d, 2), m, text
    for m in _PAIR_PLAIN.finditer(text):
        w = float(m.group(1))
        d = float(m.group(2))
        if 3 <= w <= 60 and 3 <= d <= 60:
            return round(w, 2), round(d, 2), m, text
    return None


def parse_dims(text):
    """Return (width_ft, depth_ft) or None. Accepts 12'-6" x 14'-0", 12' x 14', 12x14."""
    r = _find(text or "")
    return (r[0], r[1]) if r else None


def strip_dims(text):
    """Return the text with the dimension pair removed (what is left is the room name)."""
    r = _find(text or "")
    if not r:
        return clean_label(text)
    _, _, m, norm = r
    return clean_label(norm[:m.start()] + " " + norm[m.end():])


def clean_label(text):
    """Trim stray quotes, asterisks and separators OCR leaves around a room name."""
    return re.sub(r"\s+", " ", (text or "")).strip(" -:,;.'\"*`~|_")


_SINGLE = re.compile(r"(?<![\d'])(\d{1,3})\s*'\s*-?\s*(?:(\d{1,2})\s*(?:\"|''|in|IN)?)?(?![\d'])")


def all_singles(text):
    """Every single printed length in the text, in feet: 13'-6\" -> 13.5, 8' -> 8.0.
    Returns [] when the text is a 'W x D' pair (that case is handled by parse_dims)."""
    text = normalize(text or "")
    if _find(text):
        return []
    out = []
    for m in _SINGLE.finditer(text):
        v = _feet(m.group(1), m.group(2))
        if v is not None and 2.0 <= v <= 80.0:
            out.append(round(v, 2))
    return out


def all_lengths(text):
    """Every single printed length in the text, in feet, even when the text also holds a
    'W x D' pair. Used for schedules and dimension strings."""
    text = normalize(text or "")
    out = []
    for m in _SINGLE.finditer(text):
        v = _feet(m.group(1), m.group(2))
        if v is not None and 0.5 <= v <= 80.0:
            out.append(round(v, 2))
    return out
