"""A run trace for seeded.py: one id per run, per-stage counts and timings, and every decision
(which label reading won, which printed lengths lined up with which region, why a room was
flagged). Written as JSON next to results.json so each number in the viewer can be followed
back to the text and pixels that produced it. Pure Python, no dependencies."""
import json
import os
import re
import time
from pathlib import Path
import uuid
from contextlib import contextmanager


def _plain(o):
    """Make numpy scalars/arrays and other odd values JSON-safe."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, "tolist"):
        return _plain(o.tolist())
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")


def check_label(label):
    """A group id ends up in file names, so only plain characters are allowed."""
    if not _LABEL.fullmatch(label or ""):
        raise ValueError("group id must be 1 to 32 letters, digits, '-' or '_', got %r" % (label,))
    return label


def new_group(explicit=None):
    """The group id of a run: --group / GROUP_ID if given, else the time now (YYYYMMDD-HHMMSS).
    run_layers.sh takes this snapshot once and exports GROUP_ID, so every layer and every plan in
    that run carries the same group id."""
    return check_label(explicit or os.environ.get("GROUP_ID") or time.strftime("%Y%m%d-%H%M%S"))


def version():
    """What is in VERSION, next to this file. Stamped on every run so a result can
    always be traced back to the code that produced it, which matters when a fix is
    sent as a zip and may or may not have been unpacked."""
    try:
        return (Path(__file__).with_name("VERSION")).read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def stamp(data, layer, group=None):
    """Record in a results dict which run (group id) a layer's output came from. The first layer
    to write sets the top-level group_id; each layer is listed under `layers`."""
    gid = new_group(group)
    data.setdefault("group_id", gid)
    data.setdefault("version", version())
    data.setdefault("layers", {})[layer] = {"group_id": gid, "version": version(),
                                            "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    return gid


class Trace:
    """id: <group id>-<6 hex>, for example 20261011-024724-7f53bd. The group id is the time snapshot
    of the run, shared by every plan and every layer in it; the hex part is unique per plan.
    Without a group the time now is used. The start time is also stored as `started`."""

    def __init__(self, plan, params=None, group=None):
        tag = uuid.uuid4().hex[:6]
        group = check_label(group) if group else time.strftime("%Y%m%d-%H%M%S")
        self.id = "%s-%s" % (group, tag)
        self.t0 = time.time()
        self.data = {"trace_id": self.id, "group_id": group, "version": version(), "plan": plan,
                     "started": time.strftime("%Y-%m-%d %H:%M:%S"), "params": _plain(params or {}),
                     "stages": [], "seeds": [], "dims": [], "rooms": [], "schedules": [], "layout": [], "warnings": []}

    @contextmanager
    def stage(self, name, **info):
        t = time.time()
        rec = {"stage": name, **_plain(info)}
        try:
            yield rec
        finally:
            rec["seconds"] = round(time.time() - t, 2)
            self.data["stages"].append(rec)

    def add(self, key, value):
        self.data[key].append(_plain(value))

    def warn(self, msg):
        self.data["warnings"].append(msg)

    def write(self, path):
        self.data["total_seconds"] = round(time.time() - self.t0, 2)
        with open(path, "w") as f:
            json.dump(_plain(self.data), f, indent=1)
        return path
