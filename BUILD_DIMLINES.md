# Build spec: `dimlines.py`

Turn the proven probe into a repo module. Everything here was measured on the
Putnam sheet (`plans/Putnam_House_-_floor_plans.jpg`, 8199x6607, 74.94 px/ft),
not guessed. `dimensions.py` already exists and is tested; this module feeds it.

## Why this exists

The old path tied a printed number to a room by how close the text sat to the
region. That produced false confirmations and was replaced. The new path ties a
number to the dimension line it belongs to, so a length is bound to a measured
span of pixels rather than to proximity.

## The result this has to preserve

Second-floor bottom chain, eleven segments. Tesseract read four, the vision
model was asked for two more, the solver filled the remaining five:

```
exact to the inch 7/11   within half an inch 9/11   whole chain off by 1.0 in
```

A regression test for this chain is already in `test_pipeline.py`
(`test_solve_chain_recovers_repeated_gaps`). Do not weaken it.

## What is already done

- `dimensions.py` — `parse_length`, `believe`, `group_gaps`, `solve_chain`,
  `fmt`, `scale_from_chain`. Tested, 49 tests passing.
- `dimline_probe.py` — working chain detection and three readers. Treat it as
  the reference implementation, not as the shipped module.

## What to build

### 1. `detect_chains(gray, ppf)`

Lift `find_chains`, `thin`, `erase_rules` and `prep` from `dimline_probe.py`
unchanged; they work. Add:

- **Both axes.** The probe only finds horizontal chains. Vertical chains carry
  the room depths, and without them no room can be confirmed on both sides.
  Transpose the image and re-run rather than writing the logic twice.
- **Deduplicate.** The probe's `kept` loop already drops a chain that duplicates
  another's span, preferring the one with more ticks. Keep it.
- **Reject non-chains.** Chain 0 at `y=5114` is not a dimension chain. Its
  sixteen "ticks" are stair treads and hatching, and every crop reads as noise.
  Require at least two believed readings on a line before treating it as a chain.

### 2. `find_overall(chain, lines, ppf)`

Nothing solves without the chain's total, and on this sheet the total is printed
on a *separate* line below the segmented one.

- Look for a parallel thin line within about 4 ft, spanning the same extent
  within 2%, with fewer ticks than the chain.
- Read its text as one wide crop. This is easier than a segment: the text is
  larger and sits alone.
- If no overall line is found, or it cannot be read, return `None`.
  `solve_chain` already handles `None` by leaving gaps unknown. Do not
  substitute `measured span / ppf` — that is the manifest scale coming back in
  through the side door, and it defeats the point of the check.

### 3. `read_chain(gray, chain, ppf, cheap, expensive=None)` — two tiers

This is the part that makes it usable on 4 cores. Measured costs: Tesseract
about 0.2 s per crop, TrOCR about 1.9 s, `qwen2.5vl:3b` about 132 s.

1. Run the cheap reader over every segment.
2. `parse_length`, then `believe` against the measured length. Discard the rest.
3. Call `solve_chain`. If it returns no `None` values and no warnings, stop.
   The expensive reader is never loaded.
4. Otherwise ask the expensive reader for **only** the segments still unresolved,
   and only where it can help: a gap whose measured length matches other gaps in
   its cluster will be solved by arithmetic, so do not spend 132 s on it.
5. Solve again with the new readings.

On the Putnam chain this means two expensive calls instead of eleven.

### 4. Output

One record per chain, written into the trace:

```python
{
  "axis": "h",                  # or "v"
  "y": 5537,                    # x for a vertical chain
  "x0": 4534, "x1": 7417,
  "overall_ft": 37.958,         # or None
  "overall_source": "read",     # "read" | None
  "segments": [
    {"x0": 4534, "x1": 4863, "length_px": 329, "measured_ft": 4.39,
     "value_ft": 4.667, "source": "read", "reader": "tesseract",
     "raw_text": ". 4'8\\" \\""},
    {"x0": 4863, "x1": 5026, "length_px": 163, "measured_ft": 2.18,
     "value_ft": 1.917, "source": "solved", "reader": None, "raw_text": None},
  ],
  "rejected": [...],            # straight from solve_chain
  "warnings": [...],
}
```

`source` must stay honest: `read` only where a reader produced the value and it
passed `believe`. `solved` means arithmetic. Never label a solved value as read.

### 5. Wire into `seeded.py`

- Run chain detection before room verification.
- Derive the scale with `scale_from_chain` and compare it with the manifest.
  Warn when they differ by more than 2%; the manifest has been wrong before and
  the drawing is the better authority.
- A room is confirmed when its left and right edges sit near two chain joints on
  a horizontal chain, and its top and bottom edges near two joints on a vertical
  one. The room's width is then the sum of the segments between those joints.
  This replaces the proximity matcher as the primary path.
- Keep the existing proximity matcher as a fallback for plans with no chains,
  with its cross-axis guard intact.

### 6. CLI

```
python dimlines.py plans/<image> --ppf 74.94 [--reader tesseract|trocr]
                   [--expensive ollama] [--model qwen2.5vl:3b] [--threads 4]
                   [--axis both|h|v] [--json out.json]
```

Default to cheap-only. The expensive tier is opt-in.

## Tests to add

1. A synthetic chain image: draw a line, ticks at known spacing, printed text.
   Assert the detector finds the joints within 2 px.
2. Vertical chains found after transposing.
3. `find_overall` pairs a segmented chain with the overall line below it.
4. The two-tier reader calls the expensive reader only for unresolved segments.
   Use a fake reader that counts its calls.
5. A chain of stair treads with no readable numbers is rejected, not reported.

## Three things that will bite

- **Prompt contamination.** An earlier prompt contained `7'10 1/2"` as an
  example and the model returned that string for three unrelated crops. The
  prompt in `dimline_probe.py` now describes the format and shows no example.
  Keep it that way.
- **Tolerance must scale with length.** A flat 5% threw away a correct `2'8"`
  reading on a segment measuring 2.24 ft, because short segments are not drawn
  to scale. `believe` uses about five inches of slack or 4%, whichever is
  larger. The numbers are in `dimensions.py` with the reasoning.
- **The forced sum hides error.** `solve_chain` makes the segments add up to the
  overall, so a half inch the reader dropped reappears as a one inch error in a
  solved segment, with no warning. That is 0.2% over a 38 ft chain and fine for
  areas, but it means a clean-looking chain is not proof every segment is right.

## Out of scope for now

Vector PDFs. `ingest.py` already pulls the PDF text layer; adding
`page.get_drawings()` would give exact line endpoints and make all of the above
nearly exact. There is no vector plan to test against yet, so it waits.
