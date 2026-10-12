# plan-extraction-eval

Measured, CPU-only extraction of room areas from 2D floor plan images, with a static results viewer. Free to run, free to host.

The point is the measurement, not the demo: every number is compared with a hand-checked value, and failures are shown.

## Method (v1)

Classical computer vision, no model download, no network:

1. Threshold dark pixels as walls.
2. Close door-sized gaps (default 4 ft, from the plan scale). In the label-seeded mode a
   wall is first carried on through a stepped door casing drawn at its end
   (`segmentation.casing_depth_ft`, `segmentation.casing_reach_ft`), so a moulded jamb
   does not make an ordinary door look wider than the doorway rule. At a door in a corner
   the casing hangs off the end of the wall running the other way; it is followed from
   that end and drawn as the jamb on the door's own axis.
3. Label enclosed empty regions and drop the exterior.
4. Convert pixels to square feet using a per-plan scale.

Why not a vision-language model for areas: published zero-shot tests report large geometric errors (about 43% mean absolute percentage error in one study on 100 plans) against about 1.6% per plan for plain pixel counting on clean plans. Models are better suited to reading labels and printed dimensions, which is step 2 below.

## Design: five layers

The sheet is treated as several streams (lines, names, dimension strings, tables) that are read separately and fused by coordinates. Status is stated per layer. Where a layer is skipped it is because paid or upload-based services are excluded, not because the layer was judged unnecessary.

| Layer | Job | Status | What is used here |
|---|---|---|---|
| 1. Vector / clean engine | Render pages, remove long lines, measure regions; read the PDF text layer when one exists | Built | OpenCV, PyMuPDF, scikit-image (`ingest.py`, `extract.py`, `seeded.py`) |
| 2. Stream splitter | Decide which part of the sheet is plan, title block, legend, table or margin | Built, optional YOLO part untested | Always on: background seeds far from any room label, plus a schedule finder (`find_schedules`). Optional: a YOLO layout detector (`--layout`, `layout_check.py`) that boxes title block, legend and drawing area. SAM 2 offered as a second opinion on room regions (`--sam2`, untested). Neither model has been run by the repo author. YOLO licence is AGPL-3.0 by default; see the note below. |
| 3A. Table OCR | Read schedules (door, window, finish) as rows and columns | Partly built | Schedule finder returns the table box and per-row lengths. Excluded by free-only rule; alternatives under review: Textract, Azure Document Intelligence. |
| 3B. Rotated / hand-lettered text OCR | Read vertical dimension strings and small lettering | Built with free engines | Tesseract, Florence-2 (local, CPU), per-string reader for vertical chains. Excluded by free-only rule; alternatives under review: Google Cloud Vision. PaddleOCR is the free alternative not yet tried. |
| 4. Coordinate fusion | Join names, lengths and regions by position, then check agreement | Built | Watershed regions + printed-length agreement (`seeded.py`); no spatial index needed at this size. |
| 5. Reasoning | Interpret the compiled text, flag disagreements | Built locally; frontier model excluded | Qwen2.5 3B via Ollama (`interpret.py`) plus code checks. Frontier hosted model: excluded by free-only rule; alternatives under review. Plans would also be uploaded. |

Excluded by free-only rule; alternatives under review: Textract, Azure (3A), Google Vision (3B), hosted frontier model (5).

For a PDF exported from CAD, layer 2 needs no OCR at all: the text is read exactly. Run `python ingest.py plans_pdf/house.pdf --inches-per-foot 0.25`, put the printed `pixels_per_foot` into `manifest.csv`, then continue as below.

The step-by-step flow, with diagrams and what each trace key records, is in [PROCESS_FLOW.md](PROCESS_FLOW.md).

## Version

`VERSION` holds the version of this code, and every run prints it and writes it into
`results.json` and every trace:

```
version:  12.13.1
group id: 20261011-063511
```

It is there because fixes arrive as a zip and a zip is easy to forget to unpack. If a
result looks the same after a change, check the version printed by the run before
looking anywhere else.

## Run

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python make_synthetic.py            # optional: draws a test plan with known areas
cp manifest.example.csv manifest.csv  # list your plans and their pixels_per_foot
python extract.py                   # writes viewer/results.json and overlay images
python score.py --results viewer/results.json   # optional: score against truth.csv, see Ground truth
```

Open `viewer/index.html` through any static server, for example `python3 -m http.server -d viewer`.

To run every layer in one go instead of calling the scripts yourself, see
[Running the whole pipeline](#running-the-whole-pipeline-run_layerssh).

### Setting the scale

```
python manifest-update.py plans/<your image>              # Putnam preview formula (11.7 px/ft at 1280 px wide)
python manifest-update.py plans/<your image> --ppf 24.0   # a scale you measured yourself
```
It adds or updates that one line in `manifest.csv`, keeps the others, and saves `manifest.csv.bak`. The default formula only holds for the same drawing with the same crop as the preview; for any other plan, measure a printed length in pixels and pass `--ppf`.

## Running the whole pipeline (`run_layers.sh`)

`run_layers.sh` runs the layers in order and prints, before it starts, exactly which
technology each layer will use. Everything is set through environment variables, so a
log always says what produced it.

```
. .venv/bin/activate
cp manifest.example.csv manifest.csv     # list your plans and their pixels_per_foot

DRY_RUN=1 bash run_layers.sh             # print the plan of work and stop
bash run_layers.sh                       # closed mode: extract.py then read_labels.py
MODE=seeded bash run_layers.sh           # seeded mode: seeded.py (survey sheets, sketches)
DIMS=1 MODE=seeded bash run_layers.sh    # add the dimension-chain layer
```

Start with `DRY_RUN=1`. It prints the technology summary and runs nothing, which is the
cheapest way to check a long run is configured the way you meant.

### The settings

| variable | default | what it does |
| --- | --- | --- |
| `MODE` | `closed` | `closed` for plans whose walls form closed rooms. `seeded` reads room names first and grows regions from them: use it for survey sheets, scans and sketches. |
| `ENGINE` | `tesseract` | Text reader in seeded mode: `tesseract`, `florence`, or `both`. `both` is much slower and on a full-resolution sheet can run for hours. |
| `DIMS` | off | `1` runs `dimlines.py` before the rooms layer: finds dimension chains, reads what it can cheaply, fills the rest by arithmetic. |
| `DIMS_VLM` | none | With `DIMS=1`, a vision model for the few segments the cheap reader misses, e.g. `qwen2.5vl:3b`. Budget about two minutes per segment on CPU. |
| `DIMS_READER` | `tesseract` | The cheap reader run over every segment. `tesseract` suits printed CAD text. On a hand-lettered or scanned sheet it reads almost nothing, and `trocr` (a handwriting model, about 1.9 s per segment on CPU) is the one to use. |
| `DIMS_CHAINS` | `16` | Chains to read **per axis**: those inside the building first, then the perimeter, each longest first. One cap across both axes lets the vertical chains, which span the whole building, crowd out the horizontal ones where most printed lengths sit; on length alone the perimeter chains, which measure window piers, crowd out the interior ones, which measure rooms. |
| `DIMS_THREADS` | `nproc` | Worker threads for `DIMS_VLM`. The default matches your core count so the machine does not oversubscribe itself. |
| `LAYOUT` | off | Seeded mode only. `1` for the default YOLO layout model, or a local `.pt` or repo id. Needs `requirements-layout.txt`. Read the licence notes below before using it. |
| `LAYOUT_REV` | unpinned | Pin the Hugging Face commit you reviewed, used with `LAYOUT`. |
| `OLLAMA_MODEL` | `qwen2.5:3b` | The language model for layer 3. |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Where Ollama is listening. If it is unreachable, layer 3 is skipped and the run still completes. |
| `GROUP_ID` | the time now | The run's group id, stamped on every layer and every trace id. Set it to re-run into an existing group. |
| `BIND` | `0.0.0.0` | Address for the viewer command printed at the end. Use `127.0.0.1` to keep the viewer off the network. |
| `DRY_RUN` | off | `1` prints the technology summary and stops. |

### Common runs

```
# a quick look at a scanned sheet, cheapest path
MODE=seeded bash run_layers.sh

# the same, plus dimension chains read cheaply
DIMS=1 MODE=seeded bash run_layers.sh

# a hand-lettered or scanned sheet: Tesseract cannot read it, TrOCR can
DIMS=1 DIMS_READER=trocr MODE=seeded bash run_layers.sh

# and with the vision model filling the few segments TrOCR still misses
DIMS=1 DIMS_READER=trocr DIMS_VLM=qwen2.5vl:3b DIMS_THREADS=4 MODE=seeded bash run_layers.sh

# keep the viewer on this machine only
BIND=127.0.0.1 MODE=seeded bash run_layers.sh

# re-run into an existing group so the traces sit together
GROUP_ID=20261011-032402 MODE=seeded bash run_layers.sh
```

### How long it takes

Measured on four CPU cores with no GPU, on one full-resolution survey sheet of about
8200 x 6600 px. Scale these by your own core count.

| run | roughly |
| --- | --- |
| `MODE=seeded` | about a minute |
| `DIMS=1` with Tesseract | a minute or two |
| `DIMS=1 DIMS_READER=trocr` | about 90 minutes on the sample sheet (216 segments) |
| adding `DIMS_VLM=qwen2.5vl:3b` | another 20 to 60 minutes |
| `ENGINE=both` on a full-resolution sheet | hours |

The chain layer reads up to `DIMS_CHAINS` chains per axis, roughly eight segments each,
and tries more than one view of every segment, so the number of reader calls runs into
the hundreds. At 0.2 s each that is nothing and at 1.9 s each it is most of an hour.
`DIMS_CHAINS=4` cuts it sharply when you only want the longest chains.

With TrOCR a segment costs about 24 s on the sample sheet, against about 6 s before the
retry described under the dimension chains. A piece that reads nothing is now shown to
the reader up to four times: its own crop, the wide crop, and both again with the line's
ragged edge cleared. That is what recovered 15'7", and it makes the full run about 1 h 25 min
for the chain layer and 1 h 32 min end to end (measured, 216 segments, four cores).
In that run 121 of the 216 pieces were retried.

While the chain layer and the language-model layer run they report progress on one
line:

```
  Putnam_House_-_floor_plans.jpg segments  84/312   27%  elapsed 2:41  ETA 7:18
  rooms  12/20   60%  elapsed 0:48  ETA 0:32
```

In a terminal that line rewrites itself; piped to a file it prints every ten segments
instead. The estimate is the average so far, which holds up here because every segment
costs about the same.

Three things happen once and are not in the table: TrOCR downloads 1.3 GB the first
time, an Ollama vision model is a 3.2 GB pull, and loading a model into RAM costs about
30 s before the first reading.

Before committing to a long run:

```
DRY_RUN=1 DIMS=1 DIMS_READER=trocr MODE=seeded bash run_layers.sh   # what it will do
python dimlines.py --only <plan> --reader trocr --max-chains 1      # time one chain
```

### What it writes

- `viewer/results.json` and `viewer/out/*_overlay.png` — rooms and the pictures of them
- `viewer/out/*_dimlines.json` and `*_dimlines_overlay.png` — chains, with `DIMS=1`
- `viewer/trace_<trace id>.json` — one trace per plan per layer
- `viewer/group_<group id>.json` — the index for the run

Collect a whole run into one file to share:

```
python collect-run.py          # writes run_<group id>.zip, without the plan images
```

Traces hold text read off your plans, so `run_*.zip`, `trace_*.json`, `group_*.json`
and `probe_*/` are all in `.gitignore` and `.dockerignore`. Run `git status --short`
before committing anyway: ignore rules do not untrack a file that is already tracked.

### Reading a trace

A trace records what every layer decided and why, which is the point of it, and it is
also several thousand lines of JSON. `trace.py` prints the parts you look at when a run
disappoints you.

```
python trace.py                           # the newest trace in viewer/
python trace.py viewer/trace_2026*.json   # one or several
python trace.py --rooms                   # just the room table
python trace.py --chains --warnings
python trace.py --all                     # every section
```

The room table sets what was measured beside what was printed, which is the comparison
that matters:

```
  room                sq ft measured        printed         read so far        verdict
  LIVING 7            293.6 14'9" x 24'6"   -               d 24'4"            only the depth was found
  KITCHEN             166.9 12'0" x 15'11"  -               -                  no printed length matches
```

`--chains` prints every segment with where its number came from, and every reading that
was rejected for disagreeing with the length it measures. Reads only; changes nothing.

### If a run goes wrong

- **`manifest.csv not found`** — copy `manifest.example.csv` and list your plans.
- **`pixels_per_foot ... is not a number`** — the manifest still holds placeholder
  text. See [Setting the scale](#setting-the-scale).
- **`No module named 'cv2'`** — the virtualenv is not active. Run `. .venv/bin/activate`.
- **Layer 3 skipped** — Ollama was not reachable at `OLLAMA_URL`. The rest of the run
  is unaffected.
- **The run is very slow** — check `ENGINE=both`, which is the usual cause, and
  `DIMS_VLM`, which costs about two minutes per segment. `vmstat 1 5` during a run
  shows whether the machine is swapping or just busy.

## Dimension chains (layer 2b)

A dimension chain is a thin line with joints, an extension line at each joint and a
printed length on every segment. It is not evidence about the geometry, it is the
geometry, already in feet. `dimlines.py` finds the chains, binds each number to the
span of pixels it belongs to, and fills the gaps by arithmetic.

```
python dimlines.py                                   # every plan in manifest.csv
python dimlines.py --only putnam.jpg --axis both
python dimlines.py --expensive ollama --model qwen2.5vl:3b --threads 4
```

Or through the pipeline:

```
DIMS=1 MODE=seeded bash run_layers.sh
DIMS=1 DIMS_VLM=qwen2.5vl:3b MODE=seeded bash run_layers.sh
```

How a length becomes trusted:

1. the cheap reader tries every segment (`--reader tesseract` for printed text,
   `--reader trocr` for hand lettering)
2. a reading that disagrees with its own drawn length is thrown away
3. what is left must sum to the chain's printed overall, and unread gaps that
   measure the same as each other are the same dimension repeated
4. only what is still unresolved goes to the expensive reader, and never a gap
   that arithmetic will fill

Measured on a 1930s hand-lettered survey sheet, one chain of eleven segments: the
cheap reader managed four, a vision model was asked for two, arithmetic supplied
five. Seven came out exact to the inch, nine within half an inch, and the whole
38 ft chain was 1 inch out.

Reader costs per segment, on four CPU cores: Tesseract about 0.2 s, TrOCR about
1.9 s, `qwen2.5vl:3b` about 132 s. That gap is why the tiers exist.

Pick the cheap reader to match the sheet. On the hand-lettered survey sheet
Tesseract read 1 of 11 segments on the best chain and nothing at all on the rest,
even at full resolution; TrOCR read 4 of the same 11. Tesseract is only the right
default for printed CAD output.

Each segment records where its number came from. `source: read` means a reader
produced it and it passed the agreement check; `source: solved` means arithmetic.
A solved value is never reported as read.

Two things to know before trusting a clean-looking chain:

- `solve_chain` forces the segments to add up to the overall, so half an inch
  dropped by a reader reappears as a one inch error elsewhere, silently.
- Without a readable overall, nothing is solved and the gaps stay unknown. That
  is deliberate: deriving the total from the measured span and the manifest scale
  would smuggle the scale back in and defeat the check.

### Tuning: rules.json and lexicon.json

Every threshold the engine can be adjusted by lives in `rules.json`, each with a
description of what it is for. The modules hold no tuning constants of their own, so
a change is a data change and the reason for it is written beside the number.

```
python ruleset.py        # every rule, its value and what it does
```

Sections: `detection` (what counts as a line, a joint, a chain, a lone dimension),
`reading` (when a number is believed, how a chain is closed, how the scale is derived),
`segmentation` (doorways and jambs closed in the mask rooms grow in), `matching` (how a
dimension is attached to a room, which dimensions a room may take at all, and what
counts as agreement) and `scoring` (how `score.py` judges a run). The list changes;
`python ruleset.py` is the place to read it, not this file.

A value outside its sane range is refused with an error rather than used. A tolerance
typed as `8` instead of `0.08` would quietly accept every room on the sheet, and
stopping is better than that. A missing file falls back to the built-in defaults, so
the pipeline runs from a bare checkout.

`lexicon.json` holds the room vocabulary. It decides which lines of text are room
names and it moves no measurement: a name is only a seed and a label. Words that are
not generic room types are listed separately under `period_or_specific`, so it stays
visible what the vocabulary has been fitted to. `MEMORIAL` is there because the sample
sheet has a Memorial Room, which is the dictionary being tuned to the drawing under
test. The OCR's own text is kept in the trace under `seeds[].ocr_text`, so what was
read can always be checked against what was reported.

### Chains and lone dimensions

Two kinds of line carry a number.

A **chain** is a run divided into segments, with a joint at each division and often a
total printed below it. Perimeter chains are like this: window piers, bays and wall
offsets, all on one line.

A **lone dimension** measures one thing between two arrowheads, with no joints in
between. Most room dimensions are drawn this way, because a room width is not
subdivided. On the sample sheet `23'7"`, `15'3"`, `17'11"`, `18'7"`, `12'6"`, `9'0"`,
`15'7"` and `13'6"` are all lone dimensions, and they are the numbers that describe
rooms.

A chain is checked against its printed total. A lone dimension has no total, so its
own drawn length is the only evidence — which is enough, and is exactly what the
agreement check is for. A line whose printed text matches the length it is drawn at
is a dimension line; a wall with a number near it is not, and will not match.

Lines with nothing text-shaped on them are skipped before any reader is called. On the
sample sheet that is about half of them.

### Why three joints, not four

A chain is a line with joints on it. The obvious floor is four joints — three
segments — but on a real sheet that throws away exactly the chains you want.

An interior chain runs wall to wall. A wall is a full-height line, and the tick
finder ignores anything longer than about 12 ft on purpose, or every wall on the
drawing would read as a joint. So an interior chain's outermost joint is invisible
and it comes back one short.

On the sample sheet the chain `15'3" | 7'10.5" | 14'10"` is found with three joints
and its middle two segments, measured at 7.77 and 14.31 ft against a printed 7.88
and 14.83. The `14'10"` is a room width. At a floor of four joints every interior
chain on the drawing was discarded and only the perimeter chains survived, and those
measure window piers and describe no room at all.

Three is the floor: two joints is one segment with nothing to cross-check it. What
keeps the false ones out is the reading requirement — with three joints there are
only two segments, and both have to be read and believed before the line counts as
a chain.

The missing end joint is then looked for where the draughtsman drew it: the arrowhead
whose tip touches the wall the line ends at. Only on a line already taken as a chain,
only beyond its outermost joint, only within `detection.end_mark_window_ft` of where the
line's ink stops and only when that stretch is long enough to carry a room dimension.
It never decides whether a line is a dimension. On the sample sheet it gives 18'7",
15'3", 15'7" and both 13'6" a piece of their own. There is no fallback: where no
arrowhead is found, the stretch stays unread. The dimension layer runs before the rooms
layer, so no wall is known at that point to stop the line at.

When a piece is read, the chain's own line is cleared from its crop first, except where
a stroke carries on across it. The line runs through the numerals on most interior
chains, and the reader took it for part of a glyph: 13'6" came back as `43' 6"` and as
`2'`, and 24'4" as `( 2474 "`. A crop with nothing text-shaped left on it is not shown
to the reader at all: given blank paper it answered `1'` on a 0.84 ft stub, and that was
believed. A piece that still reads nothing is read once more with the line's ragged edge
cleared as well, a row or two each side that otherwise leaves a dotted trace through the
numerals. Only then: clearing it on every crop recovered 15'7" but lost 17'11" and 12'6",
so a piece that read anything is never read again, and the retry can add a reading but
cannot change one. Its views are marked `retry-` in the chain file.

Two lines can cover the same run, and only one is read. A line is a copy of another only
when every joint it has sits on a joint of the other: the same line found twice, or an
overall whose joints are its ends and the chain's extension lines running through it.
Length and a nearby row are not enough. On the sample sheet that rule had dropped the
second-floor room widths (15'7" and 13'6") as a copy of the window-pier chain 3.7 ft
below them, and two first-floor chains as copies of equally long lines on the other plan.

A chain counts as inside the building when a chain on the other axis crosses it within
its run and has joints on both sides of it; a perimeter chain lies beyond the outer wall,
where those chains have stopped. The rooms layer's regions would say this directly, but
they do not exist yet when this layer runs. Inside chains are read first, so when
`DIMS_CHAINS` bites it is the perimeter that waits.

`run_layers.sh` follows the reading pass with `dimlines.py --lines-only`, which records the
lines a believed reading proves, so the rooms layer takes them out of the wall map. Before
it did, a full run left every dimension line in the walls, and the rooms they cross were
cut short even when their numbers had been read.

### The scale: the drawing's is reported, the manifest's is used

`manifest.csv` holds the scale every length is measured with. The dimension layer also
works out the scale the drawing's own readings imply, and every run prints and records
how far that is from the manifest. It does not use it unless asked.

On the sample sheet the true scale, measured off the printed 43'4" overall (3245.5 px),
is 74.90 px/ft vertically, about 75.4 horizontally, and 74.6 on the graphic scale bar.
The derived scale came out 70.50, 72.57 and 73.14 in three runs, all low. One property
of one image giving three answers is a misread, not a measurement; at 73.14 every length
is 2.4% short. So:

```
python dimlines.py ... --use-drawing-scale     # measure with the derived scale instead
python seeded.py   ... --use-drawing-scale     # the rooms layer likewise
```

`--trust-manifest` (both layers) is still accepted, keeps the manifest, and wins over
`--use-drawing-scale`. The scale actually used is recorded in each trace as
`params.pixels_per_foot_used`.

How the derived scale is worked out, for when it is wanted. There is a trap in doing
this naively. A reading is checked against the length it
measures, and that check uses the scale; so if the scale is wrong, correct readings are
thrown out as disagreeing, and whatever survives is a biased sample that confirms the
wrong scale. On the sample survey sheet a 6% error was enough to reject every room
width on the sheet.

So reading and judging are separate steps. Every segment is read first, with no
judgement. The scale is then worked out from those readings — the median of pixels per
printed foot over the longer segments, since short ones are crammed and not drawn to
scale. Only then is each reading judged, at the corrected scale. Nothing is read twice,
because reading is the expensive part and judging is arithmetic.

What the run prints when they disagree:

```
the drawing's own dimensions imply 73.14 px/ft from 25 reading(s), against 74.94 in
manifest.csv (2.4% apart); the manifest was used. The derived scale is reported, not
used; pass --use-drawing-scale to use it
```

The manifest is never written to. If you have measured the scale yourself and want it
kept, put it in the manifest:

```
python manifest-update.py plans/<image> --ppf 74.90
```

The estimate needs at least four readings and ignores any vote more than 25% from the
manifest: a misread produces a wild ratio, a genuine manifest error a few percent.

### How the rooms layer uses the chains

`dimlines.py` writes `viewer/out/<plan>_dimlines.json`, and `seeded.py` reads it on
the next run. Order matters: run the chain layer first, which is what
`DIMS=1 bash run_layers.sh` does.

A room's size is looked for in two ways, best first.

**The room's edges land on chain joints.** A joint is a wall centreline, so when the
left and right edges sit on two joints of one horizontal chain, and the top and bottom
on two joints of one vertical chain, the size is the distance the drawing states
between those exact points. Strongest, and the one to trust.

**A length is drawn across the room.** A draughtsman writes a dimension on the run it
measures, so the segment covers the thing it describes: the `23'7"` and the `15'3"`
span the Old Kitchen, and they are its depth and width. This is how a person reads the
sheet, and it catches rooms whose walls are not chain joints — on a survey of an old
building that is most of them, since the perimeter chains divide by window piers
rather than partitions.

The segment's whole span is tested, not just where the text sits. It must lie within
the room's extent, cover most of it, and sit beside or inside the room on the other
axis. The span test is what stops a narrow hall claiming the `15'7"` run drawn along
the bottom of the bedroom beside it: the midpoint falls inside the hall, but the run
reaches well past it.

**A length is drawn across the room's name.** The last resort, and the one that
survives a bad outline. Every other rule keys off the traced region, so when the
region is wrong the correct dimension is rejected with it: the Old Kitchen traces as
12'0" x 15'11" when the room is 15'3" x 23'7", and no test against that box can accept
the true width. The name, though, is read where the architect wrote it, in the middle
of the room, and the dimension is drawn across the room. A run that passes over the
name and lies near it is that room's dimension, whatever the outline says. Here the
nearest run wins rather than the longest, because without an outline there is nothing
to stop a longer run two rooms away being picked.

**A neighbour in the same band already knows it.** Two rooms between the same pair of
walls have the same depth; two rooms in the same column have the same width. That is
not a guess, it is what being in the same band means. A draughtsman writes each
dimension once, so on the sample sheet the bath has its width written in it and no
depth, while the bedroom beside it has the depth and no width. Separately neither can
be confirmed; together both can. A borrowed value is recorded as borrowed, naming the
room it came from, and still has to agree with the measured area.

Where a room holds several qualifying lengths on one axis, the longest wins.

Either way `dims_source` says `dimension chain` and `chain_refs` names the chains used.

A room agrees with the drawing when its **width** matches the printed width and its
**depth** matches the printed depth, each within 8%. Not when its area matches width
times depth: a room is not a rectangle, and on the sample sheet the Dining Room fills
293 sq ft of a 14'10" x 24'4" envelope because a stair and two closets are cut out of
it. Its width and depth are right to under one percent. Comparing areas would call
that a 19% disagreement and throw away a correct answer. The area is still reported,
as how much of the envelope the room fills.

What it refuses to do:

- joints from two different chains are never added together
- a distance is never summed across an unread segment, though the rest of that chain
  stays usable: on `15'3" | 4" | 8'7" | 2" | 13'6"` the closet jambs are too small for
  any reader, and the parlour either side of them is still measured
- one side on a chain is not a confirmation; the room still needs a human check
- a dimension on another floor plan is never used: the plans on a sheet are found from
  the room regions, and a chain drawn between two plans belongs to neither
- a dimension behind any wall other than the room's own outer wall is never used
- on a chain drawn outside the building, only a distance between two partitions counts;
  window piers are not rooms
- band sharing lends only values found on the room's own walls, and records where each
  came from (`chain_borrowed_from`)
- band sharing never crosses plans: a room lends only to a room on the same plan, and a
  room not clearly on one plan neither lends nor borrows. The sample sheet's two floors
  sit side by side, and BED ROOM C on the second floor spans the same band as WEST
  PARLOR on the first; it took the parlour's 19'4" as its depth and agreed by
  coincidence. Each room's plan is recorded in `results.json` as `plan`
- a room whose region reaches past its own walls cannot agree, whatever its sides
  measure. The pipeline has no truth box, so it uses the walls the room's own label sees:
  a fan of rays from the label to the first wall on each side, the median taken. A region
  more than `matching.leak_margin_ft` past one of them went round a wall end, through an
  opening, and its size is not the room's (`region_leak_ft`, and the reason, on the
  room). WEST PARLOR read 15'3" x 19'4" correctly and agreed while its region ran 1.20 ft
  past its east wall. It is a floor, not the truth box: it misses a leak whose opening
  lies straight ahead of the label (BED ROOM A into its closet, TOILETS into the stair
  hall), and it refuses a room that faces something inside itself on one side: LATER
  KITCHEN, which wraps round its stair and cupboard, and WEST PARLOR, whose label faces
  the fireplace hearth projecting into the room, so its region fills its own walls
  exactly, both printed sides agree within 4%, and it is refused all the same. A fan's
  spread instead of its median was tried for the first limit and not kept: it did not
  catch BED ROOM A and refused HALL (front), which is correct

Rooms that no chain reaches fall back to the older checks: a printed `W x D` inside
the region, then printed lengths that line up with it on both axes.

### When a chain reads nothing

`dimlines.py --debug` prints every reader attempt for every segment and saves each
crop that would not parse, so a silent chain can be told apart from a bad one.

```
python dimlines.py --only <plan> --max-chains 2 --debug
python dimlines.py --help        # every flag, with what it does
```

Every run also keeps, in `<plan>_dimlines.json`, what became of each candidate:
`single_candidates` for lone lines and `chain_candidates` for every chain read,
including chains dropped for carrying no number, with the text each reader view
returned and the outcome. `--crops-dir D` saves the images of every segment that read
nothing.

Partial passes, for working on one part without the full half hour. Each writes only to
the directory it is given and never touches the dimension layer's own output:

```
python dimlines.py --only <plan> --reader trocr --singles-only --singles-dir D   # lone lines only
python dimlines.py --only <plan> --reader trocr --chains-only v:863,h:4834:6000 --chains-dir D
python dimlines.py --only <plan> --lines-only --lines-dir D [--dims-dir E]       # no reading at all
```

`--chains-only` takes `axis:position` (the row of a horizontal line, the column of a
vertical one, in sheet pixels), plus `:along` to pick between two lines on one row.
`--lines-only` writes `<plan>_line_geometry.json`: the detected lines that carry a
believed reading in the dimension file (from `--dims-dir`, default `viewer/out`).
`seeded.py` reads that file from its `--out` folder's `out/` if it is there and takes
those lines out of the wall mask before growing rooms, so a room stops at its wall and
not at the dimension line drawn across it. A line nothing has read stays: telling a
dimension line from a wall by its shape broke walls every time it was tried on the
sample sheet.

A reading on a line that does not survive as a chain is kept as a lone dimension over
the run of pieces it clearly fits, because beam lines and door swings crossing an
interior dimension line cut it into pieces that the number no longer matches. Where the
draughtsman's own arrowheads bound a run that fits, at each end and pointing out of it,
that run is the one: closest fit assumes the drawing is to scale, and 18'7" at the bottom
of BED ROOM C is drawn 0.46 ft short, which made the run across the partition above it
the closer fit. Each lone dimension says how it was placed in `placed_by`.

Read the output this way:

- **attempts are empty** (`clean=''  raw=''`) — the reader saw nothing it could use.
  On hand lettering that means the wrong reader: try `--reader trocr`.
- **attempts show text that will not parse** (`'28'`, `'70 7'`) — the reader is
  working and the glyphs are being mangled. Look at the saved crops: if they are
  legible, the reader is the problem, and if they are not, the crop is.
- **a chain is dropped with readings below two** — that is deliberate. Stair treads
  and hatching make convincing joints and carry no numbers.

`BUILD_DIMLINES.md` has the design notes and what is still unbuilt.

## Step 2: read labels and printed sizes with a model (local, CPU)

`read_labels.py` runs Florence-2 base (OCR with regions) over the plan in overlapping tiles, matches each text item to the room it falls inside, parses printed sizes like `12'-6" x 14'-0"`, and compares them with the area measured by code. Rooms where the two disagree by more than 10%, or where no printed size was found, are marked for human review. The model reads text only; it measures nothing.

```
pip install -r requirements-model.txt
python read_labels.py --trust-remote-code
```

Notes:
- `--trust-remote-code` is required because Florence-2 loads Python code from its model repository. Read that code first, then pin `--revision` to the commit you reviewed.
- `transformers` is pinned to 4.46.3 because newer versions have changed how this model's remote code behaves. If you upgrade, expect to debug.
- First run downloads the model weights (several hundred MB). CPU runs take seconds to minutes per tile.
- Tested here: size parsing, text-to-room matching and the review flags, using simulated OCR output (`python test_pipeline.py`). Not tested here: the actual model call, which needs a machine that can reach Hugging Face.

## Step 3: a small local language model interprets the text (optional)

`interpret.py` sends the raw OCR strings found inside each room (text only, no images) to a model running on your own machine through [Ollama](https://ollama.com), and asks for the room name and printed size as JSON.

```
ollama pull qwen2.5:3b        # confirm the model name with `ollama list`
python interpret.py --model qwen2.5:3b
```

The model proposes; code verifies:
- A size is accepted only if both numbers sit in the same printed pair in that room's text (such as `12 by 14` or `12'-6" x 14'`). Loose digits elsewhere in the room's text never count. A number the model made up is ignored and noted.
- If the rules from step 2 and the model read different sizes, the room is flagged.
- Anything the model returns is type-checked and sanitized. A failed or malformed reply never crashes the run.
- The script refuses to talk to a non-local server unless you pass `--allow-remote`, because plan text would leave your machine.

Untested here: the real model. Tested with a stub server and unit tests (`python test_pipeline.py`). A 3B model may be weak at this; measure before relying on it.

## Experimental: label-seeded mode for drawings without closed walls

`extract.py` floods enclosed empty space. On survey sheets, sketches and plans with dashed or missing walls that shatters a house into dozens of fragments (the repo's own plan-level warning fires). `seeded.py` goes the other way, and starts from the text:

1. Read the sheet in small tiles, upscaled, on a copy with long lines removed, so a name touching a wall or stair hatching is still readable.
2. Keep only text that matches a room-name lexicon (`--lexicon-extra OFFICE,SCULLERY` to add words). Each hit is a seed.
3. Grow each seed into the free space between long ink lines (watershed). Open space far from every label (margins, title block, notes) gets a background seed so no room can claim it.
4. Compare each region with printed lengths found on the sheet (horizontal text, and two rotated passes for vertical text). A room is marked `ok` only if a printed `W x D`, or printed lengths lined up with the region on both axes, agree with the measured region within tolerance. Everything else is flagged.

```
python seeded.py --manifest manifest.csv --plans plans --out viewer     # then open the viewer as above
python seeded.py --only putnam.png --fast                               # two OCR passes instead of four
MODE=seeded bash run_layers.sh                                          # same, then the local model step
DRY_RUN=1 ENGINE=both LAYOUT=1 MODE=seeded bash run_layers.sh          # print which technology each layer would use, then stop
```

Do not run `read_labels.py` after `seeded.py`: it re-measures rooms with the closed-wall method and overwrites the results. `interpret.py` is safe to run afterwards.

The text reader is Tesseract by default (`sudo apt install tesseract-ocr`). `--engine florence --trust-remote-code` (or `ENGINE=florence MODE=seeded bash run_layers.sh`) uses the same Florence-2 reader as `read_labels.py` instead, and `--engine both` pools the two readers (slowest; each catches text the other misses); same review rules for `--trust-remote-code` apply. On the one real sheet tried (a 1280 px preview), Florence read 56 printed lengths where Tesseract read none, but missed several room names Tesseract caught; neither read vertical lengths. Compare the engines on your own plans before choosing one. It runs on the original image at full size: OCR is tiled, geometry runs on an internal copy, so a large scan is fine and a small preview image loses information. Do not downscale before running it.

What to expect, tested here on a 1280 px preview of a US Historic American Buildings Survey drawing (not included in this repo):
- Names found for most rooms, including ones plain OCR missed (West Parlor, Bed Rooms C and D). Some are partly read ("BEDROOM" for "BED ROOM B").
- Regions follow the walls for most rooms but are approximate. They can absorb a neighbouring unlabelled space (a stair hall, a room whose name is not in the lexicon), and shapes with low solidity are flagged.
- At that resolution the printed lengths could not be read, so no room could be confirmed and all were flagged. Run it on the original full-resolution file to give the dimension check a chance.
- Verification is agreement, not proof. A length that lines up with a region can match by coincidence.

The same rule holds as everywhere else: the model or OCR proposes, code checks, and nothing is marked ok without a printed number agreeing.

### Run trace (what to check)

Every `seeded.py` run writes `viewer/trace_<id>.json` next to `results.json`. The id looks like `20261011-024724-7f53bd`. The middle part, `20261011-024724`, is the **group id**: one time snapshot taken when the run starts, shared by every plan and every layer in that run. The last part is unique per plan. So two plans from one run are `20261011-024724-7f53bd` and `20261011-024724-81b1b1`, and the next run has a different group id. `run_layers.sh` takes the snapshot once and exports it as `GROUP_ID`, and `extract.py`, `read_labels.py`, `seeded.py` and `interpret.py` all stamp it into `results.json` (top-level `group_id`, plus `layers.<name>` with the group each layer ran in). If a layer was run on its own later, its group differs, and that shows. Set your own with `GROUP_ID=name bash run_layers.sh` or `python seeded.py --group name` (letters, digits, `-`, `_`). A run also writes `viewer/group_<id>.json`, an index of its plans with trace ids, room counts and confirmed counts. The viewer shows `group ... | trace ...` under each plan, and each room carries `trace_ref`. To send a run for review, `python collect-run.py` packs the newest run (results, index, traces, overlays, manifest; not the plan images) into `run_<group id>.zip`; `python collect-run.py <group id>` packs a specific one. It records:
- `stages`: name and seconds for each stage (OCR passes, segmentation, dimension reader).
- `seeds`: every room-name hit, with the decision (`kept`, or `collapsed into ...`).
- `dims`: how many printed lengths each pass read, horizontal and vertical.
- `rooms`: per room, the measured box, the printed lengths aligned to it on each axis, and why it is `ok` or flagged.
- `schedules`: tables found, with rows and lengths.
- `warnings`: for example "no vertical printed lengths were read". If this appears, no room can be confirmed on both axes, and that is the first thing to look at.

Copy `results.json` and the trace file after each run; the next run overwrites `results.json`.

### Additions in this mode

- Vertical dimension reader: finds each dimension-string blob on a copy that keeps strokes crossing the dimension line, crops it, rotates both ways for vertical strings, upscales to a readable size and reads it as one line. Only readings that contain a foot length are kept. `--no-dimension-reader` turns it off. Tested on a synthetic sheet only; on the one real sheet tried, no engine read the vertical chain at 1280 px.
- Unlabelled spaces (a hall, a room whose name is not in the lexicon) get no region: only a room name read on the sheet seeds one. The shape-based finder that used to look for them, and its `--auto-seeds` / `--no-auto-seeds` flags, were removed in 12.32.0. On the sample sheet it never helped: it split named rooms, cut the only region right on both sides, found neither unlabelled hall, and produced a false agreement in the bath's double-wall cavity. Seeding enclosed spaces that hold no label and no printed size is the open item for them.
- Schedule finder: three or more stacked lines that each hold two or more lengths are treated as a table. Its text is excluded from room matching so a door schedule does not become a room size. It finds the table; it does not read cells reliably.
- SAM 2 second opinion (`--sam2`, optional): not in `requirements.txt`. Install per its repository (`pip install git+https://github.com/facebookresearch/sam2.git`), check its licence yourself, and expect a weights download. Each room gets a `sam_iou` score and a note when the two methods disagree. Agreement never confirms a room. Untested against the real model here.
- `--engine both` pools Tesseract and Florence-2; `--lexicon-extra` adds names.
- Two labels joined by marks into one OCR line (`HALL ----- BATH`) are cut into two seeds when the name words are more than 3 ft apart. Lines from Florence-2 carry no word positions and are not cut.
- A label that sits on a wall line is moved to the nearest free space before growing its region.
- An OCR line holding 4 or more lengths is a whole dimension chain read as one string. Its lengths have no usable positions, so it is skipped and counted in the trace warnings.
- Every run prints the sheet size implied by the manifest scale. A wrong scale poisons every number, and this is the quickest check.

### Optional: YOLO layout splitter (`--layout`)

A detector that boxes the title block, legend, tables (if the model has that class) and the drawing area. It does not read anything and does not measure anything.

```
pip install -r requirements-layout.txt
python seeded.py --layout                                  # default model, see layout_check.py
python seeded.py --layout path/to/weights.pt               # a file you downloaded and checked
python seeded.py --layout GreenMap/yolo11x-blueprint-layout-detector --layout-revision <commit>
LAYOUT=1 ENGINE=both MODE=seeded bash run_layers.sh        # same through the wrapper
```

What it changes:
- Text inside a title-block, legend or table box is set aside, so a legend line like `KITCHEN` or a scale note is not read as a room or a length.
- Those boxes are planted as background, so no room region grows into them.
- A room label outside every drawing-area box is a warning in the trace, not a deletion. A wrong box must not remove a real room.
- Every box (class, confidence, coordinates) is written to the trace under `layout`, and `<plan>_layout_overlay.png` draws them. Check that overlay first: red boxes are set aside, green is the drawing area.
- Trace warnings to look for: "room-name text inside layout boxes was set aside" (a real room may be inside a wrong box), "layout detector found no boxes", and "a title/legend/table box sits mostly inside the drawing area".

Licence and safety, read before use:
- Ultralytics code, and models trained with it, are AGPL-3.0 by default; a model card may state a different licence, and the two can conflict. This repo does not bundle the code or weights and does not list Ultralytics in `requirements.txt`. Decide the licence question yourself before using it anywhere beyond your own machine.
- Weights are `.pt` files, which are Python pickles and can run code when loaded. Pin `--layout-revision` to a commit you have looked at.
- Untested here (Hugging Face is not reachable from the build environment). Only a fake detector is used in `python test_pipeline.py`. Whether the default model works on hand-lettered or scanned sheets is unknown; its card does not name its training data. YOLO11x is a large model; expect a slower CPU run than the nano models.

## Deploying the viewer

The image serves `viewer/`: `index.html`, `results.json` and the overlays in
`viewer/out/`. `.dockerignore` keeps the trace files out, because a trace holds every
line of text read off a plan and the overlays and results are enough to look at.

Publish a drawing only if its licence allows it. The overlays are pictures of the
plans, so look at them before you deploy.

## Ground truth and scoring

`truth.csv` is the answer key, and only `score.py` reads it. One row per room on the
drawing, including rooms with nothing printed:

```
plan,room,printed_w_ft,printed_d_ft,w_source,d_source,notes,box_x0,box_y0,box_x1,box_y1
```

- `printed_w_ft`, `printed_d_ft`: the size printed on the sheet, in decimal feet, blank
  when not printed.
- `w_source`, `d_source`: `printed` (read off the sheet), `band` (shared from a
  neighbour between the same two walls) or `none`.
- `box_*`: where the room is on the full-size sheet, in pixels, on the inner face of
  its walls.

Take every value from the drawing, by eye. Never from the tool's own output.
`truth.example.csv` shows the shape.

```
python score.py --results <out>/results.json [--truth truth.csv] [--json summary.json]
```

Regions are matched to rooms by where they were grown from, never by the label read,
so a misread name does not lose a room. For each plan it prints rooms found, the error
of each measured side against its printed size (printed and band sides kept apart),
regions that reach outside their room by more than a wall's thickness (one wall's,
also where two walls stand back to back with a cavity: a region in the cavity reached it
through an opening and is outside its room), regions that match no room, and agreement split three ways: agreed and correct, agreed and wrong,
not agreed. It writes nothing unless `--json` is given, and **exits 1 if any agreement
is wrong**.

### What "confirmed" means

A room is confirmed when its measured width matches its printed width and its measured
depth matches its printed depth, each within the tolerance in `rules.json`
(`matching.side_tol_rel`), and its region lies inside its own walls. Never area
against width times depth: a room with a stair cut out of it fills far less than its
envelope and is still correct. A room the pipeline marks as agreeing when any of that
fails is a false agreement, the one result this project must never ship.

### The current result on the sample sheet

Two counts, kept apart:

- **Both sides printed: 2 of 18.** BED ROOM B, 17'11" x 9'0", measured 17.14 x 9.36 ft,
  and BED ROOM C, 15'7" x 18'7", measured 15.29 x 18.00 ft, both inside their walls.
- **Scorer-confirmed: 4 of 18.** Those two, plus BATH, 12'6" x 9'0", measured
  12.64 x 8.93 ft, and BED ROOM D, 13'6" x 18'7", measured 12.93 x 18.00 ft, both inside
  their walls. Neither depth is printed for that room: BATH's 9'0" is BED ROOM B's and
  BED ROOM D's 18'7" is BED ROOM C's, each shared across the band the two rooms span, so
  they are confirmed by the scorer and not on printed sizes alone.

No false agreements. **This is the result of one full run**, not of a prepared input:

```
DIMS=1 DIMS_READER=trocr MODE=seeded bash run_layers.sh
```

on the full-resolution survey sheet, default settings, 1 h 32 min on four cores. The
pipeline found, selected and read every line these numbers come from by itself, and the
lines those readings prove were taken out of the wall map in the same run. Earlier
figures in this project were measured on a dimension file assembled from targeted reads
of named lines; a real run then reported 0 confirmed, because the duplicate rule dropped
three of those lines and the line geometry was never written. Eight rooms carry a size
on both sides and four of them are refused. Three of the four are refused by the
pipeline's own leak check. EAST PARLOR and OLD KITCHEN do reach past their walls. WEST
PARLOR does not: its label faces the hearth (see the leak check's limits above), so a
room that is right on both printed sides is refused. The fourth, a hall, measures 19%
narrower than the 5'0" matched to it.

The rest are not confirmed, for three reasons:

- **Nothing is printed.** The later kitchen, new bay, toilets and memorial room, and
  the attic's depth, carry no size, so they always need a human.
- **The number was not read.** 8'7" stays unread even on the retry, and 14'10" misses
  the tolerance on the piece it sits on by 0.01 ft.
- **The region is wrong.** Rooms still flow through wide openings into the hall or
  vestibule beside them, or stop at a dimension line nobody has read.

Treat any number on the sample sheet other than BED ROOM B's and BED ROOM C's, and BATH's
and BED ROOM D's on their shared depths, as unverified.

## Sample plans

Not included. Use plans whose license allows your use, and keep them out of the repo (the `plans/` folder is git-ignored). Check each dataset's license yourself: CubiCasa5k is CC BY-NC-SA 4.0 (non-commercial).

## Limits

- Needs a known scale per plan (pixels per foot). Wrong scale means wrong areas.
- Doors without a visible gap, hatched walls, furniture and dimension lines can merge or split rooms.
- Tested on synthetic plans (including open plans with vertical dimension chains and an unlabelled room) and one real survey drawing at preview resolution. The vertical reader, unlabelled-space finder, schedule finder, Florence-2 in `seeded.py` and SAM 2 have not been validated on real drawings by this repo's author. That proves the code runs, not that it works on arbitrary drawings.
- OCR on small or rotated text in real plans is likely to miss items. Missing text is flagged for review, not guessed.
- No handling of curved walls, multiple floors on one sheet, or sloped ceilings.

## Deploy (static viewer)

```
gcloud artifacts repositories create plan-extraction-eval --repository-format=docker --location=us-central1
gcloud builds submit --config cloudbuild.yaml .
```

Run `extract.py` first so `viewer/results.json` and `viewer/out/` exist before building. They are git-ignored, so build from your machine, or remove those two lines from `.gitignore` once you have reviewed the contents.

All data shown is from public sample plans. Do not upload anyone's client drawings.
