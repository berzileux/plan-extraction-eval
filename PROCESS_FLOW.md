# Process flow

How a plan image becomes a list of rooms, which parts are code and which are models, and where each decision is recorded. Written from the code in `seeded.py` (label-seeded mode); the closed-wall mode is the short path at the end.

Legend: solid box = built and tested here. Dotted box = optional, off by default, not run by the repo author. Red dashed box = excluded by free-only rule; alternatives under review.

## 1. Whole pipeline

```mermaid
flowchart TD
    IN["Plan image + pixels_per_foot from manifest.csv"] --> CHK["Checks: file inside plans/, size limit, readable"]
    CHK --> L1

    subgraph L1["Layer 1 - clean engine"]
        direction TB
        T1["Text layer A: long lines removed"]
        T2["Text layer B: raw image"]
        T3["Text layer C: lines removed, crossing strokes kept"]
    end

    L1 --> L3B

    subgraph L3B["Layer 3B - text reading, free engines"]
        direction TB
        O1["Horizontal OCR passes - Tesseract, Florence-2 or both"]
        O2["Rotated OCR passes - clockwise and counter-clockwise"]
        O3["Dimension-string reader - crops each string, turns it upright, reads one line"]
    end

    L3B --> L2

    subgraph L2["Layer 2 - stream splitter"]
        direction TB
        Y["YOLO layout detector - title block, legend, drawing area"]:::optional
        S["Schedule finder - 3+ stacked lines with lengths"]
        X["Set aside text inside those boxes"]
    end

    L2 --> SEED["Room-name seeds - lexicon match, fuzzy, nearby duplicates collapsed"]
    L2 --> DIMS["Printed lengths - horizontal and vertical lists"]

    SEED --> SEG
    subgraph SEG["Layer 1 again - geometry"]
        direction TB
        G1["Long ink lines to walls, dilated; dimension lines a believed reading proves taken out (line geometry from the dimension layer)"]
        G1b["Door gaps closed; a stepped casing at a wall end carried on as its jamb, also at a corner"]
        G2["Distance transform"]
        G3["Watershed from name seeds + background seeds + layout boxes"]
        G4["Regrow into the wall band"]
    end

    SEG --> FUSE
    DIMS --> FUSE
    subgraph FUSE["Layer 4 - coordinate fusion"]
        direction TB
        F1["Per room: printed W x D inside region, or aligned printed lengths on both axes"]
        F2["Agree within 8 percent and region inside the walls its label sees? yes: ok. otherwise: needs human check"]
    end

    FUSE --> SAMC["SAM 2 second opinion - sam_iou only, never confirms"]:::optional
    FUSE --> OUT
    SAMC --> OUT
    OUT["results.json + overlay PNG + trace_ID.json"] --> L5

    subgraph L5["Layer 5 - reasoning, local"]
        direction TB
        Q1["Cleaned per-room text"]
        Q2["Qwen2.5 3B via Ollama proposes name and size"]
        Q3["Code accepts a size only if both numbers are in one printed pair"]
    end

    L5 --> VIEW["Static viewer - banner, rooms, reasons"]

    EXA["Table OCR: Textract, Azure"]:::excluded -.-> L3B
    EXB["Rotated text OCR: Google Vision"]:::excluded -.-> L3B
    EXC["Hosted frontier model"]:::excluded -.-> L5

    classDef optional stroke-dasharray: 3 3
    classDef excluded stroke:#c00,stroke-dasharray: 6 4,color:#c00
```

The OCR layer is drawn before the splitter because that is the code order: the layout detector runs on the finished OCR results and filters them, it does not run first.

## 2. What decides "ok"

```mermaid
flowchart TD
    R["A room region with a name seed"] --> A{"Printed W x D pair inside the region?"}
    A -- yes --> B{"Both within 8 percent of measured width and depth?"}
    A -- no --> C{"Printed lengths lined up with the region on both axes?"}
    C -- yes --> B
    C -- no --> N["Needs human check: no printed length matches this region"]
    B -- yes --> L{"Region inside the walls its own label sees?"}
    L -- yes --> OK["ok"]
    L -- no --> N4["Needs human check: the region reaches past its own walls, through an opening"]
    B -- no --> N2["Needs human check: printed and measured sizes disagree"]
    U["Space with no room name read"] --> N3["No region: nothing seeds it, so nothing is measured"]
```

Nothing is marked ok on shape, OCR confidence, SAM 2 overlap or model opinion alone. Agreement with a printed number is still not proof; a length can match by coincidence.

## 3. Where each step is recorded

Every `seeded.py` run writes `trace_<id>.json` (id format `<group id>-<6hex>`, for example `20261011-024724-7f53bd`; the group id is one time snapshot per run shared by every plan and layer, and `viewer/group_<id>.json` indexes the run). Each room in `results.json` carries `trace_ref` pointing at it.

| Flow step | Trace key | What to check |
|---|---|---|
| OCR passes, dimension reader, layout detector, segmentation, SAM 2 | `stages` | seconds and item counts per pass; a pass with 0 items is a silent failure |
| Room-name seeds | `seeds` | `kept` or `collapsed into ...`; a real room collapsed into a neighbour |
| Printed lengths | `dims` | orientation, position and raw OCR text; junk readings |
| Layout boxes | `layout` | class, confidence, box; open `<plan>_layout_overlay.png` |
| Schedule finder | `schedules` | table box and rows; table text must not appear in any room |
| Room verdicts | `rooms` | `aligned_printed_lengths` per axis with `rel_err`, reasons, `sam_iou` |
| Anything suspicious | `warnings` | for example "no vertical printed lengths were read", "room-name text inside layout boxes was set aside" |

## 4. Closed-wall mode (short path)

For plans whose walls form closed rooms.

```mermaid
flowchart LR
    I["Image + scale"] --> E["extract.py: threshold, close door gaps, label enclosed regions, area in sq ft"]
    E --> RL["read_labels.py: Florence-2 or PDF text layer, size parsing, text matched to the room it falls inside"]
    RL --> IN2["interpret.py: local model, grounded sizes"]
    IN2 --> V["Viewer"]
```

Do not run `read_labels.py` after `seeded.py`: it re-measures rooms with the closed-wall method and overwrites the seeded results.

## 5. Run order

```
MODE=closed bash run_layers.sh                              # walls form closed rooms
MODE=seeded ENGINE=both bash run_layers.sh                  # names first, then regions
DIMS=1 DIMS_READER=trocr MODE=seeded bash run_layers.sh     # dimension chains read first, then the lines they prove kept out of the walls
LAYOUT=1 MODE=seeded ENGINE=both bash run_layers.sh         # plus the optional YOLO splitter
python seeded.py --sam2                                     # plus the optional SAM 2 second opinion
```

Copy `viewer/results.json` and the trace file after each run; the next run overwrites `results.json`.
