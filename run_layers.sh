#!/usr/bin/env bash
# Runs the pipeline layer by layer. Needs manifest.csv and images in plans/.
# Optional environment variables:
#   OLLAMA_URL   (default http://127.0.0.1:11434)
#   OLLAMA_MODEL (default qwen2.5:3b)
#   MODE         closed (default): walls form closed rooms, use extract.py + read_labels.py
#                seeded: survey sheets and sketches, use seeded.py (reads names first)
#   ENGINE       text reader for seeded mode: tesseract (default), florence, or both
#   LAYOUT       seeded mode only. Set to 1 for the default YOLO layout model, or to a local .pt file
#                or repo id[:filename]. Needs requirements-layout.txt. Optional, off by default.
#   DIMS         1 to run the dimension-chain layer (dimlines.py) before the rooms layer.
#                Off by default. It finds the chains, reads what it can cheaply, and fills
#                the rest by arithmetic.
#   DIMS_READER  cheap reader for DIMS: tesseract (default) or trocr. Tesseract is for printed
#                CAD text; on hand-lettered or scanned sheets it reads almost nothing and trocr
#                (a handwriting model, about 1.9 s per segment on CPU) is the one to use.
#   DIMS_VLM     with DIMS=1, name a vision model for the few segments the cheap reader
#                misses, e.g. qwen2.5vl:3b. Slow: budget about two minutes per segment on CPU.
#   DIMS_CHAINS  longest chains to read per axis (default 16)
#   DIMS_THREADS worker threads for DIMS_VLM (default: nproc, so it does not oversubscribe)
#   BIND         address for the viewer command printed at the end (default 0.0.0.0, reachable from the
#                network; use 127.0.0.1 to keep it on this machine)
#   GROUP_ID     group id of this run (default: the time now, YYYYMMDD-HHMMSS, taken once here). It is
#                stamped on every layer's output and is the start of every trace id.
#   LAYOUT_REV   pin the Hugging Face commit you reviewed (used with LAYOUT)
#   DRY_RUN      set to 1 to print the technology summary and stop, without running anything
set -euo pipefail
cd "$(dirname "$0")"

URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
MODE="${MODE:-closed}"
ENGINE="${ENGINE:-tesseract}"
LAYOUT="${LAYOUT:-}"
LAYOUT_REV="${LAYOUT_REV:-}"
EXTRA=()
if [ -n "$LAYOUT" ]; then
  if [ "$LAYOUT" = "1" ]; then EXTRA+=(--layout); else EXTRA+=(--layout "$LAYOUT"); fi
  if [ -n "$LAYOUT_REV" ]; then EXTRA+=(--layout-revision "$LAYOUT_REV"); fi
fi

DIMS="${DIMS:-}"
DIMS_READER="${DIMS_READER:-tesseract}"
DIMS_VLM="${DIMS_VLM:-}"
DIMS_THREADS="${DIMS_THREADS:-$(nproc 2>/dev/null || echo 0)}"
DIMS_CHAINS="${DIMS_CHAINS:-16}"
BIND="${BIND:-0.0.0.0}"
export GROUP_ID="${GROUP_ID:-$(date +%Y%m%d-%H%M%S)}"
VERSION="$(cat VERSION 2>/dev/null || echo unknown)"
echo "version:  $VERSION"
echo "group id: $GROUP_ID (stamped on every layer and on every trace id in this run)"

case "$DIMS_READER" in
  tesseract|trocr) ;;
  *) echo "DIMS_READER must be tesseract or trocr, got '$DIMS_READER'" >&2; exit 1 ;;
esac
case "$MODE" in
  closed|seeded) ;;
  *) echo "MODE must be closed or seeded, got '$MODE'" >&2; exit 1 ;;
esac
case "$ENGINE" in
  tesseract|florence|both) ;;
  *) echo "ENGINE must be tesseract, florence or both, got '$ENGINE'" >&2; exit 1 ;;
esac

# What technology each layer will use in this run (printed first so a log is self-explaining)
case "$ENGINE" in
  tesseract) READER="tesseract" ;;
  florence)  READER="florence-2" ;;
  both)      READER="tesseract + florence-2" ;;
esac
if [ "$MODE" = "seeded" ]; then
  L2="ENGINE=$READER"
  if [ -n "$LAYOUT" ]; then
    if [ "$LAYOUT" = "1" ]; then
      LAYOUT_NAME="$(python -c 'import layout_check; print(layout_check.DEFAULT_MODEL)' 2>/dev/null || echo default)"
    else
      LAYOUT_NAME="$LAYOUT"
    fi
    SPLITTER="splitter(yolo)=on  weights=$LAYOUT_NAME  revision=${LAYOUT_REV:-unpinned}"
  else
    SPLITTER="splitter(yolo)=off"
  fi
else
  L2="ENGINE=florence-2 (PDF text layer first if the plan has one)"
  SPLITTER="splitter(yolo)=n/a in closed mode$([ -n "$LAYOUT" ] && echo " (LAYOUT is ignored)")"
fi
OLLAMA_UP=0
OLLAMA_NOTE="ollama=off  (not reachable at $URL, layer 3 will be skipped)"
if TAGS="$(curl -fsS -m 5 "$URL/api/tags" 2>/dev/null)"; then
  OLLAMA_UP=1
  if printf '%s' "$TAGS" | grep -Fq "\"$MODEL\""; then
    OLLAMA_NOTE="ollama=on  model=$MODEL  url=$URL"
  else
    OLLAMA_NOTE="ollama=on  model=$MODEL NOT installed (run: ollama pull $MODEL)  url=$URL"
  fi
fi
echo
echo "== Technology in this run =="
echo "version   $VERSION"
echo "group id  $GROUP_ID"
echo "mode      $MODE"
echo "layer 1   engine room     OpenCV + scikit-image (always)"
echo "layer 2   text reader     $L2"
echo "          $SPLITTER"
if [ -n "$DIMS" ]; then
  if [ -n "$DIMS_VLM" ]; then
    DIMS_NOTE="dimlines=on  cheap=$DIMS_READER  expensive=$DIMS_VLM (threads=$DIMS_THREADS)  chains<=$DIMS_CHAINS per axis"
  else
    DIMS_NOTE="dimlines=on  cheap=$DIMS_READER  expensive=none  chains<=$DIMS_CHAINS per axis"
  fi
else
  DIMS_NOTE="dimlines=off  (set DIMS=1 to read dimension chains)"
fi
echo "layer 2b  dimension lines $DIMS_NOTE"
echo "layer 3   reasoning       $OLLAMA_NOTE"
echo

if [ "${DRY_RUN:-}" = "1" ]; then
  echo "DRY_RUN=1: nothing was run."
  exit 0
fi

if [ ! -f manifest.csv ]; then
  echo "manifest.csv not found. Copy manifest.example.csv and list your plans." >&2
  exit 1
fi

if [ -n "$DIMS" ]; then
  echo "== Layer 2b: dimension chains (dimlines.py) =="
  DIMS_ARGS=(--max-chains "$DIMS_CHAINS" --reader "$DIMS_READER")
  if [ -n "$DIMS_VLM" ]; then
    DIMS_ARGS+=(--expensive ollama --model "$DIMS_VLM" --ollama-url "$URL" --threads "$DIMS_THREADS")
  fi
  python dimlines.py "${DIMS_ARGS[@]}"
  # ANCHOR BEFORE 20261012_033439
  # Which detected lines a believed reading proves, so the rooms layer keeps them out of
  # the wall map. Geometry only, a few seconds. Without it every dimension line stays in
  # the walls and cuts the rooms it runs through, whatever was read on it.
  python dimlines.py --lines-only
  # ANCHOR AFTER 20261012_033439
  echo
fi

if [ "$MODE" = "seeded" ]; then
  echo "== Layers 1 and 2: read room names, grow regions, check printed sizes (seeded.py) =="
  if [ "$ENGINE" = "florence" ] || [ "$ENGINE" = "both" ]; then
    python seeded.py --engine "$ENGINE" --trust-remote-code "${EXTRA[@]}"
  else
    python seeded.py "${EXTRA[@]}"
  fi
else
  echo "== Layer 1: engine room (measure rooms with OpenCV) =="
  python extract.py

  echo
  echo "== Layer 2: text lifter (PDF text layer, else Florence-2 OCR) =="
  python read_labels.py --trust-remote-code
fi

echo
echo "== Layer 3: brain (local language model reads the text) =="
if [ "$OLLAMA_UP" = "1" ]; then
  python interpret.py --model "$MODEL" --url "$URL"
else
  echo "Ollama not reachable at $URL, skipping layer 3."
fi

echo
echo "Done. View results with: python3 -m http.server -d viewer 8765 --bind $BIND"
