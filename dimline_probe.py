"""Probe v2: find dimension chains, crop each segment, read it with a chosen reader.

Tesseract cannot read this hand lettering; the crops are fine. This version lets a
handwriting model read the same crops so the two can be compared directly.

Standalone. Nothing from the repo, writes nothing but PNGs into --out.

    python dimline_probe.py plans/Putnam_House_-_floor_plans.jpg 74.94

It finds long thin horizontal lines with at least three extension-line ticks,
then crops each segment and runs Tesseract on it with several settings.
Print the output and send it back.
"""
import argparse, os, shlex, sys, time
import cv2, numpy as np, pytesseract

CFG = '--psm %d -c ' + shlex.quote("tessedit_char_whitelist=0123456789'\"")

# No worked example here, deliberately. An earlier version of this prompt offered
# 7'10 1/2" as an illustration and the model returned that same string for three
# different crops: a small vision model will copy an example out of the prompt
# rather than admit it cannot read the image. Describe the format, never show it.
PROMPT = (
    'Transcribe the handwritten text in this image, character for character. '
    'It is one architectural dimension: a number, a foot mark, then inches, '
    'sometimes with a fraction. Copy only what you can actually see. '
    'Do not guess, do not complete a partial number, do not repeat an example. '
    'Reply with the transcription alone and nothing else. '
    'If the image has no readable text, reply with a single question mark.'
)


def thin(ink, orient, min_px, max_thick):
    k = (1, min_px) if orient == 'v' else (min_px, 1)
    m = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if orient == 'v' and w <= max_thick:
            out.append((x + w // 2, y, y + h))
        if orient == 'h' and h <= max_thick:
            out.append((y + h // 2, x, x + w))
    return out


def find_chains(gray, ppf):
    ink = (gray < 150).astype(np.uint8)
    thick = max(3, int(0.25 * ppf))
    ticks = thin(ink, 'v', max(8, int(0.9 * ppf)), thick)
    lines = thin(ink, 'h', int(8 * ppf), thick)
    chains = []
    for y, x0, x1 in lines:
        near = sorted(x for x, a, b in ticks
                      if x0 - thick * 2 <= x <= x1 + thick * 2
                      and a - thick * 3 <= y <= b + thick * 3
                      and (b - a) <= int(12 * ppf))
        merged = []
        for x in near:
            if merged and x - merged[-1][-1] <= thick * 2:
                merged[-1].append(x)
            else:
                merged.append([x])
        merged = [int(np.mean(g)) for g in merged]
        if len(merged) >= 3:
            chains.append({'y': int(y), 'x0': int(x0), 'x1': int(x1), 'ticks': merged})
    return chains


def erase_rules(crop, ppf):
    ink = (crop < 160).astype(np.uint8)
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (max(6, int(1.2 * ppf)), 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, crop.shape[0])))
    out = crop.copy()
    out[cv2.dilate(h | v, np.ones((2, 2), np.uint8)) > 0] = 255
    return out


def prep(crop, target_h=64):
    f = max(1.0, target_h / max(1, crop.shape[0]))
    c = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    c = cv2.threshold(cv2.GaussianBlur(c, (3, 3), 0), 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.copyMakeBorder(c, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)


class Reader:
    """Whatever reads one cropped segment. Loaded lazily so tesseract needs no extra install."""

    def __init__(self, kind, model=None, url=None, timeout=600, threads=0):
        self.kind, self.name = kind, model
        if kind == 'trocr':
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.name = model or 'microsoft/trocr-base-handwritten'
            self.proc = TrOCRProcessor.from_pretrained(self.name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.name)
            self.model.eval()
        elif kind == 'ollama':
            self.name = model or 'qwen2.5vl:3b'
            self.url = (url or 'http://localhost:11434').rstrip('/')
            self.timeout = timeout
            self.threads = threads
            # Load the model once, before the loop. On CPU the first call also pays
            # for reading several GB off disk, and that alone can blow the timeout.
            t0 = time.time()
            self._ask('Reply with: ok', None, warm=True)
            print('ollama warm-up %.1f s' % (time.time() - t0))

    def __call__(self, bgr_or_gray):
        if self.kind == 'tesseract':
            out = []
            for psm in (7, 8, 13):
                t = pytesseract.image_to_string(bgr_or_gray, config=CFG % psm).strip()
                t = t.replace('\n', ' ').strip()
                if t and t not in out:
                    out.append(t)
            return ' | '.join(out)
        if self.kind == 'trocr':
            from PIL import Image
            im = Image.fromarray(cv2.cvtColor(bgr_or_gray, cv2.COLOR_GRAY2RGB))
            px = self.proc(images=im, return_tensors='pt').pixel_values
            ids = self.model.generate(px, max_new_tokens=16)
            return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if self.kind == 'ollama':
            # Every pixel becomes vision tokens, so a wide crop is slow for no gain.
            img = bgr_or_gray
            if img.shape[1] > 512:
                f = 512.0 / img.shape[1]
                img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            return self._ask(PROMPT, img)
        raise ValueError(self.kind)

    def _ask(self, prompt, img, warm=False):
        import base64, json, urllib.request
        opts = {'temperature': 0, 'num_predict': 24}
        if self.threads:
            # More worker threads than cores makes the kernel shuffle them instead of
            # computing: the run queue climbs and system time eats the gain.
            opts['num_thread'] = self.threads
        payload = {'model': self.name, 'stream': False, 'prompt': prompt,
                   'keep_alive': '15m', 'options': opts}
        if img is not None:
            payload['images'] = [base64.b64encode(cv2.imencode('.png', img)[1]).decode()]
        req = urllib.request.Request(self.url + '/api/generate', json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r).get('response', '').strip().replace('\n', ' ')
        except Exception as e:
            if warm:
                raise SystemExit('ollama did not answer at %s: %s\n'
                                 'Is it running, and is the model pulled?' % (self.url, e))
            return '(timeout)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('ppf', type=float, help='pixels per foot for this image')
    ap.add_argument('--out', default='probe_out')
    ap.add_argument('--max-chains', type=int, default=6)
    ap.add_argument('--reader', default='tesseract',
                    choices=('tesseract', 'trocr', 'ollama'))
    ap.add_argument('--reader-model', default=None)
    ap.add_argument('--ollama-url', default='http://localhost:11434')
    ap.add_argument('--min-ticks', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=600, help='seconds per reader call')
    ap.add_argument('--max-segments', type=int, default=0,
                    help='stop after this many crops; use a small number to time the reader first')
    ap.add_argument('--one-pass', action='store_true',
                    help='read only the line-erased crop, not both versions')
    ap.add_argument('--threads', type=int, default=0,
                    help='worker threads for the ollama reader; set this to nproc')
    ap.add_argument('--only', default='',
                    help='read just these crops, as chain:segment pairs, e.g. 1:4,1:6 '
                         '(chain:* for a whole chain). Everything else is still detected '
                         'and measured, only the reader is skipped.')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gray = cv2.imread(a.image, 0)
    if gray is None:
        sys.exit('cannot read ' + a.image)
    ppf = a.ppf
    print('image %dx%d  ppf %.2f  sheet about %.0f x %.0f ft'
          % (gray.shape[1], gray.shape[0], ppf, gray.shape[1] / ppf, gray.shape[0] / ppf))

    chains = find_chains(gray, ppf)
    # Two lines can span the same distance: the segmented chain and the overall below it.
    # Keep the one with more ticks, since the overall carries only its two arrowheads.
    chains = [c for c in chains if len(c['ticks']) >= a.min_ticks]
    chains.sort(key=lambda c: (-(c['ticks'][-1] - c['ticks'][0]), -len(c['ticks'])))
    kept = []
    for c in chains:
        span = (c['ticks'][-1] - c['ticks'][0]) / ppf
        dup = next((k for k in kept
                    if abs((k['ticks'][-1] - k['ticks'][0]) / ppf - span) < 1.0
                    and abs(k['y'] - c['y']) < 4 * ppf), None)
        if dup is None:
            kept.append(c)
        elif len(c['ticks']) > len(dup['ticks']):
            kept[kept.index(dup)] = c
    kept.sort(key=lambda c: -(c['ticks'][-1] - c['ticks'][0]))
    chains = kept
    want = set()
    for part in filter(None, (p.strip() for p in a.only.split(','))):
        ci, _, si = part.partition(':')
        want.add((int(ci), None if si.strip() == '*' else int(si)))
    reader = Reader(a.reader, a.reader_model, a.ollama_url, a.timeout, a.threads)
    print('reader %s%s' % (a.reader, ' ' + reader.name if reader.name else ''))
    print('%d chain candidates, showing the %d longest\n' % (len(chains), a.max_chains))
    t_start = time.time()

    band = max(10, int(0.8 * ppf))
    done = 0
    for ci, c in enumerate(chains[:a.max_chains]):
        span_ft = (c['ticks'][-1] - c['ticks'][0]) / ppf
        print('chain %d  y=%d  x %d..%d  ticks=%d  span=%.2f ft'
              % (ci, c['y'], c['x0'], c['x1'], len(c['ticks']), span_ft))
        y0, y1 = max(0, c['y'] - band), min(gray.shape[0], c['y'] + band)
        for si, (p, q) in enumerate(zip(c['ticks'], c['ticks'][1:])):
            ins = max(2, int((q - p) * 0.06))
            if q - p - 2 * ins < 6:
                continue
            if want and (ci, si) not in want and (ci, None) not in want:
                print('  seg%02d  %5.2f ft measured  (not selected)' % (si, (q - p) / ppf))
                continue
            if a.max_segments and done >= a.max_segments:
                print('  (stopped after %d crops)' % done)
                break
            raw = gray[y0:y1, p + ins:q - ins]
            passes = [('clean', erase_rules(raw, ppf))]
            if not a.one_pass:
                passes.insert(0, ('raw', raw))
            got = []
            t0 = time.time()
            for tag, img in passes:
                t = reader(prep(img))
                if t and t != '?':
                    got.append('%s=%r' % (tag, t))
            done += 1
            cv2.imwrite(os.path.join(a.out, 'c%d_s%02d.png' % (ci, si)),
                        prep(erase_rules(raw, ppf)))
            print('  seg%02d  %5.2f ft measured  %5.1fs  %s'
                  % (si, (q - p) / ppf, time.time() - t0,
                     '  '.join(got) if got else '(nothing read)'))
        print()
    print('%.1f s total' % (time.time() - t_start))
    print('crops written to %s/ - open a few and check they are tight on the numbers' % a.out)


if __name__ == '__main__':
    main()
