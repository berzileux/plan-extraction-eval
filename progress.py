"""A one-line progress report for work that takes minutes.

Both the dimension layer and the language-model layer spend seconds per item and
can sit silent for half an hour. Rewriting one line keeps a terminal readable;
piped to a file there is no cursor to rewind, so it prints a new line now and then
instead. The estimate is the average so far, which is honest for this work because
every item costs about the same.
"""
import sys
import time


class Progress:
    def __init__(self, total, label, every=10, stream=None, quiet=False):
        self.total, self.label = max(1, total), label
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.every, self.done, self.t0 = every, 0, time.time()
        self.quiet = quiet
        self._show()

    @staticmethod
    def _clock(sec):
        sec = int(max(0, sec))
        return "%d:%02d:%02d" % (sec // 3600, sec % 3600 // 60, sec % 60) if sec >= 3600 \
            else "%d:%02d" % (sec // 60, sec % 60)

    def _show(self, end=False):
        if self.quiet:
            return
        gone = time.time() - self.t0
        line = "  %s  %d/%d  %3.0f%%  elapsed %s" % (
            self.label, self.done, self.total, 100.0 * self.done / self.total,
            self._clock(gone))
        if self.done and not end:
            line += "  ETA %s" % self._clock(gone / self.done * (self.total - self.done))
        if end:
            line += "  done"
        if self.tty:
            self.stream.write("\r\033[K" + line)
            if end:
                self.stream.write("\n")
            self.stream.flush()
        elif end or self.done % self.every == 0:
            self.stream.write(line + "\n")
            self.stream.flush()

    def tick(self, n=1):
        self.done += n
        self._show()

    def finish(self):
        self.done = self.total
        self._show(end=True)
        return time.time() - self.t0
