import sys
from typing import Callable, Optional

from src.main import logger


class Spinner:
    def __init__(self, total: Optional[int] = None):
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.total = total
        self.index = 0

    def spin(self, type_str: str = "chunks", show_index: bool = False) -> None:
        frame = self.frames[self.index % len(self.frames)]
        dots = ("." * (self.index % 3 + 1)).ljust(3)
        if show_index:
            sys.stdout.write(
                f"\r{frame} Processing {dots} ({self.index + 1}/{self.total}) {type_str}" if self.total is not None else f"\r{frame} Processing {dots} {self.index + 1} {type_str}")
        else:
            sys.stdout.write(
                f"\r{frame} Processing {dots} {type_str}" if self.total is not None else f"\r{frame} Processing {dots} {type_str}")
        sys.stdout.flush()
        self.index += 1

    def finish(self, message: str) -> None:
        sys.stdout.write("\r" + " " * 50 + "\r")
        sys.stdout.flush()
        print(f"✔ {message}")


def make_progress_emitter(
    on_progress: Callable[[str], None] | None,
) -> Callable[[str], None]:
    if on_progress is None:
        return lambda _msg: None

    def emit(msg: str) -> None:
        try:
            on_progress(msg)
        except Exception as exc:
            logger.warning("on_progress callback raised: %s", exc)

    return emit
