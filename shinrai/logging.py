"""Logging / console helpers.

Wraps Rich when available, falls back to plain ``print`` otherwise.
Import the module-level ``log`` singleton everywhere; never instantiate
directly.
"""

from __future__ import annotations

import sys
from typing import Optional


class Logger:
    """Thin wrapper around Rich Console (or plain print)."""

    def __init__(self) -> None:
        try:
            from rich.console import Console
            self._console: Optional[object] = Console()
            self._has_rich = True
        except ImportError:
            self._console = None
            self._has_rich = False

    # ── Styled print helpers ─────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        if self._has_rich:
            self._console.print(msg, style="cyan")  # type: ignore[union-attr]
        else:
            print(msg)

    def success(self, msg: str) -> None:
        if self._has_rich:
            self._console.print(msg, style="bold green")  # type: ignore[union-attr]
        else:
            print(msg)

    def warn(self, msg: str) -> None:
        if self._has_rich:
            self._console.print(msg, style="yellow")  # type: ignore[union-attr]
        else:
            print(f"[WARN] {msg}")

    def error(self, msg: str) -> None:
        if self._has_rich:
            self._console.print(msg, style="bold red")  # type: ignore[union-attr]
        else:
            print(f"[ERROR] {msg}", file=sys.stderr)

    def rule(self, title: str = "") -> None:
        if self._has_rich:
            self._console.rule(title)  # type: ignore[union-attr]
        else:
            width = 60
            print(f"{'─' * ((width - len(title) - 2) // 2)} {title} {'─' * ((width - len(title) - 2) // 2)}")

    def banner(self, text: str) -> None:
        if self._has_rich:
            self._console.print(
                f"\n[bold magenta]{text}[/bold magenta]\n", justify="center"
            )  # type: ignore[union-attr]
        else:
            print(f"\n{'═' * 60}\n  {text}\n{'═' * 60}\n")

    def print(self, msg: str, **kwargs) -> None:
        if self._has_rich:
            self._console.print(msg, **kwargs)  # type: ignore[union-attr]
        else:
            print(msg)


# Module-level singleton — import this everywhere.
log = Logger()
