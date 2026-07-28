"""Internal utilities for PH-Neuro example scripts."""

from __future__ import annotations


def print_header(title: str) -> None:
    """Print a section header with decorative border."""
    width = min(len(title) + 4, 80)
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
