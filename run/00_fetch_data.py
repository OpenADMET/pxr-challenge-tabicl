"""Fetch the raw PXR challenge CSVs from Hugging Face into data/raw."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import fetch_raw  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    """Download every raw challenge file and log where it landed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = fetch_raw()
    for name, path in paths.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
