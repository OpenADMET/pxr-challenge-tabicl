"""Build the challenge fit/test resource CSVs from the raw downloads."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# resolve src/ without an installed package (flat, non-installable layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import build_split  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    """Write the resource CSVs and log the resulting split sizes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_split()


if __name__ == "__main__":
    main()
