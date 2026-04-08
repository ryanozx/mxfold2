#!/usr/bin/env python3

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mxfold2.__main__ import main


if __name__ == "__main__":
    sys.argv.insert(1, "pair_fusion")
    main()
