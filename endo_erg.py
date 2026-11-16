"""Entry point for the portable endoscopist ergonomics analysis package."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    from src.cli import main

    raise SystemExit(main())
