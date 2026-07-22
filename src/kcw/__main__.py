"""Allow `python -m src.kcw` to open the pipeline CLI."""

from src.kcw.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
