"""
Quick check for the raw data loader.

Run from repository root:
    python scripts/check_data_loader.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import build_recording_index


def main():
    raw_data_dir = PROJECT_ROOT / "data" / "raw"

    recording_index = build_recording_index(raw_data_dir)

    print("Found recordings:")
    print(recording_index)

    print()
    print(f"Number of recordings found: {len(recording_index)}")


if __name__ == "__main__":
    main()