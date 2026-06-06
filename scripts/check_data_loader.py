"""
Small test script for our data loader.

We use this script to check whether our raw data folder is structured correctly
and whether the loader can find the Sensor Logger ZIP files.

Run this script from the repository root:

    python scripts/check_data_loader.py

At the moment, it is also okay if it finds 0 recordings, because we may not have
added the raw ZIP files to the repository yet. Once the recordings are placed in
data/raw/, this script should list them with label, subtype and person.
"""

from pathlib import Path
import sys

#This makes sure that Python can import files from the src folder when we run this script from the repository root.
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