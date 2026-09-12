"""Download/import official MangoLeafBD v1, verify originals, then remove the ZIP.

Example:
    python scripts/download_mangoleafbd.py --output data/raw/mangoleafbd
An already downloaded ZIP can be supplied with --archive. It is preserved unless
--remove-archive is set. A ZIP downloaded by this script is removed after success.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
from zipfile import ZipFile

from PIL import Image, ImageOps


SOURCE = "https://data.mendeley.com/datasets/hxsnvwty3r/1"
DOWNLOAD = "https://data.mendeley.com/public-api/zip/hxsnvwty3r/download/1"
CATEGORIES = {
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--remove-archive", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = args.archive
    downloaded_here = archive is None
    if downloaded_here:
        archive = root / "mangoleafbd-v1.zip.part"
        if archive.exists():
            raise FileExistsError(f"Supply --archive to reuse the existing {archive}")
        request = urllib.request.Request(DOWNLOAD, headers={"User-Agent": "leaf-it-to-me-personal-experiment/1.0"})
        print("Downloading official MangoLeafBD v1 ZIP...", flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("xb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)

    with archive.open("rb") as archive_file:
        archive_sha256 = hashlib.file_digest(archive_file, "sha256").hexdigest()
    archive_bytes = archive.stat().st_size
    records = []
    counts, modes, sizes = Counter(), Counter(), Counter()
    with ZipFile(archive) as zipped:
        entries = [info for info in zipped.infolist() if not info.is_dir()]
        # v1 is exactly 4,000 JPGs, with no leaf-to-copy table or other metadata.
        if len(entries) != 4000:
            raise ValueError(f"Unexpected release contents: {len(entries)} files")
        paths = set()
        for info in entries:
            parts = PurePosixPath(info.filename).parts
            if (len(parts) != 3 or parts[0] != "MangoLeafBD Dataset"
                    or parts[1] not in CATEGORIES or ".." in parts
                    or not parts[2].lower().endswith(".jpg")):
                raise ValueError(f"Unexpected/unsafe ZIP path: {info.filename}")
            relative = Path(parts[1]) / parts[2]
            if relative.as_posix().casefold() in paths:
                raise ValueError(f"Duplicate/colliding ZIP name: {relative}")
            paths.add(relative.as_posix().casefold())
            data = zipped.read(info)  # ZIP CRC checked as each member is read.
            digest = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as picture:
                modes[picture.mode] += 1
                sizes[f"{picture.width}x{picture.height}"] += 1
                ImageOps.exif_transpose(picture).convert("RGB").load()
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                    raise FileExistsError(f"Existing file differs; preserving it: {destination}")
            else:
                with destination.open("xb") as target:
                    target.write(data)
            # Verify the saved copy before deleting the source ZIP.
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise OSError(f"Saved file hash mismatch: {destination}")
            counts[parts[1]] += 1
            records.append({"path": relative.as_posix(), "bytes": len(data), "sha256": digest})
            if len(records) % 500 == 0:
                print(f"Verified {len(records)}/4000 original images", flush=True)
    if dict(counts) != {category: 500 for category in CATEGORIES}:
        raise ValueError(f"Unexpected class distribution: {dict(counts)}")
    receipt = {
        "source_dataset": "mangoleafbd", "source_url": SOURCE,
        "download_url": DOWNLOAD, "version": 1, "doi": "10.17632/hxsnvwty3r.1",
        "license": "CC BY-NC 3.0",
        "downloaded_verified_utc": datetime.now(timezone.utc).isoformat(),
        "archive_bytes": archive_bytes, "archive_sha256": archive_sha256,
        "image_count": len(records), "class_counts": dict(sorted(counts.items())),
        "original_image_bytes": sum(record["bytes"] for record in records),
        "original_modes": dict(modes), "original_dimensions": dict(sorted(sizes.items())),
        "readability_checked": True, "saved_file_sha256_verified": True,
        "zip_member_crc_verified": True, "original_bytes_preserved": True,
        "removed_wrapper_directory": "MangoLeafBD Dataset",
        "leaf_grouping_metadata": None,
        "grouping_note": "Release contains only JPGs. Filenames alone do not verify physical leaf/augmentation families. Keep Mango training-only until reviewed complete leaf groups are provided.",
        "archive_deleted_after_verification": False,
        "files": records,
    }
    receipt_path = root / "download_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if downloaded_here or args.remove_archive:
        archive.unlink()
        receipt["archive_deleted_after_verification"] = True
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
