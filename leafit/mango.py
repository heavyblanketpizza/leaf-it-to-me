"""Choose representative Mango photos without modifying downloaded files.

Capture metadata and the existing visual/group checks identify likely relatives.
They cannot prove which file is an original or recover every physical leaf.
"""

import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image


CURATION_FIELDS = [
    "image_path", "relative_path", "class_label", "sha256_pixels", "capture_id",
    "capture_time", "camera", "family_id", "action", "representative_path", "reason",
]
GROUPING_METHOD = (
    "Existing exact-pixel, rotation/flip-aware perceptual, official and reviewed "
    "groups, supplemented by Mango camera make/model + EXIF DateTimeOriginal "
    "at one-second precision. Connections are transitive and span class labels."
)
REPRESENTATIVE_POLICY = (
    "Keep one Mango photo per final global connected family and class label: "
    "prefer no Photo Editor/Photoshop/GIMP software indicator, then larger pixel "
    "area, then fewer Custom/copy/number suffix markers, then relative path. "
    "Other dataset rows and original files are preserved."
)
LIMITATION = (
    "Approximate representative subset, not verified original images or complete "
    "physical-leaf identities. Copies without detectable relationships may cross "
    "splits; distinct captures in the same camera second or false visual matches "
    "may be grouped together. Evaluation is an exploratory estimate."
)


def _text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return " ".join(str(value or "").replace("\x00", "").split())


def _metadata(path):
    """Read capture evidence; incomplete/invalid dates cannot make a group."""
    try:
        with Image.open(path) as photo:
            exif = photo.getexif()
            capture = exif.get_ifd(34665)
            make, model = _text(exif.get(271)), _text(exif.get(272))
            raw_time = _text(capture.get(36867))
            software = _text(exif.get(305)).casefold()
            area = photo.width * photo.height
    except Exception as error:
        raise ValueError(
            f"Cannot read Mango capture metadata from {path}: {error}. "
            "Check this file before rebuilding the merged dataset."
        ) from error
    capture_time = ""
    # strptime alone accepts single-digit fields; require the standard EXIF form.
    if re.fullmatch(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", raw_time):
        try:
            capture_time = datetime.strptime(raw_time, "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            pass
    capture_key = (make.casefold(), model.casefold(), capture_time) if make and model and capture_time else None
    capture_id = ("capture_" + hashlib.sha256("\n".join(capture_key).encode()).hexdigest()[:16]
                  if capture_key else "")
    return dict(capture_key=capture_key, capture_id=capture_id, capture_time=capture_time,
                camera=" | ".join(value for value in (make, model) if value), area=area,
                edited=any(marker in software for marker in ("photo editor", "photoshop", "gimp")))


def _representative_rank(row, metadata):
    stem = Path(row["relative_path"]).stem
    suffix_markers = len(re.findall(r"\(\s*custom\s*\)|\bcopy\b", stem, flags=re.IGNORECASE))
    # A trailing copy number is useful evidence; the timestamp inside a camera
    # filename is not. Treat one long, all-digit timestamp as a neutral name.
    suffix_markers += len(re.findall(r"(?:[ _-]\d{1,3}|\(\d+\))(?=\s*(?:\([^)]*\)\s*)*$)", stem))
    return metadata["edited"], -metadata["area"], suffix_markers, row["relative_path"], row["image_path"]


def curate_mango(rows, union, output):
    """Return retained rows, remapped global groups, and a JSON-ready audit.

    Call after ``connect_related`` so exact conflicts have already been handled
    and visual or reviewed relations from all datasets are retained.
    """
    # Local import avoids a cycle when preparation calls this helper.
    from .prepare import UnionFind, write_csv

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mango_indexes = [i for i, row in enumerate(rows) if row["source_dataset"] == "mangoleafbd"]
    metadata, capture_members = {}, defaultdict(list)
    for index in mango_indexes:
        metadata[index] = _metadata(rows[index]["image_path"])
        key = metadata[index]["capture_key"]
        if key is not None:
            capture_members[key].append(index)
    for indexes in capture_members.values():
        for index in indexes[1:]:
            union.union(indexes[0], index)

    # Final roots include every source. This matters when a dropped Mango photo
    # forms a bridge between retained relatives from other datasets.
    components = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)
    families, choices = {}, defaultdict(list)
    for index in mango_indexes:
        root = union.find(index)
        choices[(root, rows[index]["class_label"])].append(index)
        if root not in families:
            identifiers = sorted(f"{rows[i]['source_dataset']}:{rows[i]['relative_path']}"
                                 for i in components[root])
            families[root] = "mango_family_" + hashlib.sha256("\n".join(identifiers).encode()).hexdigest()[:16]
    representatives = {key: min(indexes, key=lambda index: _representative_rank(rows[index], metadata[index]))
                       for key, indexes in choices.items()}
    selected = set(representatives.values())
    surviving_indexes = [index for index, row in enumerate(rows)
                         if row["source_dataset"] != "mangoleafbd" or index in selected]
    retained_union, retained_roots = UnionFind(len(surviving_indexes)), {}
    for new_index, old_index in enumerate(surviving_indexes):
        root = union.find(old_index)
        if root in retained_roots:
            retained_union.union(new_index, retained_roots[root])
        else:
            retained_roots[root] = new_index

    audit = []
    for index in mango_indexes:
        row, details = rows[index], metadata[index]
        root = union.find(index)
        representative = representatives[(root, row["class_label"])]
        keep = index == representative
        reason = ("Only Mango photo with this label in the detected family."
                  if len(choices[(root, row["class_label"])]) == 1 else
                  "Selected by the representative policy; original status is unverified."
                  if keep else
                  "Related same-label photo excluded from the manifest; source file preserved.")
        audit.append({**row, "capture_id": details["capture_id"], "capture_time": details["capture_time"],
                      "camera": details["camera"], "family_id": families[root],
                      "action": "keep" if keep else "exclude",
                      "representative_path": rows[representative]["image_path"], "reason": reason})
    audit.sort(key=lambda row: (row["relative_path"], row["image_path"]))
    write_csv(output / "mango_curation.csv", audit, CURATION_FIELDS)

    input_counts = Counter(rows[index]["class_label"] for index in mango_indexes)
    retained_counts = Counter(rows[index]["class_label"] for index in selected)
    summary = {
        "input_images": len(mango_indexes), "retained_images": len(selected),
        "excluded_images": len(mango_indexes) - len(selected),
        "capture_families": len(capture_members),
        "images_without_capture_metadata": sum(metadata[index]["capture_key"] is None for index in mango_indexes),
        "families": len(families),
        "by_class": {label: {"input": count, "retained": retained_counts[label],
                             "excluded": count - retained_counts[label]}
                     for label, count in sorted(input_counts.items())},
        "grouping_method": GROUPING_METHOD, "representative_policy": REPRESENTATIVE_POLICY,
        "limitation": LIMITATION,
    }
    return [rows[index] for index in surviving_indexes], retained_union, summary
