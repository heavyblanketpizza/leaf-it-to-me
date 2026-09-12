"""Fetch just the requested original PlantVillage color files from its authors."""
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

HF_REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
GITHUB_REVISION = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
HF = f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{HF_REVISION}"
GH = f"https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/{GITHUB_REVISION}"


def fetch(url):
    request = Request(url, headers={"User-Agent": "leaf-it-to-me-weekend-experiment/1"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=90) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def download(args):
    if args.limit_per_class < 0 or args.workers < 1:
        raise ValueError("limit-per-class must be >= 0 and workers must be >= 1")
    from .labels import PLANTVILLAGE_LABELS
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = {}
    paths = set()
    for name in ("color_train.txt", "color_test.txt"):
        body = fetch(f"{HF}/splits/{name}")
        metadata[name] = hashlib.sha256(body).hexdigest()
        (root / "splits").mkdir(exist_ok=True)
        (root / "splits" / name).write_bytes(body)
        paths.update(body.decode().splitlines())
    # The source's old 80/20 membership is not used for our new grouped 80/10/10.
    selected = []
    counts = Counter()
    available = Counter()
    for name in sorted(paths):
        parts = Path(name).parts
        if len(parts) != 4 or parts[:2] != ("raw", "color"):
            continue
        category = parts[2]
        if category not in PLANTVILLAGE_LABELS:
            continue
        available[category] += 1
        if not args.limit_per_class or counts[category] < args.limit_per_class:
            selected.append(name)
            counts[category] += 1
    if len(available) != 9 or sum(available.values()) != 13241:
        raise ValueError(f"Unexpected pinned source inventory; inspect before using: {dict(available)}")
    leaf_map = root / "leaf_grouping" / "leaf-map.json"
    leaf_map.parent.mkdir(exist_ok=True)
    leaf_bytes = fetch(f"{HF}/leaf_grouping/leaf-map.json")
    leaf_map.write_bytes(leaf_bytes)
    metadata["leaf-map.json"] = hashlib.sha256(leaf_bytes).hexdigest()
    print(f"Official inventory: {len(paths):,} color paths; tree subset: {sum(available.values()):,}", flush=True)
    print(f"Fetching {len(selected):,} selected originals to {root}", flush=True)

    def get_one(name):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            body = fetch(f"{GH}/{quote(name, safe='/')}")
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(body)
            temporary.replace(destination)
        return {"path": name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}

    # A failed request raises an error. Rerunning resumes without replacing originals.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        downloaded = []
        for item in pool.map(get_one, selected):
            downloaded.append(item)
            if len(downloaded) % 100 == 0:
                print(f"Checked/downloaded {len(downloaded)}/{len(selected)}", flush=True)
    receipt = {"github_revision": GITHUB_REVISION, "hf_revision": HF_REVISION,
               "source": GH, "metadata_sha256": metadata,
               "official_total_color_paths": len(paths), "official_tree_counts": dict(available),
               "selected_count": len(selected), "selected_counts": dict(counts), "files": downloaded}
    (root / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"downloaded_or_already_present": len(downloaded), "counts": dict(counts)}, indent=2))


def add_commands(subparsers):
    parser = subparsers.add_parser("download-plantvillage", help="Download original tree color images + leaf IDs")
    parser.add_argument("--output", default="data/raw/plantvillage")
    parser.add_argument("--limit-per-class", type=int, default=0, help="0 = all; 6 = tiny real-data trial")
    parser.add_argument("--workers", type=int, default=4)
    parser.set_defaults(func=download)
