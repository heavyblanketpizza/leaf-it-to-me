"""One model label means both the crop/tree category and the leaf condition."""

import re


TREE_CONDITIONS = {
    "apple": ["healthy", "scab", "black_rot", "cedar_apple_rust", "multiple_diseases"],
    "cherry": ["healthy", "powdery_mildew"],
    "peach": ["healthy", "bacterial_spot"],
    "orange": ["citrus_greening"],
    "mango": [
        "healthy", "anthracnose", "bacterial_canker", "cutting_weevil",
        "dieback", "gall_midge", "powdery_mildew", "sooty_mould",
    ],
}

LABELS = [
    {"index": index, "label": label, "tree_type": tree, "condition": condition}
    for index, (label, tree, condition) in enumerate(sorted(
        (f"{tree}_{condition}", tree, condition)
        for tree, conditions in TREE_CONDITIONS.items() for condition in conditions
    ))
]
BY_LABEL = {entry["label"]: entry for entry in LABELS}

# Spellings below are the release's actual names, including its "Haunglongbing" typo.
PLANTVILLAGE_LABELS = {
    "Apple___Apple_scab": "apple_scab",
    "Apple___Black_rot": "apple_black_rot",
    "Apple___Cedar_apple_rust": "apple_cedar_apple_rust",
    "Apple___healthy": "apple_healthy",
    "Cherry_(including_sour)___Powdery_mildew": "cherry_powdery_mildew",
    "Cherry_(including_sour)___healthy": "cherry_healthy",
    "Peach___Bacterial_spot": "peach_bacterial_spot",
    "Peach___healthy": "peach_healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "orange_citrus_greening",
}
CORNELL_LABELS = {
    "healthy": "apple_healthy", "scab": "apple_scab",
    "rust": "apple_cedar_apple_rust", "multiple_diseases": "apple_multiple_diseases",
}
MANGO_LABELS = {
    "Anthracnose": "mango_anthracnose", "Bacterial Canker": "mango_bacterial_canker",
    "Cutting Weevil": "mango_cutting_weevil", "Die Back": "mango_dieback",
    "Gall Midge": "mango_gall_midge", "Healthy": "mango_healthy",
    "Powdery Mildew": "mango_powdery_mildew", "Sooty Mould": "mango_sooty_mould",
}


def normalize_name(name):
    """Ignore spaces, punctuation, and capitalization in imported folder names."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def label_mapping():
    return [entry.copy() for entry in LABELS]
