"""Command line entry point: python -m leafit --help."""
import argparse
import os
from pathlib import Path

# Keep downloaded weights and plotting caches inside this project by default.
os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path(".cache/torch").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))


def main(argv=None):
    from . import download, prepare, model, smoke
    parser = argparse.ArgumentParser(description="Learn to classify tree leaves with MobileNetV4.")
    commands = parser.add_subparsers(dest="command", required=True)
    for module in (download, prepare, model, smoke):
        module.add_commands(commands)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
