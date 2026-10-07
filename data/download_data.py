"""
Download MMCultureQA data and unpack the media
==============================================
Pulls the row files and the media archives for the tracks you choose, unzips the
archives into the paths the rows point at, and checks that every referenced
image and clip is really there.

Usage
-----
    # everything for one track and split
    python download_data.py --tracks en --splits dev --root ~/mmcqa

    # both tasks, three tracks, dev and devtest
    python download_data.py --tracks en msa arz --splits dev devtest --root ~/mmcqa

    # rows only, no media (they are the big part)
    python download_data.py --tracks ajp --splits train --root ~/mmcqa --no-media

    # Task 2 only: the written question, so no audio is fetched
    python download_data.py --task qa --tracks en --splits dev --root ~/mmcqa

Afterwards ``<root>`` holds

    rows/qa_mena_dev_en.parquet          the rows, as released
    images/<id>.jpg                      what the "image" column points at
    audio/<lang>/<id>.wav                what the "audio" column points at

so joining rows to files is a path join against ``<root>``:

    row["image"] -> os.path.join(root, row["image"])

Needs ``pip install huggingface_hub pyarrow``.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO = "QCRI/MMCQA-SemEval27"
REGION = "mena"
TRACKS = ["en", "msa", "arz", "ajp"]        # more regions and languages are coming
SPLITS = ["train", "dev", "devtest"]


def fetch(path_in_repo: str, root: Path) -> Path:
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(REPO, path_in_repo, repo_type="dataset")
    return Path(cached)


def unzip(archive: Path, root: Path) -> int:
    """Unpack into root, skipping what is already there."""
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        todo = [n for n in names if not (root / n).exists()]
        for name in todo:
            zf.extract(name, root)
    print(f"  {archive.name}: {len(names)} files ({len(todo)} new)")
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="both", choices=["qa", "sqa", "both"],
                        help="qa = Task 2 (written), sqa = Task 1 (spoken), both (default)")
    parser.add_argument("--tracks", nargs="+", default=["en"], choices=TRACKS)
    parser.add_argument("--splits", nargs="+", default=["dev"], choices=SPLITS)
    parser.add_argument("--root", type=Path, default=Path("mmcqa"),
                        help="Where the media is unpacked and the rows are copied")
    parser.add_argument("--no-media", action="store_true", help="Rows only")
    args = parser.parse_args()
    tasks = ["qa", "sqa"] if args.task == "both" else [args.task]
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "rows").mkdir(exist_ok=True)

    print("rows")
    rows_by_file: dict[Path, list[dict]] = {}
    for task in tasks:
        for split in args.splits:
            for track in args.tracks:
                name = f"{task}/{REGION}/{split}_{track}.parquet"
                local = args.root / "rows" / f"{task}_{REGION}_{split}_{track}.parquet"
                if not local.exists():
                    local.write_bytes(fetch(name, args.root).read_bytes())
                import pyarrow.parquet as pq
                rows_by_file[local] = pq.read_table(local).to_pylist()
                fields = list(rows_by_file[local][0])
                print(f"  {local.name}: {len(rows_by_file[local])} rows, {fields}")

    if not args.no_media:
        print("\nmedia")
        for split in args.splits:
            unzip(fetch(f"archives/images_{REGION}_{split}.zip", args.root), args.root)
            if "sqa" in tasks:
                for track in args.tracks:
                    unzip(fetch(f"archives/audio_{REGION}_{track}_{split}.zip", args.root),
                          args.root)

        print("\ncheck")
        for local, rows in rows_by_file.items():
            missing = [r[key] for r in rows for key in ("image", "audio")
                       if key in r and not (args.root / r[key]).is_file()]
            state = "ok" if not missing else f"{len(missing)} FILES MISSING (e.g. {missing[0]})"
            print(f"  {local.name}: {state}")

    print(f"\nroot: {args.root.resolve()}")
    print("use it as: os.path.join(root, row['image'])  /  os.path.join(root, row['audio'])")


if __name__ == "__main__":
    main()
