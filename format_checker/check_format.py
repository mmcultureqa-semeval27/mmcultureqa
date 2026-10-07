"""
MMCultureQA Format Checker (Task 1 and Task 2)
==============================================
Run this before you submit. The checks are the submission checks themselves:
the file is read and parsed by ``scorer/backbone.py``, the evaluation script the
leaderboard runs, so an error here is the error you would get on the platform.

Usage
-----
    # shape of the file only
    python check_format.py --pred prediction.jsonl

    # also check that your ids are exactly the ids of the split you answered
    python check_format.py --pred prediction.jsonl --ids ../data/devtest_en.parquet

    # one file answering several tracks
    python check_format.py --pred prediction.jsonl \
        --ids ../data/devtest_en.parquet ../data/devtest_msa.parquet

Submission format: one JSON object per line in ``prediction.jsonl``, nested,

    {"id": "<id>", "answers": {"en": "...", "msa": "..."}}

or flat, one line per item and track,

    {"id": "<id>", "lang": "en", "answer": "..."}

Both tasks use the same format; the answer is always text. A ``.zip`` holding
``prediction.jsonl`` is accepted too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import submission_io as backbone  # noqa: E402  the leaderboard's own reader and parser

LANGS = ["en", "msa", "arz", "ajp"]


def ids_from(paths: list[Path]) -> dict[str, set[str]]:
    """{track: ids} from released split files."""
    wanted: dict[str, set[str]] = {}
    for path in paths:
        if not path.exists():
            sys.exit(f"ERROR: no such file: {path}")
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
        else:
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines()
                    if l.strip()]
        tail = path.stem.rsplit("_", 1)[-1].lower()
        if tail not in LANGS:
            sys.exit(f"ERROR: cannot tell the track from {path.name}; "
                     "expected a name like devtest_en.parquet")
        wanted.setdefault(tail, set()).update(str(r["id"]) for r in rows)
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", required=True, type=Path,
                        help="prediction.jsonl, prediction.zip, or a .json array")
    parser.add_argument("--ids", nargs="*", type=Path, default=[],
                        help="Released split files whose ids the submission must match")
    parser.add_argument("--langs", default=",".join(LANGS), help="Allowed track codes")
    args = parser.parse_args()
    if not args.pred.exists():
        sys.exit(f"ERROR: no such file: {args.pred}")
    langs = [l.strip().lower() for l in args.langs.split(",") if l.strip()]

    try:
        rows, name = backbone.read_submission(str(args.pred))
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"file   : {name}")
    print(f"rows   : {len(rows)}")

    wanted = ids_from(args.ids) if args.ids else {}
    # parse_predictions does the real validation: shape, unknown tracks, duplicates,
    # and the id match. Without --ids there is nothing to match against, so the ids
    # of the file stand in and only the other checks bite.
    gold_ids = set().union(*wanted.values()) if wanted else {str(r.get("id", "")) for r in rows}
    try:
        preds = backbone.parse_predictions(rows, gold_ids, langs, name)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    tracks = sorted({l for answers in preds.values() for l in answers}, key=langs.index)
    print(f"tracks : {', '.join(tracks) or 'none'}")
    problems = []
    for track in tracks:
        answers = {i: a for i, a in ((i, p.get(track)) for i, p in preds.items()) if a is not None}
        filled = sum(1 for a in answers.values() if a.strip())
        print(f"  {track:<4} {len(answers):>6} ids, {filled} answered, {len(answers)-filled} empty")
        if filled < len(answers):
            print(f"  WARNING: {track}: {len(answers)-filled} empty answers, each scores zero")
        if wanted.get(track):
            missing = wanted[track] - set(answers)
            if missing:
                problems.append(f"{track}: {len(missing)} ids of the split are unanswered "
                                f"(e.g. {sorted(missing)[:3]})")
            else:
                print(f"  {track:<4} ids match the split exactly ({len(wanted[track])})")
    for track in set(wanted) - set(tracks):
        problems.append(f"{track}: the split was given but the submission has no answers for it")

    for problem in problems:
        print(f"ERROR: {problem}")
    if problems:
        sys.exit(1)
    print("\nformat OK" + ("" if args.ids else "  (ids not checked; pass --ids to check them)"))


if __name__ == "__main__":
    main()
