"""
MMCultureQA Format Checker (Task 1 and Task 2)
==============================================
Run this before you submit. It reads and validates your file with the
leaderboard's own code (``submission_io.py``, copied from the evaluation
script), so an error here is the error the platform gives you.

A submission covers **one track**: one line per item of the split, with the
answer in the language of that track. On the platform you pick the track in the
``Language`` field of the upload form.

    {"id": "<id>", "answer": "..."}

A ``lang`` key on every line works too, and is what the checker reads when you
do not pass ``--lang``.

Usage
-----
    # shape of the file
    python check_format.py --pred prediction.jsonl --lang en

    # and that your ids are exactly the ids of the split you answered
    python check_format.py --pred prediction.jsonl --lang en --ids devtest_en.parquet

Accepts ``prediction.jsonl``, a ``.zip`` containing it, or a ``.json`` array.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import submission_io as backbone  # noqa: E402  the leaderboard's own reader and parser


def ids_from(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            sys.exit(f"ERROR: no such file: {path}")
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
        else:
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines()
                    if l.strip()]
        ids |= {str(r["id"]) for r in rows}
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", required=True, type=Path,
                        help="prediction.jsonl, prediction.zip, or a .json array")
    parser.add_argument("--lang", help="Track code of this submission, e.g. en, msa, arz, ajp")
    parser.add_argument("--ids", nargs="*", type=Path, default=[],
                        help="The released split file whose ids the submission must match")
    args = parser.parse_args()
    if not args.pred.exists():
        sys.exit(f"ERROR: no such file: {args.pred}")

    try:
        rows, name = backbone.read_submission(str(args.pred))
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"file   : {name}")
    print(f"rows   : {len(rows)}")

    wanted = ids_from(args.ids) if args.ids else {str(r.get("id", "")) for r in rows}
    try:
        # exactly what the platform calls: the track comes from the form, or from
        # a lang key in the file; the ids must match the split
        lang, preds = backbone.parse_predictions(rows, wanted, args.lang, name)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    answered = sum(1 for a in preds.values() if a.strip())
    print(f"track  : {lang}")
    print(f"answers: {len(preds)} ids, {answered} answered, {len(preds) - answered} empty")
    if answered < len(preds):
        print(f"WARNING: {len(preds) - answered} empty answers, each scores zero")
    long = [i for i, a in preds.items() if len(a) > backbone.MAX_CHARS]
    if long:
        print(f"WARNING: {len(long)} answers are longer than {backbone.MAX_CHARS} characters "
              "and will be cut before scoring")
    if args.ids:
        print(f"ids    : match the split exactly ({len(wanted)})")
    else:
        print("ids    : not checked, pass --ids <split file> to check them")
    print("\nformat OK")


if __name__ == "__main__":
    main()
