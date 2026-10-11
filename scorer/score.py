"""
MMCultureQA Official Scorer (Task 1 Spoken VQA / Task 2 Textual VQA)
====================================================================
Both tasks answer in free text and are scored the same way, so one scorer
serves both. The scoring is ``backbone.py``, the evaluation script the
leaderboard runs, bundled here unmodified: what you get locally is what the
leaderboard reports.

Score yourself on a labelled split — train or dev. devtest and test ship
without answers.

Usage
-----
    # the released dev rows are the gold; the track comes from the file name
    python score.py --gold dev_en.parquet --pred prediction.jsonl

    # a file whose name does not say the track
    python score.py --gold my_dev.jsonl --pred prediction.jsonl --lang arz

    # compare with the older encoder (the ranking stays on mmBERT)
    python score.py --gold dev_msa.parquet --pred prediction.jsonl --backbone mbert

One submission covers one track, as on the platform, where the track is picked
in the ``Language`` field of the upload form.

Metrics
    BERTScore-F1/P/R   jhu-clsp/mmBERT-base, layer 17.  **F1 is the ranking metric.**
    BLEU               sacreBLEU, tokenize=flores200 (script-agnostic)
    ROUGE-L            ROUGE-L F with a Unicode word tokenizer, mean per item
    Coverage           share of items you answered

An unanswered item scores zero, so a poor answer still beats no answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backbone  # noqa: E402  the leaderboard's evaluation script, unmodified

# BERTScore encoders. mmBERT is what the leaderboard ranks on; mBERT is the
# older multilingual encoder, kept for comparison.
BACKBONES = {
    "mmbert": ("jhu-clsp/mmBERT-base", 17, True),
    "mbert": ("bert-base-multilingual-cased", 9, False),
}


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    if not rows:
        sys.exit(f"{path}: file is empty")
    return rows


def build_gold(paths: list[Path], override: str | None) -> tuple[list[dict], str]:
    """The released rows, in the gold shape the backbone reads: answer per track."""
    gold: dict[str, dict] = {}
    tracks: set[str] = set()
    for path in paths:
        rows = read_rows(path)
        if isinstance(rows[0].get("answer"), dict):          # already a gold file
            for r in rows:
                entry = gold.setdefault(str(r["id"]), {"id": str(r["id"]), "answer": {}})
                entry["answer"].update(r["answer"])
                for key in ("country", "category", "subcategory"):
                    if r.get(key):
                        entry[key] = r[key]
                tracks.update(r["answer"])
            continue
        track = override or path.stem.rsplit("_", 1)[-1].lower()   # dev_msa.parquet -> msa
        if not track or track in ("dev", "train", "devtest", "test"):
            sys.exit(f"{path.name}: cannot tell the track from the file name; pass --lang")
        blank = [r for r in rows if not str(r.get("answer") or "").strip()]
        if blank:
            sys.exit(f"{path.name}: {len(blank)} rows have no answer. devtest and test are released "
                     "without answers; score yourself on train or dev.")
        for r in rows:
            entry = gold.setdefault(str(r["id"]), {"id": str(r["id"]), "answer": {}})
            entry["answer"][track] = str(r["answer"])
        tracks.add(track)
    if len(tracks) > 1:
        sys.exit(f"one submission covers one track; these gold files mix {sorted(tracks)}")
    track = tracks.pop()
    print(f"[score] gold: {len(gold)} items, track {track!r}")
    return list(gold.values()), track


def use_backbone(name: str, num_layers: int | None) -> None:
    """Swap the encoder without touching the bundled leaderboard script."""
    if name == "mmbert" and not num_layers:
        return                                   # the default inside backbone.py
    model, layers, fast = BACKBONES.get(name, (name, num_layers, True))
    layers = num_layers or layers
    if not layers:
        sys.exit(f"--num-layers is required for the custom encoder {model!r}")
    import torch
    from bert_score import BERTScorer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BERTScorer(model_type=model, num_layers=layers, batch_size=backbone.BATCH_SIZE,
                        nthreads=4, idf=False, device=device, lang=None,
                        rescale_with_baseline=False, use_fast_tokenizer=fast)
    scorer._tokenizer.model_max_length = 512
    backbone._SCORER = scorer                    # get_scorer() hands it back as is
    print(f"[score] encoder: {model} layer {layers} on {device} "
          f"(the leaderboard ranks on {BACKBONES['mmbert'][0]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", nargs="+", required=True, type=Path,
                        help="Released rows with answers for one track")
    parser.add_argument("--pred", required=True, type=Path, help="prediction.jsonl or .zip")
    parser.add_argument("--lang", help="Track code, if the gold file name does not say")
    parser.add_argument("--phase", default="dev", help="Phase name, only used in the log")
    parser.add_argument("--backbone", default="mmbert",
                        help=f"{' / '.join(BACKBONES)} or any HF encoder (default: mmbert)")
    parser.add_argument("--num-layers", type=int, help="Encoder layer, for a custom encoder")
    parser.add_argument("--out", type=Path, help="Write the full result as JSON")
    args = parser.parse_args()

    gold, track = build_gold(args.gold, args.lang)
    use_backbone(args.backbone, args.num_layers)

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for row in gold:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        gold_path = fh.name
    try:
        # the platform takes the track from the upload form; locally we pass it the same way
        result = backbone.evaluate(gold_path, str(args.pred), args.phase,
                                   submission_metadata=[{"name": "Language", "value": track}])
    except ValueError as exc:               # the message the leaderboard would show
        sys.exit(f"\nERROR: {exc}")
    finally:
        os.unlink(gold_path)

    print()
    for block in result["result"]:
        for name, scores in block.items():
            print(name)
            width = max(len(k) for k in scores)
            for key, value in scores.items():
                print(f"  {key:<{width}}  {value}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
