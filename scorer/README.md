# Scorer

`backbone.py` is the evaluation script the leaderboard runs, bundled here **unmodified**, so a
score you compute locally is the score the leaderboard reports. Do not edit it; `score.py` is a
thin command-line wrapper around it.

Score yourself on a labelled split — train or dev. devtest and test ship without answers.

```bash
# the released dev rows are the gold; the track comes from the file name
python score.py --gold dev_en.parquet --pred prediction.jsonl

# a gold file whose name does not say the track
python score.py --gold my_dev.jsonl --pred prediction.jsonl --lang arz

# write the full result, including the per-country breakdown of F1
python score.py --gold dev_arz.parquet --pred prediction.jsonl --out scores.json
```

One submission covers one track, as on the platform, where the track is picked in the
`Language` field of the upload form.

## Metrics

| metric | what it is |
|---|---|
| **BERTScore-F1** | **the ranking metric**: `jhu-clsp/mmBERT-base` at layer 17, idf off, no rescaling |
| BERTScore-P / R | precision and recall of the same matching |
| BLEU | sacreBLEU corpus BLEU, `tokenize=flores200`, so scripts are handled alike |
| ROUGE-L | ROUGE-L F measure, Unicode word tokenizer, averaged over items |
| Coverage | share of items with a non-empty answer |

An item you do not answer scores zero in every metric, so a poor answer still beats no answer.

## The encoder, exactly

| | |
|---|---|
| model | `jhu-clsp/mmBERT-base` (ModernBERT, 22 layers) |
| revision | `c5955035435e2bf121cde7f3c8863ef52ff35d82` |
| layer | 17 |
| settings | `idf=False`, no baseline rescaling, fast tokenizer, max length 512, batch 64 |

`score.py` fetches that revision and scores against it, which is what the leaderboard loads,
so local numbers and leaderboard numbers come from the same weights. `--no-pin` uses whatever
is current on the Hub instead. The first run downloads about 1 GB.

## Install

The versions are the ones the leaderboard worker installs; torch 2.6 is required because the
mmBERT checkpoint ships only `pytorch_model.bin` and transformers refuses `torch.load` on
older torch.

```bash
pip install -r ../requirements.txt
```

## Alternative encoder

mBERT, the older multilingual encoder, is available for comparison. **The ranking stays on
mmBERT.**

```bash
python score.py --gold dev_msa.parquet --pred prediction.jsonl --backbone mbert
```

Any other encoder works too, with `--backbone <hf-id> --num-layers N`.

## Example

```text
$ python score.py --gold dev_en.parquet --pred prediction.jsonl
[score] gold: 1000 items, track 'en'
[mmcqa-eval] BERTScorer ready: model=jhu-clsp/mmBERT-base layer=17 device=cuda
[mmcqa-eval] RESULT dev/en: F1=0.8518 BLEU=68.84 ROUGE-L=0.7041 coverage=0.95

en
  BERTScore-F1  0.85177
  BERTScore-P   0.856984
  BERTScore-R   0.847327
  BLEU          68.8442
  ROUGE-L       0.704095
  Coverage      0.95
```
