# Format Checker

Run this before you submit. It reads and parses your file with `scorer/backbone.py`, the
evaluation script the leaderboard runs, so an error here is exactly the error the platform
would give you.

```bash
# the shape of the file
python check_format.py --pred prediction.jsonl

# and that your ids are exactly the ids of the split you answered
python check_format.py --pred prediction.jsonl --ids devtest_en.parquet

# one file answering several tracks
python check_format.py --pred prediction.jsonl --ids devtest_en.parquet devtest_msa.parquet
```

Accepts `prediction.jsonl`, a `.zip` containing it, or a `.json` array.

## What it checks

- every line is a JSON object with an `id`
- rows are either nested (`answers`) or flat (`lang` + `answer`), with known track codes
- no duplicate answer for the same item and track
- with `--ids`: your ids are exactly the ids of the split, none missing, none unknown
- it warns about empty answers, which score zero

```text
$ python check_format.py --pred prediction.jsonl --ids devtest_en.parquet
file   : prediction.jsonl
rows   : 1000
tracks : en
  en     1000 ids, 998 answered, 2 empty
  WARNING: en: 2 empty answers, each scores zero
  en   ids match the split exactly (1000)

format OK
```
