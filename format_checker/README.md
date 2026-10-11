# Format Checker

Run this before you submit. It reads and validates your file with the leaderboard's own code
(`submission_io.py`, copied unchanged from the evaluation script, with no metric code), so an
error here is exactly the error the platform gives you.

```bash
# the shape of the file
python check_format.py --pred prediction.jsonl --lang en

# and that your ids are exactly the ids of the split you answered
python check_format.py --pred prediction.jsonl --lang en --ids devtest_en.parquet
```

Accepts `prediction.jsonl`, a `.zip` containing it, or a `.json` array.

## The format

A submission covers **one track**: one line per item of the split.

```json
{"id": "<id>", "answer": "..."}
```

On the platform the track comes from the `Language` field of the upload form. A `lang` key on
every line works as well, and is what the checker falls back to when you omit `--lang`.

## What it checks

- every line is a JSON object with an `id` and an `answer`
- the track is known and determined, from `--lang` or from the file
- no duplicate id
- with `--ids`: your ids are exactly the ids of the split, none missing, none unknown
- it warns about empty answers, which score zero, and about answers over 5000 characters,
  which are cut before scoring

```text
$ python check_format.py --pred prediction.jsonl --lang en --ids devtest_en.parquet
file   : prediction.jsonl
rows   : 1000
track  : en
answers: 1000 ids, 998 answered, 2 empty
WARNING: 2 empty answers, each scores zero
ids    : match the split exactly (1000)

format OK
```
