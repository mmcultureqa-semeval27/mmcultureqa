# Baselines

One reference system covers both tasks, because
[Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) takes images, audio and text in the
same conversation:

| task | what the model gets |
|---|---|
| Task 1, Spoken Visual QA (`--task sqa`) | the image and the question as a wav |
| Task 2, Textual Visual QA (`--task qa`) | the image and the question as text |

It is a **reference, not a strong system**: 4-bit weights, greedy decoding, a two-line prompt,
no retrieval and no fine-tuning. Treat it as a working starting point and change everything.

## Run it

```bash
pip install -r requirements.txt

# get the data first (see ../data)
python ../data/download_data.py --tracks en --splits dev --root ~/mmcqa

# Task 2, English, on dev (which has answers, so you can score yourself)
python baseline_qwen_omni.py --task qa --track en --split dev \
    --media-root ~/mmcqa --limit 50

# Task 1, Egyptian Arabic, on devtest, into the same prediction file
python baseline_qwen_omni.py --task sqa --track arz --split devtest \
    --media-root ~/mmcqa --out prediction.jsonl --zip
```

Run it once per track; answers merge into the same `prediction.jsonl`, so one file can carry
all the tracks you enter. `--track` takes any released variety code and `--region` defaults to
`mena`, so new languages and regions need no code change.

## Hardware

`Qwen/Qwen2.5-Omni-3B` in 4-bit NF4 fits a 16 GB card (a free Colab T4). The talker is
disabled, so nothing is spent on speech output.

| flag | effect |
|---|---|
| `--no-quantize` | full precision, needs roughly twice the memory |
| `--model Qwen/Qwen2.5-Omni-7B` | the larger model |
| `--max-new-tokens` | answer length cap, 64 by default |
| `--max-pixels` | image budget; lower it if you run out of memory |
| `--limit N` | first N items, for a smoke test |

`torchvision` is required even though no video is involved: the Omni processor loads
`AutoVideoProcessor` at construction.

## After the run

The format checker and the official scorer follow shortly; they will validate `prediction.jsonl`
and score it against a labelled split with the leaderboard's own metrics.
