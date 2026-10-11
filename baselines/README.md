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

## Colab

[`baseline_qa_colab.ipynb`](./baseline_qa_colab.ipynb) runs the whole thing on a free T4:
pulls the English dev split, answers it, saves `prediction_qa_en.jsonl`, checks the format
and scores itself against the references.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcultureqa-semeval27/mmcultureqa/blob/main/baselines/baseline_qa_colab.ipynb)

Switch `TASK` to `"sqa"` in the config cell for Task 1, `TRACK` for another language, and
`SPLIT` to `devtest` for the submission split. Answers are written as they are produced, so a
dropped runtime costs you one item, and re-running continues where it stopped.

## Run it locally

```bash
pip install -r requirements.txt

# get the data first (see ../data)
python ../data/download_data.py --tracks en --splits dev --root ~/mmcqa

# Task 2, English, on dev (which has answers, so you can score yourself)
python baseline_qwen_omni.py --task qa --track en --split dev \
    --media-root ~/mmcqa --limit 50

# Task 1, Egyptian Arabic, on devtest
python baseline_qwen_omni.py --task sqa --track arz --split devtest \
    --media-root ~/mmcqa --zip        # -> prediction_sqa_arz.jsonl(.zip)
```

A submission covers one track, so run it once per track and submit each file under that
track. `--track` takes any released variety code and `--region` defaults to `mena`, so new
languages and regions need no code change.

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

```bash
python ../format_checker/check_format.py --pred prediction_qa_en.jsonl --lang en \
    --ids ~/mmcqa/rows/qa_mena_devtest_en.parquet

python ../scorer/score.py --gold ~/mmcqa/rows/qa_mena_dev_en.parquet \
    --pred prediction_qa_en.jsonl
```
