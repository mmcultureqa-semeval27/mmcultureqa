# Task 2: Textual Visual QA

Given an image and the question **as text**, produce a short open-ended answer in the language
the question was asked in. Same images and same questions as [Task 1](../task1), with the
speech step taken out.

```text
input   image  +  question text
output  one or two sentences of text
```

## Tracks

`qa_mena_en`, `qa_mena_msa`, `qa_mena_arz`, `qa_mena_ajp` — English, Modern Standard Arabic,
Egyptian Arabic, Levantine Arabic, the tracks released so far. More languages and regions
follow as their own tracks. Every item appears in all four varieties under the same `id`, so
the tracks are comparable. Systems are ranked per track; enter any subset.

## Data

```python
from datasets import load_dataset
qa = load_dataset("QCRI/MMCQA-SemEval27", "qa_mena_en", split="dev")
# {'id': ..., 'image': 'images/<id>.jpg', 'question': '...', 'answer': '...'}
```

train and dev carry `answer`; devtest does not, since the development phase is scored on it.
See [../data](../data) for the image archive.

## Baseline

[`../baselines/baseline_qwen_omni.py`](../baselines) runs Qwen2.5-Omni in 4-bit on the image
plus the question text:

```bash
python ../baselines/baseline_qwen_omni.py --task qa --track en --split devtest \
    --media-root ~/mmcqa --out prediction.jsonl
```

## Submit

```json
{"id": "<id>", "answers": {"en": "..."}}
```

```bash
python ../format_checker/check_format.py --pred prediction.jsonl \
    --ids ~/mmcqa/rows/qa_mena_devtest_en.parquet
```

The official scorer follows shortly; see the repository README.

Ranked by **BERTScore-F1**; BLEU, ROUGE-L and Coverage are reported alongside.
