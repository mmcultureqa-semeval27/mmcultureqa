# Task 1: Spoken Visual QA

Given an image and the question **as an audio clip**, produce a short open-ended answer in the
language the question was asked in. The system hears the question; it is never given the text.

```text
input   image  +  audio of the question
output  one or two sentences of text
```

The same questions are given as text in [Task 2](../task2), so comparing the two tells you
what the speech step costs your system. You may enter either task or both.

## Tracks

`sqa_mena_en`, `sqa_mena_msa`, `sqa_mena_arz`, `sqa_mena_ajp` — English, Modern Standard
Arabic, Egyptian Arabic, Levantine Arabic, the tracks released so far. More languages and
regions follow as their own tracks. Every item appears in all four varieties under the same
`id`, so the tracks are comparable. Systems are ranked per track; enter any subset.

## Data

```python
from datasets import load_dataset
sqa = load_dataset("QCRI/MMCQA-SemEval27", "sqa_mena_arz", split="devtest")
# {'id': ..., 'image': 'images/<id>.jpg', 'audio': 'audio/arz/<id>.wav'}
```

train and dev also carry `answer`; devtest does not, since the development phase is scored on
it. The clips are 24 kHz mono. See [../data](../data) for the media archives.

## Baseline

[`../baselines/baseline_qwen_omni.py`](../baselines) runs Qwen2.5-Omni in 4-bit on the image
plus the clip:

```bash
python ../baselines/baseline_qwen_omni.py --task sqa --track arz --split devtest \
    --media-root ~/mmcqa --out prediction.jsonl
```

## Submit

```json
{"id": "<id>", "answers": {"arz": "..."}}
```

The format checker and the scorer follow shortly; see the repository README.

Ranked by **BERTScore-F1**; BLEU, ROUGE-L and Coverage are reported alongside.
