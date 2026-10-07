# [MMCultureQA](https://mmcultureqa-semeval27.github.io/) at SemEval 2027

MMCultureQA is a shared task on culturally grounded visual question answering. A system is
given an image and a question about it, spoken or written, and writes a short open-ended
answer. Questions cover food, places, customs and everyday objects, so the answer depends on
cultural knowledge as much as on what the image shows.

This repository holds the format checker, the scorer, and the baselines.
The data lives on [Hugging Face](https://huggingface.co/datasets/QCRI/MMCQA-SemEval27).

> **Status:** the data, the download script, the baseline and the format checker are here.
> The official scorer follows shortly, together with the submission platform.

- [Task 1: Spoken Visual QA](task1) — the question is an audio clip.
- [Task 2: Textual Visual QA](task2) — the same question as text.

Both tasks return a text answer and are scored identically, so one scorer and one format
checker serve both.

## Language tracks

Every item exists in all four varieties under the same `id`, with the same image, so results
are comparable across tracks. A track is `<task>_<region>_<variety>`; the region is what the questions
are about, which is why the same language can appear in more than one region (English about
the Arab world and English about another region are separate tracks).

| variety | code | Task 1 | Task 2 |
|---|---|---|---|
| English | `en` | `sqa_mena_en` | `qa_mena_en` |
| Modern Standard Arabic | `msa` | `sqa_mena_msa` | `qa_mena_msa` |
| Egyptian Arabic | `arz` | `sqa_mena_arz` | `qa_mena_arz` |
| Levantine Arabic | `ajp` | `sqa_mena_ajp` | `qa_mena_ajp` |

These are the tracks released so far, all in the MENA region. **More languages and more
regions follow**, each as its own track in the same layout, so a system written for one track
runs on a new one by changing the config name. Enter any subset: one task or both, one track
or all of them. Systems are ranked per task and per track.

## Repository structure

```text
.
├── baselines/           Qwen2.5-Omni reference system for both tasks
├── data/                how to download the data and the media
├── format_checker/      run before you submit
├── scorer/              the leaderboard's own evaluation script        (coming shortly)
├── task1/               Spoken Visual QA
├── task2/               Textual Visual QA
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. get the data: rows, media, unpacked and checked
python data/download_data.py --tracks en --splits dev --root ~/mmcqa

# 2. produce prediction.jsonl — baselines/ has a reference system
python baselines/baseline_qwen_omni.py --task qa --track en --split dev --media-root ~/mmcqa

# 3. check the format before you submit
python format_checker/check_format.py --pred prediction.jsonl --ids ~/mmcqa/rows/qa_mena_dev_en.parquet
```

## Submission format

One JSON object per line in `prediction.jsonl`, nested, answering several tracks per item,

```json
{"id": "<id>", "answers": {"en": "...", "msa": "..."}}
```

or flat, one line per item and track,

```json
{"id": "<id>", "lang": "en", "answer": "..."}
```

Submit exactly the ids of the split you are answering. An unanswered item scores zero, so a
poor answer still beats no answer.

## Evaluation

| metric | role |
|---|---|
| **BERTScore-F1** | **the official ranking metric**, `bert-base-multilingual-cased`, layer 9 |
| BERTScore-P / R | reported alongside |
| BLEU | sacreBLEU, `flores200` tokenizer |
| ROUGE-L | F measure with a Unicode word tokenizer |
| Coverage | share of items answered |

Scores are reported per track, and `all` is the macro-average over the tracks of the phase.
`scorer/` also offers [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base) as an alternative
encoder for comparison; the ranking stays on mBERT.

## Timeline

The official schedule is on the [task website](https://mmcultureqa-semeval27.github.io/#dates).
Train, dev and devtest for the MENA region are out now; more varieties and the test set follow.

## Licensing

The dataset terms are on the [dataset page](https://huggingface.co/datasets/QCRI/MMCQA-SemEval27).
Code in this repository is under the license in [LICENSE](LICENSE).

## Contact

- Website: <https://mmcultureqa-semeval27.github.io/>
- Dataset: <https://huggingface.co/datasets/QCRI/MMCQA-SemEval27>

## Citation

The task builds on the OASIS dataset. If you take part or use these resources, please cite:

```bibtex
@article{alam2025everydaymmqa,
  title = {{OASIS}: A Multilingual and Multimodal Dataset for Culturally Grounded Spoken Visual QA},
  author = {Alam, Firoj and Shahroor, Ali Ezzat and Hasan, Md. Arid and Ali, Zien Sheikh and Bhatti, Hunzalah Hassan and Kmainasi, Mohamed Bayan and Chowdhury, Shammur Absar and Mousi, Basel and Dalvi, Fahim and Durrani, Nadir and Milic-Frayling, Natasa},
  journal = {arXiv preprint arXiv:2510.06371},
  year = {2025},
}
```
