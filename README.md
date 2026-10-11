# [MMCultureQA](https://mmcultureqa-semeval27.github.io/) at SemEval 2027

MMCultureQA is a shared task on culturally grounded visual question answering. A system is
given an image and a question about it, spoken or written, and writes a short open-ended
answer. Questions cover food, places, customs and everyday objects, so the answer depends on
cultural knowledge as much as on what the image shows.

This repository holds the format checker, the scorer, and the baselines.
The data lives on [Hugging Face](https://huggingface.co/datasets/QCRI/MMCQA-SemEval27).

## The two tasks

Both take an image and a question and return a short text answer; they differ only in how the
question arrives. The images and the questions are the same, so comparing the two tells you
what the speech step costs your system. Enter either task or both.

| | Task 1, Spoken Visual QA | Task 2, Textual Visual QA |
|---|---|---|
| input | image + the question as a wav, 24 kHz mono | image + the question as text |
| output | one or two sentences of text | one or two sentences of text |
| configs | `sqa_mena_<variety>` | `qa_mena_<variety>` |
| row fields | `id`, `image`, `audio` | `id`, `image`, `question` |

train and dev rows also carry `answer`; devtest does not, since the development phase is
scored on it. Both tasks are scored identically, so one scorer and one format checker serve
both.

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

These four are open now, all in the MENA region. **More language tracks open as their data
lands**, in South Asia, Europe, East Asia and the Horn of Africa; each arrives in the same
layout, so a system written for one track runs on a new one by changing the config name. The
[task website](https://mmcultureqa-semeval27.github.io/) announces a track when it opens.
Enter any subset: one task or both, one track or all of them. Systems are ranked per task and
per track.

## Repository structure

```text
.
├── data/                download_data.py: rows and media, unpacked and checked
├── baselines/           Qwen2.5-Omni reference system for both tasks
├── format_checker/      run before you submit
├── scorer/              the leaderboard's own evaluation script, so you can score locally
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. get the data: rows, media, unpacked and checked
python data/download_data.py --tracks en --splits dev --root ~/mmcqa

# 2. answer: --task qa for Task 2 (written), --task sqa for Task 1 (spoken)
python baselines/baseline_qwen_omni.py --task qa --track en --split devtest --media-root ~/mmcqa

# 3. check the format before you submit
python format_checker/check_format.py --pred prediction_qa_en.jsonl --lang en \
    --ids ~/mmcqa/rows/qa_mena_devtest_en.parquet

# 4. score yourself on a labelled split
python scorer/score.py --gold ~/mmcqa/rows/qa_mena_dev_en.parquet --pred prediction_qa_en.jsonl
```

## Submission format

A submission covers **one track**. One JSON object per line in `prediction.jsonl`, one line
per item of the split:

```json
{"id": "<id>", "answer": "..."}
```

The track is chosen in the `Language` field of the upload form, not read from the file; a
`lang` key on every line also works. Submit exactly the ids of the split you are answering.
An unanswered item scores zero, so a poor answer still beats no answer.

Upload `prediction.jsonl`, or a `.zip` containing it, to the phase of the stage you are
entering (Dev-test now; Test during the evaluation period).

## Evaluation

| metric | role |
|---|---|
| **BERTScore-F1** | **the official ranking metric**, [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base), layer 17 |
| BERTScore-P / R | reported alongside |
| BLEU | sacreBLEU, `flores200` tokenizer |
| ROUGE-L | F measure with a Unicode word tokenizer |
| Coverage | share of items answered |

Each submission is scored for its own track, and a per-country breakdown of F1 comes back
with it. `scorer/` can also run `bert-base-multilingual-cased` for comparison; the ranking
stays on mmBERT.

## Timeline

The official schedule is on the [task website](https://mmcultureqa-semeval27.github.io/#dates).
Train, dev and devtest for the MENA region are out now; more varieties and the test set follow.
Submissions run in stages: the **Dev-test** leaderboard is open on the devtest split, and the
**Test** phase opens for the evaluation period.

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
