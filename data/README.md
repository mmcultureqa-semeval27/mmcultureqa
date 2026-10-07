# Data

Everything is on Hugging Face: <https://huggingface.co/datasets/QCRI/MMCQA-SemEval27>.
Nothing is duplicated in this repository.

The tracks below are the ones released so far, all in the MENA region; more languages and
regions follow in the same layout.

## download_data.py

One command gets the rows, the media archives, and unpacks them into the paths the rows use:

```bash
pip install huggingface_hub pyarrow

python download_data.py --tracks en --splits dev --root ~/mmcqa              # one track
python download_data.py --tracks en msa arz --splits dev devtest --root ~/mmcqa
python download_data.py --task qa --tracks ajp --splits train --root ~/mmcqa  # no audio
python download_data.py --tracks en --splits train --root ~/mmcqa --no-media  # rows only
```

It ends by checking that every image and clip a row points at is on disk.

## Rows

Rows are small and load in one line. Media is referenced by relative path.

```python
from datasets import load_dataset

qa  = load_dataset("QCRI/MMCQA-SemEval27", "qa_mena_en",  split="dev")      # Task 2
sqa = load_dataset("QCRI/MMCQA-SemEval27", "sqa_mena_arz", split="devtest") # Task 1
```

| field | Task 1 (`sqa_*`) | Task 2 (`qa_*`) | meaning |
|---|---|---|---|
| `id` | ✓ | ✓ | sha256 of the image; the same item carries the same id in every track |
| `image` | ✓ | ✓ | relative path, `images/<id>.jpg` |
| `audio` | ✓ | — | relative path, `audio/<lang>/<id>.wav`, 24 kHz mono |
| `question` | — | ✓ | the question as text |
| `answer` | train, dev | train, dev | the reference answer |

| split | items per track | has `answer` |
|---|---|---|
| train | 10,000 | yes |
| dev | 1,000 | yes |
| devtest | 1,000 | no — the development phase is scored on it |
| test | — | released for the evaluation period |

## Media

Images and audio ship as zip archives that unpack into exactly the paths the rows use, so
download only the tracks you work on.

```python
import zipfile
from huggingface_hub import hf_hub_download

REPO, ROOT = "QCRI/MMCQA-SemEval27", "mmcqa"     # ROOT: where you keep the media

for name in ["images_mena_devtest.zip", "audio_mena_arz_devtest.zip"]:
    path = hf_hub_download(REPO, f"archives/{name}", repo_type="dataset")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(ROOT)                      # -> mmcqa/images/, mmcqa/audio/arz/
```

```text
archives/images_mena_{train,dev,devtest}.zip           -> images/<id>.jpg
archives/audio_mena_<lang>_{train,dev,devtest}.zip     -> audio/<lang>/<id>.wav
```

The image archive is shared by all four tracks, so it is downloaded once. An audio archive is
about 2 GB for train and a tenth of that for dev or devtest.

Joining the rows to the files is then a path join:

```python
import os
from datasets import Audio, Image

sqa = (sqa.map(lambda r: {"image": os.path.join(ROOT, r["image"]),
                          "audio": os.path.join(ROOT, r["audio"])})
          .cast_column("image", Image()).cast_column("audio", Audio(sampling_rate=24000)))
```

## Audio

The spoken questions in train, dev and devtest are synthetic, each clip voice-cloned from a
real speaker of that variety, single channel at 24 kHz. Speakers do not cross splits. The
test set will use human recordings.
