"""
MMCultureQA baseline — Qwen2.5-Omni, 4-bit
==========================================
One open model for both tasks, because Qwen2.5-Omni takes images, audio and
text in the same conversation:

    Task 1 (Spoken VQA)   image + the spoken question (wav)
    Task 2 (Textual VQA)  image + the written question

It writes a submission-ready ``prediction.jsonl``. Run it once per track and
the answers merge into the same file, so one file can cover all four tracks.

This is a reference, not a strong system: greedy decoding, a short prompt, no
retrieval and no fine-tuning. You are free to change everything.

Setup
-----
    pip install -r requirements.txt
    # media: unzip the archives you need next to each other, e.g.
    #   mmcqa/images/<id>.jpg and mmcqa/audio/<lang>/<id>.wav

Usage
-----
    # Task 2, English, the dev split (which has answers, so you can score it)
    python baseline_qwen_omni.py --task qa --track en --split dev \
        --media-root ~/mmcqa --limit 50

    # Task 1, Egyptian Arabic, the devtest split, appending to the same file
    python baseline_qwen_omni.py --task sqa --track arz --split devtest \
        --media-root ~/mmcqa --out prediction.jsonl

A 3B model in 4-bit fits a 16 GB GPU (a free Colab T4). Add ``--no-quantize``
if you have the memory, or ``--model Qwen/Qwen2.5-Omni-7B`` for the larger one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = "QCRI/MMCQA-SemEval27"
# Released tracks today. The task covers more languages and more regions over
# time, so --track takes any released code: the track name is built as
# <task>_<region>_<variety> and looked up on the Hub.
LANGUAGE = {"en": "English", "msa": "Modern Standard Arabic",
            "arz": "Egyptian Arabic", "ajp": "Levantine Arabic"}
# Qwen2.5-Omni expects its own system identity; the task line is appended to it.
SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
          "perceiving auditory and visual inputs, as well as generating text and speech.")
INSTRUCTION = ("Answer the question about the image in one or two sentences, in {language}. "
               "Give only the answer.")


def load_rows(task: str, track: str, split: str, limit: int | None,
              region: str = "mena") -> list[dict]:
    from datasets import load_dataset
    config = f"{task}_{region}_{track}"
    rows = load_dataset(REPO, config, split=split).to_list()
    return rows[:limit] if limit else rows


def build_model(model_id: str, quantize: bool, max_pixels: int):
    import torch
    from transformers import (BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration,
                              Qwen2_5OmniProcessor)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = None
    if quantize:
        # 4-bit NF4 roughly halves the weight memory, so the 3B fits a 16 GB card
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, max_pixels=max_pixels)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", quantization_config=quant,
        attn_implementation="sdpa")
    model.disable_talker()              # text only; no speech is generated
    model.eval()
    print(f"[baseline] {model_id} loaded (4-bit: {quantize}, dtype {dtype})")
    return model, processor


def answer(model, processor, image: Path, audio: Path | None, question: str | None,
           track: str, max_new_tokens: int) -> str:
    from qwen_omni_utils import process_mm_info
    content: list[dict] = [{"type": "image", "image": str(image)}]
    if audio is not None:
        content.append({"type": "audio", "audio": str(audio)})
    else:
        content.append({"type": "text", "text": question})
    language = LANGUAGE.get(track, "the language of the question")
    content.append({"type": "text", "text": INSTRUCTION.format(language=language)})
    conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": content}]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)
    generated = model.generate(**inputs, return_audio=False, do_sample=False,
                               max_new_tokens=max_new_tokens, use_audio_in_video=False)
    out = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out[0].strip()


def merge_into(path: Path, track: str, answers: dict[str, str]) -> None:
    """Keep one row per id, with one answer per track, so runs can be combined."""
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = {"id": str(row["id"]),
                                        "answers": dict(row.get("answers", {}))}
    for item_id, text in answers.items():
        rows.setdefault(item_id, {"id": item_id, "answers": {}})["answers"][track] = text
    with path.open("w", encoding="utf-8") as fh:
        for row in rows.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["sqa", "qa"],
                        help="sqa = Task 1 (spoken question), qa = Task 2 (written question)")
    parser.add_argument("--track", required=True,
                        help=f"Variety code of a released track, e.g. {', '.join(LANGUAGE)}")
    parser.add_argument("--region", default="mena", help="Region of the track (default: mena)")
    parser.add_argument("--split", default="devtest", choices=["train", "dev", "devtest"])
    parser.add_argument("--media-root", required=True, type=Path,
                        help="Where you unzipped the archives (holds images/ and audio/)")
    parser.add_argument("--out", default=Path("prediction.jsonl"), type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--no-quantize", action="store_true", help="Load in full precision")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--limit", type=int, help="Only the first N items, for a smoke test")
    parser.add_argument("--zip", action="store_true", help="Also write prediction.zip")
    args = parser.parse_args()

    rows = load_rows(args.task, args.track, args.split, args.limit, args.region)
    print(f"[baseline] {args.task}_{args.region}_{args.track} / {args.split}: {len(rows)} items")
    missing = [r for r in rows if not (args.media_root / r["image"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images are not under {args.media_root} "
                         f"(e.g. {missing[0]['image']}). Unzip archives/images_{args.region}_{args.split}.zip there.")
    if args.task == "sqa":
        gone = [r for r in rows if not (args.media_root / r["audio"]).is_file()]
        if gone:
            raise SystemExit(f"{len(gone)} clips are not under {args.media_root} "
                             f"(e.g. {gone[0]['audio']}). Unzip "
                             f"archives/audio_{args.region}_{args.track}_{args.split}.zip there.")

    model, processor = build_model(args.model, not args.no_quantize, args.max_pixels)
    answers, started = {}, time.time()
    for i, row in enumerate(rows, 1):
        try:
            answers[str(row["id"])] = answer(
                model, processor, args.media_root / row["image"],
                args.media_root / row["audio"] if args.task == "sqa" else None,
                row.get("question"), args.track, args.max_new_tokens)
        except Exception as exc:                 # one bad item must not lose the run
            print(f"  [{row['id'][:12]}] failed: {type(exc).__name__}: {str(exc)[:120]}")
            answers[str(row["id"])] = ""
        if i % 25 == 0 or i == len(rows):
            rate = i / (time.time() - started)
            print(f"  {i}/{len(rows)}  {rate:.2f} items/s  "
                  f"eta {(len(rows)-i)/rate/60:.0f} min", flush=True)

    merge_into(args.out, args.track, answers)
    print(f"[baseline] wrote {args.out} ({len(answers)} answers for {args.track})")
    if args.zip:
        import zipfile
        archive = args.out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, "prediction.jsonl")
        print(f"[baseline] wrote {archive}")
    print("check the format and score it with the tools in ../format_checker and ../scorer")


if __name__ == "__main__":
    main()
