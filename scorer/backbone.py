"""
EvalAI evaluation script -- MMCultureQA 2027 (SemEval), Task 1 Spoken Visual QA / Task 2 Textual Visual QA.

How a submission is interpreted
    phase codename      = stage (dev, devtest, test ...)             -> chosen in the phase dropdown
    language            = the "Language" field of the upload form     -> chosen in the form (NOT read from the file)
    prediction.jsonl    = one line per item of the stage's public file: {"id": "...", "answer": "..."}
The result is returned for exactly one dataset split (the chosen language), so the team appears on that language tab only.

Fallbacks (API / scripts that do not fill the form): a "lang" key on every line, or an "answers" object with a single
language. If neither the form nor the file determines the language, the submission is rejected with a clear message.

Metrics (per language track)
    BERTScore-F1/P/R  jhu-clsp/mmBERT-base (ModernBERT, 22 layers), layer 17, idf off, no rescaling   -> ranking: F1
    BLEU              sacreBLEU corpus BLEU, tokenize=flores200 (falls back to 'intl' without network)
    ROUGE-L           rouge_score ROUGE-L F with a Unicode word tokenizer, mean over items
    Coverage          fraction of items with a non-empty answer (empty answers score 0)
"""
from __future__ import annotations

import collections
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

HUB_MODEL = "jhu-clsp/mmBERT-base"
HUB_REVISION = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
NUM_LAYERS = 17        # 22-layer ModernBERT; same 0.75 relative depth as mBERT layer 9 (bert_score default)
BATCH_SIZE = int(os.environ.get("MMCQA_BATCH_SIZE", "64"))
MAX_CHARS = 5000
LANG_FIELD = "Language"
M_F1, M_P, M_R, M_BLEU, M_ROUGE, M_COV = "BERTScore-F1", "BERTScore-P", "BERTScore-R", "BLEU", "ROUGE-L", "Coverage"
KNOWN_LANGS = {"msa", "ajp", "arz", "en", "bn", "ur", "hi", "as", "gu", "mr", "it", "es", "pt", "tr", "am", "om", "so", "ti", "zh", "ne", "si", "te", "sw"}

_SCORER = None
_ROUGE = None
_BLEU_TOK = None


def _log(msg: str) -> None:
    print(f"[mmcqa-eval] {msg}", flush=True)


# ----------------------------------------------------------------------------- I/O
def _read_jsonl_bytes(data: bytes, fname: str) -> list[dict]:
    rows = []
    for i, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{fname}:{i}: invalid JSON ({e.msg})") from None
        if not isinstance(obj, dict):
            raise ValueError(f"{fname}:{i}: each line must be a JSON object")
        rows.append(obj)
    if not rows:
        raise ValueError(f"{fname}: file is empty")
    return rows


def read_submission(path: str) -> tuple[list[dict], str]:
    p = Path(path)
    data = p.read_bytes()
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".jsonl") and "__MACOSX" not in n and not Path(n).name.startswith("._")]
            if not names:
                raise ValueError(f"the zip must contain prediction.jsonl (found: {zf.namelist()[:10]})")
            pref = [n for n in names if Path(n).name == "prediction.jsonl"]
            name = pref[0] if pref else names[0]
            return _read_jsonl_bytes(zf.read(name), Path(name).name), Path(name).name
    text = data.decode("utf-8-sig").lstrip()
    if text.startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise ValueError("a .json submission must be a list of objects")
        return rows, p.name
    return _read_jsonl_bytes(data, p.name), p.name


def load_gold(path: str) -> tuple[dict[str, dict], list[str]]:
    rows = _read_jsonl_bytes(Path(path).read_bytes(), Path(path).name)
    gold = {str(r["id"]): r for r in rows}
    langs = sorted({l for r in rows for l in r["answer"].keys()})
    return gold, langs


def _clean(x) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()[:MAX_CHARS]


def language_from_form(kwargs: dict) -> str | None:
    """The 'Language' answer of the upload form. Hosted workers pass the serialized submission as submission_metadata
    (a dict whose 'submission_metadata' is the attribute list); our remote worker passes the same dict."""
    meta = kwargs.get("submission_metadata")
    attrs = meta.get("submission_metadata") if isinstance(meta, dict) else meta
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except json.JSONDecodeError:
            attrs = None
    if isinstance(attrs, list):
        for a in attrs:
            if isinstance(a, dict) and str(a.get("name", "")).strip().lower() == LANG_FIELD.lower():
                v = a.get("value")
                if isinstance(v, list):
                    v = v[0] if len(v) == 1 else None
                return _lang_code(v) if v else None
    if isinstance(attrs, dict) and attrs.get(LANG_FIELD):
        return _lang_code(attrs[LANG_FIELD])
    return None


def _lang_code(value) -> str:
    """'Egyptian Arabic (arz)' -> 'arz'; 'arz' -> 'arz'."""
    v = str(value).strip()
    m = re.search(r"\(([A-Za-z_-]{2,8})\)\s*$", v)
    return (m.group(1) if m else v).lower()


def parse_predictions(rows: list[dict], gold_ids: set[str], form_lang: str | None, fname: str) -> tuple[str, dict[str, str]]:
    """Return (language, {id: answer}). The language comes from the form; the file may only confirm it."""
    file_langs: set[str] = set()
    for r in rows:
        if "lang" in r:
            file_langs.add(str(r["lang"]).strip().lower())
        if isinstance(r.get("answers"), dict):
            file_langs.update(str(k).strip().lower() for k in r["answers"])
    if form_lang:
        lang = form_lang
    elif len(file_langs) == 1:
        lang = next(iter(file_langs))
    else:
        raise ValueError("no language selected: choose the language in the 'Language' field of the submission form "
                         f"(languages found in the file: {sorted(file_langs) or 'none'})")
    preds: dict[str, str] = {}
    for i, r in enumerate(rows, 1):
        if "id" not in r:
            raise ValueError(f"{fname}:{i}: missing key 'id'")
        rid = str(r["id"]).strip()
        if "answer" in r:
            if "lang" in r and str(r["lang"]).strip().lower() != lang:
                continue                      # flat multi-language file: keep only the chosen language
            ans = r["answer"]
        elif isinstance(r.get("answers"), dict):
            if lang not in {str(k).strip().lower() for k in r["answers"]}:
                continue
            ans = {str(k).strip().lower(): v for k, v in r["answers"].items()}[lang]
        else:
            raise ValueError(f"{fname}:{i}: each line needs 'id' and 'answer'")
        if rid in preds:
            raise ValueError(f"{fname}:{i}: duplicate id {rid!r}")
        preds[rid] = _clean(ans)
    missing, extra = gold_ids - set(preds), set(preds) - gold_ids
    if missing or extra:
        raise ValueError(
            f"id mismatch with the gold set of this phase: missing={len(missing)} (e.g. {sorted(missing)[:3]}), "
            f"unknown={len(extra)} (e.g. {sorted(extra)[:3]}). Submit one line for each of the {len(gold_ids)} ids "
            "of this phase's public file (dev file for the Dev phase, devtest file for the Dev-test phase).")
    return lang, preds


# ----------------------------------------------------------------------------- metrics
def get_scorer():
    global _SCORER
    if _SCORER is None:
        import torch
        from bert_score import BERTScorer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(max(1, int(os.environ.get("MMCQA_THREADS", os.cpu_count() or 1))))
        model_type = os.environ.get("MMCQA_MODEL_DIR") or HUB_MODEL
        t0 = time.time()
        _SCORER = BERTScorer(model_type=model_type, num_layers=NUM_LAYERS, batch_size=BATCH_SIZE, nthreads=4,
                             idf=False, device=device, lang=None, rescale_with_baseline=False, use_fast_tokenizer=True)
        _SCORER._tokenizer.model_max_length = 512
        _log(f"BERTScorer ready: model={model_type} layer={NUM_LAYERS} device={device} ({time.time()-t0:.1f}s)")
    return _SCORER


class _UnicodeWordTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


def get_rouge():
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rougeL"], tokenizer=_UnicodeWordTokenizer())
    return _ROUGE


def corpus_bleu(cands: list[str], refs: list[str]) -> float:
    global _BLEU_TOK
    import sacrebleu
    if _BLEU_TOK is None:
        try:
            sacrebleu.corpus_bleu(["test"], [["test"]], tokenize="flores200"); _BLEU_TOK = "flores200"
        except Exception as e:  # noqa: BLE001
            _log(f"flores200 tokenizer unavailable ({e}); using 'intl'"); _BLEU_TOK = "intl"
    return float(sacrebleu.corpus_bleu(cands, [refs], tokenize=_BLEU_TOK).score) if cands else 0.0


def warm_up() -> None:
    get_scorer(); get_rouge(); corpus_bleu(["warm up"], ["warm up"])


def mean(xs) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


# ----------------------------------------------------------------------------- entry point
def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    t_start = time.time()
    gold, released = load_gold(test_annotation_file)
    rows, fname = read_submission(user_submission_file)
    form_lang = language_from_form(kwargs)
    if form_lang and form_lang not in released:
        hint = "this track is not open yet" if form_lang in KNOWN_LANGS else "unknown language code"
        raise ValueError(f"language '{form_lang}': {hint}. Languages open in this phase: {released}")
    lang_ids = lambda l: {i for i, r in gold.items() if l in r["answer"]}
    probe = form_lang or next(iter({str(r.get("lang", "")).lower() for r in rows if "lang" in r} or {""}))
    lang, preds = parse_predictions(rows, lang_ids(probe) if probe in released else set(gold), form_lang, fname)
    if lang not in released:
        hint = "this track is not open yet" if lang in KNOWN_LANGS else "unknown language code"
        raise ValueError(f"language '{lang}': {hint}. Languages open in this phase: {released}")
    gold = {i: gold[i] for i in lang_ids(lang)}
    _log(f"phase={phase_codename} language={lang} items={len(gold)} file={Path(user_submission_file).name}")

    ids = sorted(gold)
    cands = [preds[i] for i in ids]
    refs = [_clean(gold[i]["answer"][lang]) for i in ids]
    P = [0.0] * len(ids); R = [0.0] * len(ids); F = [0.0] * len(ids)
    nz = [k for k, c in enumerate(cands) if c]
    if nz:
        p, r, f = get_scorer().score([cands[k] for k in nz], [refs[k] for k in nz], verbose=False, batch_size=BATCH_SIZE)
        for j, k in enumerate(nz):
            P[k], R[k], F[k] = float(p[j]), float(r[j]), float(f[j])
    rouge = get_rouge()
    RL = [rouge.score(refs[k], cands[k])["rougeL"].fmeasure if cands[k] else 0.0 for k in range(len(ids))]
    block = {M_F1: round(mean(F), 6), M_P: round(mean(P), 6), M_R: round(mean(R), 6),
             M_BLEU: round(corpus_bleu(cands, refs), 4), M_ROUGE: round(mean(RL), 6), M_COV: round(len(nz) / len(ids), 4)}

    per_country = collections.defaultdict(list)
    for k, i in enumerate(ids):
        per_country[gold[i].get("country", "?")].append(F[k])
    meta = {"phase": phase_codename, "language": lang, "n_items": len(ids), "n_empty": len(ids) - len(nz),
            "duration_s": round(time.time() - t_start, 1),
            "f1_per_country": {c: round(mean(v), 4) for c, v in sorted(per_country.items())}}
    _log(f"RESULT {phase_codename}/{lang}: F1={block[M_F1]:.4f} BLEU={block[M_BLEU]:.2f} ROUGE-L={block[M_ROUGE]:.4f} "
         f"coverage={block[M_COV]} ({meta['duration_s']}s)")
    return {"result": [{lang: block}], "submission_result": block, "submission_metadata": meta}


if __name__ == "__main__":   # python main.py gold.jsonl prediction.jsonl <phase> <language>
    meta = [{"name": LANG_FIELD, "value": sys.argv[4]}] if len(sys.argv) > 4 else None
    out = evaluate(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "dev", submission_metadata=meta)
    print(json.dumps(out, indent=2, ensure_ascii=False))
