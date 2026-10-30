"""Beam-search evaluation, timing, temperature sweep, fertility, and ethics probes."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from src.config import CHECKPOINT_DIR, DATA_DIR, GENERATION_DIR, RESULTS_DIR, device, ensure_dirs, load_config
from src.data import load_splits
from src.transformer import TransformerMT
from src.vocab import SentencePieceVocab


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _score(hypotheses: list[str], references: list[str]) -> dict:
    import sacrebleu

    tokenize = "char"
    bleu_metric = sacrebleu.metrics.BLEU(tokenize="char")
    bleu = bleu_metric.corpus_score(hypotheses, [references])
    chrf_metric = sacrebleu.metrics.CHRF(word_order=0)
    chrf = chrf_metric.corpus_score(hypotheses, [references])
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "bleu_signature": str(bleu),
        "chrf_signature": str(chrf),
        "bleu_tokenize": tokenize,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_transformer(cfg: dict):
    blob = torch.load(CHECKPOINT_DIR / "transformer.pt", map_location="cpu", weights_only=False)
    spec = blob["transformer"]
    model = TransformerMT(
        vocab_size=int(blob["vocab_size"]),
        d_model=int(spec["d_model"]),
        nhead=int(spec["nhead"]),
        num_layers=int(spec["num_layers"]),
        d_ff=int(spec["d_ff"]),
        dropout=0.0,
        pad_id=0,
        max_len=int(blob["max_len"]),
    )
    model.load_state_dict(blob["model"])
    dev = device()
    model.to(dev).eval()
    vocab = SentencePieceVocab(CHECKPOINT_DIR / "spm.model")
    return model, vocab, dev


def _transformer_texts(model, vocab, dev, sentences: list[str], cfg: dict, temperature: float | None) -> list[str]:
    texts = []
    max_len = int(cfg["max_len"])
    for sentence in sentences:
        ids = vocab.encode(sentence, max_len)
        src = torch.tensor([ids], dtype=torch.long, device=dev)
        if temperature is None:
            predicted = model.beam_search(src, vocab.bos_id, vocab.eos_id, int(cfg["beam_size"]), max_len)[0]
        else:
            predicted = model.sample(src, vocab.bos_id, vocab.eos_id, float(temperature), max_len)[0]
        text = vocab.decode(predicted).strip() or "."
        texts.append(text)
    return texts


def _timed(fn, sentences: list[str]) -> tuple[list[str], float]:
    if len(sentences) > 2:
        fn(sentences[:2])
    _sync()
    started = time.perf_counter()
    texts = fn(sentences)
    _sync()
    elapsed = time.perf_counter() - started
    return texts, elapsed / max(len(sentences), 1)


def _fertility_transformer(vocab, english: list[str], hindi: list[str]) -> dict:
    return {
        "english_tokens": sum(vocab.token_length(text) for text in english) / len(english),
        "hindi_tokens": sum(vocab.token_length(text) for text in hindi) / len(hindi),
    }


def _fertility_hf(tokenizer, english: list[str], hindi: list[str]) -> dict:
    def length(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return {
        "english_tokens": sum(length(text) for text in english) / len(english),
        "hindi_tokens": sum(length(text) for text in hindi) / len(hindi),
    }


def _free(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def available_models() -> list[str]:
    found = []
    if (CHECKPOINT_DIR / "transformer.pt").exists() and (CHECKPOINT_DIR / "spm.model").exists():
        found.append("transformer")
    for name in ("mt5", "gemma", "llama"):
        if (CHECKPOINT_DIR / name).exists():
            found.append(name)
    return found


def evaluate(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs()
    splits = load_splits()
    test = splits["test"]
    references = [row["hi"] for row in test]
    english = [row["en"] for row in test]
    speed_n = int(cfg["speed_sentences"])
    temp_n = int(cfg["temperature_sentences"])
    probes = json.loads((DATA_DIR / "ethics_probes.json").read_text(encoding="utf-8"))
    train_stats = {}
    stats_path = RESULTS_DIR / "train_stats.json"
    if stats_path.exists():
        train_stats.update(json.loads(stats_path.read_text(encoding="utf-8")))
    transformer_stats_path = RESULTS_DIR / "transformer_train.json"
    if transformer_stats_path.exists():
        raw = json.loads(transformer_stats_path.read_text(encoding="utf-8"))
        train_stats["transformer"] = {
            "train_seconds": raw["train_seconds"],
            "parameters": raw["parameters"],
            "trainable_parameters": raw["parameters"],
        }

    metrics = {"models": {}, "temperatures": {}, "fertility": {}, "ethics": []}
    sample_rows = []

    def official(name: str, texts: list[str], seconds_per: float, total: int, trainable: int) -> None:
        scored = _score(texts, references)
        info = train_stats.get(name, {})
        metrics["models"][name] = {
            **scored,
            "inference_seconds_per_sentence": seconds_per,
            "parameters": int(info.get("parameters", total)),
            "trainable_parameters": int(info.get("trainable_parameters", trainable)),
            "train_seconds": info.get("train_seconds"),
        }
        rows = []
        for index, (src, ref, hyp) in enumerate(zip(english, references, texts)):
            rows.append({"id": index, "en": src, "hi": ref, "prediction": hyp})
        _write_jsonl(GENERATION_DIR / f"{name}.jsonl", rows)

    names = available_models()
    print("evaluating", names)
    if "transformer" in names:
        model, vocab, dev = _load_transformer(cfg)
        texts, per_sentence = _timed(
            lambda batch: _transformer_texts(model, vocab, dev, batch, cfg, None),
            english[:speed_n],
        )
        full = texts + _transformer_texts(model, vocab, dev, english[speed_n:], cfg, None)
        total = sum(parameter.numel() for parameter in model.parameters())
        official("transformer", full, per_sentence, total, total)
        metrics["fertility"]["transformer"] = _fertility_transformer(vocab, english[:200], references[:200])
        temp_scores = {}
        for temperature in cfg["temperatures"]:
            sampled = _transformer_texts(model, vocab, dev, english[:temp_n], cfg, float(temperature))
            temp_scores[str(temperature)] = _score(sampled, references[:temp_n])["chrf"]
        metrics["temperatures"]["transformer"] = temp_scores
        ethics = _transformer_texts(model, vocab, dev, [row["en"] for row in probes], cfg, None)
        for probe, text in zip(probes, ethics):
            metrics["ethics"].append({**probe, "model": "transformer", "prediction": text})
        _free(model)

    from src.hf_experiment import generate_causal, generate_mt5, load_causal, load_mt5

    if "mt5" in names:
        model, tokenizer = load_mt5()
        texts, per_sentence = _timed(
            lambda batch: generate_mt5(model, tokenizer, batch, cfg, None, batch_size=1),
            english[:speed_n],
        )
        full = texts + generate_mt5(model, tokenizer, english[speed_n:], cfg, None, batch_size=8)
        total = sum(parameter.numel() for parameter in model.parameters())
        official("mt5", full, per_sentence, total, total)
        metrics["fertility"]["mt5"] = _fertility_hf(tokenizer, english[:200], references[:200])
        temp_scores = {}
        for temperature in cfg["temperatures"]:
            sampled = generate_mt5(model, tokenizer, english[:temp_n], cfg, float(temperature), batch_size=8)
            temp_scores[str(temperature)] = _score(sampled, references[:temp_n])["chrf"]
        metrics["temperatures"]["mt5"] = temp_scores
        ethics = generate_mt5(model, tokenizer, [row["en"] for row in probes], cfg, None, batch_size=4)
        for probe, text in zip(probes, ethics):
            metrics["ethics"].append({**probe, "model": "mt5", "prediction": text})
        _free(model, tokenizer)

    for name in ("gemma", "llama"):
        if name not in names:
            continue
        model, tokenizer = load_causal(name)
        texts, per_sentence = _timed(
            lambda batch, model=model, tokenizer=tokenizer: generate_causal(model, tokenizer, batch, cfg, None, 1),
            english[:speed_n],
        )
        full = texts + generate_causal(model, tokenizer, english[speed_n:], cfg, None, 2)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        official(name, full, per_sentence, total, trainable)
        metrics["fertility"][name] = _fertility_hf(tokenizer, english[:200], references[:200])
        temp_scores = {}
        for temperature in cfg["temperatures"]:
            sampled = generate_causal(model, tokenizer, english[:temp_n], cfg, float(temperature), 4)
            temp_scores[str(temperature)] = _score(sampled, references[:temp_n])["chrf"]
        metrics["temperatures"][name] = temp_scores
        ethics = generate_causal(model, tokenizer, [row["en"] for row in probes], cfg, None, 2)
        for probe, text in zip(probes, ethics):
            metrics["ethics"].append({**probe, "model": name, "prediction": text})
        _free(model, tokenizer)

    if (RESULTS_DIR / "llama_status.json").exists() and "llama" not in metrics["models"]:
        metrics["models"]["llama"] = json.loads((RESULTS_DIR / "llama_status.json").read_text(encoding="utf-8"))

    for index in range(min(int(cfg["sample_rows"]), len(test))):
        row = {"id": index, "en": english[index], "hi": references[index]}
        for name in metrics["models"]:
            path = GENERATION_DIR / f"{name}.jsonl"
            if path.exists():
                # filled below from the jsonl to avoid holding every string twice
                pass
        sample_rows.append(row)
    by_model = {}
    for name in list(metrics["models"]):
        path = GENERATION_DIR / f"{name}.jsonl"
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()[: len(sample_rows)]
            by_model[name] = [json.loads(line)["prediction"] for line in lines]
    for index, row in enumerate(sample_rows):
        for name, preds in by_model.items():
            row[name] = preds[index]
    _write_jsonl(RESULTS_DIR / "samples.jsonl", sample_rows)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_samples_md(sample_rows, list(by_model))
    print("wrote", RESULTS_DIR / "metrics.json")
    return metrics


def _write_samples_md(rows: list[dict], models: list[str]) -> None:
    header = ["English", "Reference", *models]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells = [row["en"], row["hi"]] + [row.get(model, "") for model in models]
        cells = [cell.replace("|", "/").replace("\n", " ") for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    (RESULTS_DIR / "samples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
