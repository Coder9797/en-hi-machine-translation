"""Download, clean, and split the IIT Bombay English–Hindi corpus."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

from src.config import PROCESSED_DIR, ensure_dirs, load_config

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")
URL = re.compile(r"https?://\S+|www\.\S+")
SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = URL.sub(" ", text)
    text = SPACE.sub(" ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def accept_pair(en: str, hi: str, cfg: dict) -> bool:
    if not en or not hi or en == hi:
        return False
    if not LATIN.search(en) or not DEVANAGARI.search(hi):
        return False
    if max(len(en), len(hi)) > int(cfg["max_chars"]):
        return False
    en_words, hi_words = word_count(en), word_count(hi)
    if en_words < 1 or hi_words < 1:
        return False
    if en_words > int(cfg["max_words"]) or hi_words > int(cfg["max_words"]):
        return False
    ratio = max(en_words, hi_words) / min(en_words, hi_words)
    return ratio <= float(cfg["length_ratio"])


def clean_rows(rows: list[tuple[str, str]], cfg: dict) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    for en, hi in rows:
        en, hi = normalize(en), normalize(hi)
        if not accept_pair(en, hi, cfg):
            continue
        key = (en, hi)
        if key in seen:
            continue
        seen.add(key)
        kept.append({"en": en, "hi": hi})
    return kept


def _as_pair(row: dict) -> tuple[str, str]:
    translation = row.get("translation") if isinstance(row, dict) else None
    if isinstance(translation, dict):
        return str(translation.get("en", "")), str(translation.get("hi", ""))
    return str(row.get("en", "")), str(row.get("hi", ""))


def _load_hf_split(dataset_id: str, split: str) -> list[tuple[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    return [_as_pair(row) for row in dataset]


def build_splits(cfg: dict | None = None, force: bool = False) -> dict[str, list[dict]]:
    """Train and validation come from the official train split.

    The test set is drawn from the official held-out test split so training
    never sees those sentences.
    """
    cfg = cfg or load_config()
    ensure_dirs()
    out = {
        "train": PROCESSED_DIR / "train.jsonl",
        "validation": PROCESSED_DIR / "validation.jsonl",
        "test": PROCESSED_DIR / "test.jsonl",
    }
    if not force and all(path.exists() for path in out.values()):
        return {name: read_jsonl(path) for name, path in out.items()}

    dataset_id = cfg["dataset_id"]
    print(f"Loading {dataset_id}")
    train_rows = clean_rows(_load_hf_split(dataset_id, "train"), cfg)
    test_rows = clean_rows(_load_hf_split(dataset_id, "test"), cfg)
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    train_size = int(cfg["train_size"])
    val_size = int(cfg["val_size"])
    test_size = int(cfg["test_size"])
    need = train_size + val_size
    if len(train_rows) < need:
        raise RuntimeError(f"Need {need} clean training pairs, found {len(train_rows)}")
    if len(test_rows) < test_size:
        raise RuntimeError(f"Need {test_size} clean test pairs, found {len(test_rows)}")

    splits = {
        "train": train_rows[:train_size],
        "validation": train_rows[train_size : train_size + val_size],
        "test": test_rows[:test_size],
    }
    for name, rows in splits.items():
        write_jsonl(out[name], rows)
        print(f"{name}: {len(rows)}")
    manifest = {
        "dataset_id": dataset_id,
        "dataset_page": cfg["dataset_page"],
        "dataset_mirror": cfg["dataset_mirror"],
        "license": "CC BY-NC 4.0",
        "seed": int(cfg["seed"]),
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "filters": {
            "max_words": cfg["max_words"],
            "max_chars": cfg["max_chars"],
            "length_ratio": cfg["length_ratio"],
        },
    }
    (PROCESSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return splits


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_splits() -> dict[str, list[dict]]:
    ensure_dirs()
    paths = {
        "train": PROCESSED_DIR / "train.jsonl",
        "validation": PROCESSED_DIR / "validation.jsonl",
        "test": PROCESSED_DIR / "test.jsonl",
    }
    if not all(path.exists() for path in paths.values()):
        return build_splits()
    return {name: read_jsonl(path) for name, path in paths.items()}
