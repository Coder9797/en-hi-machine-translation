"""Train the from-scratch encoder–decoder Transformer."""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F

from src.config import CHECKPOINT_DIR, RESULTS_DIR, device, ensure_dirs, load_config, set_seed
from src.data import load_splits
from src.transformer import TransformerMT
from src.vocab import SentencePieceVocab, train_sentencepiece


def _pad(sequences: list[list[int]], pad_id: int) -> torch.Tensor:
    width = max(len(sequence) for sequence in sequences)
    batch = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        batch[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return batch


def _epoch(model, rows, vocab, cfg, dev, optimizer, train: bool) -> float:
    model.train(train)
    max_len = int(cfg["max_len"])
    batch_size = int(cfg["transformer"]["batch_size"])
    smoothing = float(cfg["transformer"]["label_smoothing"])
    clip = float(cfg["transformer"]["clip"])
    base_lr = float(cfg["transformer"]["lr"])
    warmup = int(cfg["transformer"]["warmup_steps"])
    total_loss = 0.0
    total_tokens = 0
    step = getattr(_epoch, "step", 0)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        src = _pad([vocab.encode(row["en"], max_len) for row in batch], vocab.pad_id).to(dev)
        tgt = _pad([vocab.encode(row["hi"], max_len) for row in batch], vocab.pad_id).to(dev)
        tgt_in, labels = tgt[:, :-1], tgt[:, 1:]
        if train:
            if step < warmup:
                lr = base_lr * (step + 1) / warmup
                for group in optimizer.param_groups:
                    group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(src, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=vocab.pad_id,
                label_smoothing=smoothing if train else 0.0,
            )
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                step += 1
        tokens = int(labels.ne(vocab.pad_id).sum().item())
        total_loss += float(loss.item()) * max(tokens, 1)
        total_tokens += max(tokens, 1)
    _epoch.step = step
    return total_loss / max(total_tokens, 1)


def train(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs()
    set_seed(int(cfg["seed"]))
    splits = load_splits()
    vocab = train_sentencepiece(splits["train"], int(cfg["sp_vocab_size"]), CHECKPOINT_DIR / "spm")
    dev = device()
    spec = cfg["transformer"]
    model = TransformerMT(
        vocab_size=vocab.vocab_size,
        d_model=int(spec["d_model"]),
        nhead=int(spec["nhead"]),
        num_layers=int(spec["num_layers"]),
        d_ff=int(spec["d_ff"]),
        dropout=float(spec["dropout"]),
        pad_id=vocab.pad_id,
        max_len=int(cfg["max_len"]),
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["lr"]), betas=(0.9, 0.98), eps=1e-9)
    best_state = None
    best_val = float("inf")
    patience_left = int(spec["patience"])
    history = []
    started = time.perf_counter()
    _epoch.step = 0
    for epoch in range(1, int(spec["epochs"]) + 1):
        train_loss = _epoch(model, splits["train"], vocab, cfg, dev, optimizer, train=True)
        val_loss = _epoch(model, splits["validation"], vocab, cfg, dev, optimizer, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"transformer epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}")
        if val_loss + 1e-4 < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(spec["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("transformer early stop")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab_size": vocab.vocab_size,
            "transformer": spec,
            "max_len": int(cfg["max_len"]),
            "spm": str(CHECKPOINT_DIR / "spm.model"),
        },
        CHECKPOINT_DIR / "transformer.pt",
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    stats = {
        "train_seconds": elapsed,
        "parameters": parameters,
        "trainable_parameters": parameters,
        "best_val_loss": best_val,
        "history": history,
    }
    (RESULTS_DIR / "transformer_train.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"transformer trained in {elapsed:.1f}s, parameters {parameters}")
    return stats


def smoke() -> None:
    """One synthetic step and a short beam search, used as a local correctness check."""
    torch.manual_seed(0)
    model = TransformerMT(vocab_size=40, d_model=32, nhead=4, num_layers=1, d_ff=64, dropout=0.0, max_len=12)
    src = torch.randint(4, 40, (2, 5))
    tgt = torch.randint(4, 40, (2, 6))
    tgt[:, 0] = 2
    tgt[:, -1] = 3
    logits = model(src, tgt[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 40), tgt[:, 1:].reshape(-1))
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("transformer smoke loss was not finite")
    hypothesis = model.beam_search(src[:1], bos_id=2, eos_id=3, beam_size=2, max_len=6)
    if len(hypothesis) != 1:
        raise RuntimeError("beam search did not return one hypothesis")
    print("transformer smoke ok", float(loss))
