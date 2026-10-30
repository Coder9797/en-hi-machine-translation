"""Fine-tune mT5 and a decoder-only model with LoRA."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

from src.config import CHECKPOINT_DIR, RESULTS_DIR, ensure_dirs, load_config, set_seed
from src.data import load_splits


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def login_if_needed() -> None:
    token = hf_token()
    if not token:
        return
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def _is_gated(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "gated" in name or "gated" in text or "awaiting a review" in text or "authorized list" in text


def _device_of(model) -> torch.device:
    if hasattr(model, "device") and str(model.device) != "meta":
        try:
            return model.device
        except Exception:
            pass
    return next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")


def _ids(value) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token) for token in value]


def causal_example(tokenizer, prompt: str, source: str, target: str, max_len: int):
    user = prompt.format(sentence=source)
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]
    full = _ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    prefix = _ids(
        tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
    )
    matched = 0
    while matched < len(prefix) and matched < len(full) and prefix[matched] == full[matched]:
        matched += 1
    full = full[:max_len]
    if matched >= len(full):
        return None
    labels = [-100] * matched + full[matched:]
    return full, labels


def _pad(rows: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(ids) for ids, _ in rows)
    input_ids, labels, mask = [], [], []
    for ids, labs in rows:
        extra = width - len(ids)
        input_ids.append(ids + [pad_id] * extra)
        labels.append(labs + [-100] * extra)
        mask.append([1] * len(ids) + [0] * extra)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def _mt5_rows(tokenizer, pairs, prefix: str, max_len: int) -> list[tuple[list[int], list[int], list[int]]]:
    encoded = []
    for row in pairs:
        source = tokenizer(
            prefix + row["en"],
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
        )
        target = tokenizer(text_target=row["hi"], max_length=max_len, truncation=True)
        labels = [(-100 if token == tokenizer.pad_token_id else int(token)) for token in target["input_ids"]]
        if any(token != -100 for token in labels):
            encoded.append((list(map(int, source["input_ids"])), list(map(int, source["attention_mask"])), labels))
    return encoded


def _collate_mt5(rows, pad_id: int) -> dict[str, torch.Tensor]:
    src_w = max(len(row[0]) for row in rows)
    tgt_w = max(len(row[2]) for row in rows)
    input_ids, mask, labels = [], [], []
    for src, src_mask, labs in rows:
        input_ids.append(src + [pad_id] * (src_w - len(src)))
        mask.append(src_mask + [0] * (src_w - len(src)))
        labels.append(labs + [-100] * (tgt_w - len(labs)))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _move(batch: dict[str, torch.Tensor], dev: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(dev) for key, value in batch.items()}


def _counts(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def _write_stats(name: str, seconds: float, total: int, trainable: int) -> None:
    path = RESULTS_DIR / "train_stats.json"
    stats = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stats[name] = {
        "train_seconds": seconds,
        "parameters": total,
        "trainable_parameters": trainable,
    }
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _run_steps(model, batches, dev, optimizer, base_lr, warmup, clip, accum, quantized: bool, use_fp16: bool = True) -> float:
    model.train()
    use_scaler = dev.type == "cuda" and use_fp16 and not quantized
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    seen = 0
    step = 0
    for index, batch in enumerate(batches):
        lr_scale = min(1.0, (step + 1) / max(warmup, 1))
        for group in optimizer.param_groups:
            group["lr"] = base_lr * lr_scale
        batch = _move(batch, dev)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda" and use_fp16):
            loss = model(**batch).loss / accum
        if not math.isfinite(float(loss.detach().float().item())):
            raise RuntimeError("training loss became non-finite; refusing to keep a broken checkpoint")
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (index + 1) % accum == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        total += float(loss.item()) * accum
        seen += 1
        if step and index % 50 == 0:
            print(f"  step {step} loss {total / seen:.4f}", flush=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total / max(seen, 1)


def train_mt5(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs()
    set_seed(int(cfg["seed"]))
    login_if_needed()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    spec = cfg["mt5"]
    tokenizer = AutoTokenizer.from_pretrained(spec["model_name"], token=hf_token())
    model = AutoModelForSeq2SeqLM.from_pretrained(spec["model_name"], token=hf_token())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    splits = load_splits()
    rows = _mt5_rows(tokenizer, splits["train"], cfg["mt5_prefix"], int(cfg["max_len"]))
    batches = []
    batch_size = int(spec["batch_size"])
    for start in range(0, len(rows), batch_size):
        batches.append(_collate_mt5(rows[start : start + batch_size], tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(spec["lr"]))
    started = time.perf_counter()
    loss = _run_steps(
        model,
        batches,
        dev,
        optimizer,
        float(spec["lr"]),
        int(spec["warmup_steps"]),
        float(spec["clip"]),
        int(spec["grad_accum"]),
        quantized=False,
        use_fp16=False,
    )
    elapsed = time.perf_counter() - started
    out = CHECKPOINT_DIR / "mt5"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    total, trainable = _counts(model)
    _write_stats("mt5", elapsed, total, trainable)
    print(f"mt5 trained in {elapsed:.1f}s final loss {loss:.4f}")
    return {"train_seconds": elapsed, "parameters": total, "final_loss": loss}


def _load_causal_base(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = hf_token()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantized = False
    kwargs = {"token": token, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = "auto"
            quantized = True
        except Exception as exc:
            print("4-bit setup unavailable:", type(exc).__name__)
    if not quantized and torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    elif not quantized:
        kwargs["torch_dtype"] = torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as exc:
        if _is_gated(exc):
            raise
        print("Retrying without 4-bit and with eager attention:", type(exc).__name__)
        kwargs.pop("quantization_config", None)
        kwargs["attn_implementation"] = "eager"
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        quantized = False
    return model, tokenizer, quantized


def train_causal(model_key: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs()
    set_seed(int(cfg["seed"]))
    login_if_needed()
    model_name = cfg["decoder"]["models"][model_key]
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(model_name, "config.json", token=hf_token())
    except Exception as exc:
        if model_key == "llama" and _is_gated(exc):
            status = {
                "model": model_name,
                "status": "pending",
                "reason": "Access has been requested and is awaiting approval from the model authors.",
            }
            (RESULTS_DIR / "llama_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print("llama skipped: access is still pending")
            return status
        raise

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model, tokenizer, quantized = _load_causal_base(model_name)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if quantized:
        model = prepare_model_for_kbit_training(model)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    leaf_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [name for name in ("q_proj", "k_proj", "v_proj", "o_proj") if name in leaf_names]
    spec = cfg["decoder"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(spec["lora_r"]),
            lora_alpha=int(spec["lora_alpha"]),
            lora_dropout=float(spec["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    splits = load_splits()
    examples = []
    for row in splits["train"]:
        built = causal_example(tokenizer, cfg["prompt"], row["en"], row["hi"], int(spec["max_len"]))
        if built is not None:
            examples.append(built)
    print(f"{model_key}: {len(examples)} training examples, 4-bit={quantized}, targets={targets}")
    batches = []
    batch_size = int(spec["batch_size"])
    for start in range(0, len(examples), batch_size):
        batches.append(_pad(examples[start : start + batch_size], tokenizer.pad_token_id))
    dev = _device_of(model)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(spec["lr"]))
    started = time.perf_counter()
    loss = _run_steps(
        model,
        batches,
        dev,
        optimizer,
        float(spec["lr"]),
        int(spec["warmup_steps"]),
        float(spec["clip"]),
        int(spec["grad_accum"]),
        quantized=quantized,
    )
    elapsed = time.perf_counter() - started
    out = CHECKPOINT_DIR / model_key
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "base_model.txt").write_text(model_name, encoding="utf-8")
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        total, trainable = _counts(model)
    _write_stats(model_key, elapsed, total, trainable)
    print(f"{model_key} trained in {elapsed:.1f}s final loss {loss:.4f} params {total} trainable {trainable}")
    return {"train_seconds": elapsed, "parameters": total, "trainable_parameters": trainable, "final_loss": loss}


def load_mt5():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    path = CHECKPOINT_DIR / "mt5"
    tokenizer = AutoTokenizer.from_pretrained(path)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=dtype)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()
    return model, tokenizer


def load_causal(model_key: str):
    from peft import PeftModel

    path = CHECKPOINT_DIR / model_key
    base_name = (path / "base_model.txt").read_text(encoding="utf-8").strip()
    model, tokenizer, _ = _load_causal_base(base_name)
    model = PeftModel.from_pretrained(model, path)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_mt5(model, tokenizer, sentences: list[str], cfg: dict, temperature: float | None, batch_size: int = 8) -> list[str]:
    dev = _device_of(model)
    outputs = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start : start + batch_size]
        encoded = tokenizer(
            [cfg["mt5_prefix"] + sentence for sentence in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(cfg["max_len"]),
        )
        encoded = {key: value.to(dev) for key, value in encoded.items()}
        kwargs = {"max_new_tokens": int(cfg["max_len"])}
        if temperature is None:
            kwargs.update(num_beams=int(cfg["beam_size"]), do_sample=False, early_stopping=True)
        else:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=1.0, num_beams=1)
        with torch.no_grad():
            generated = model.generate(**encoded, **kwargs)
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(text.strip() or "." for text in texts)
    return outputs


def generate_causal(model, tokenizer, sentences: list[str], cfg: dict, temperature: float | None, batch_size: int = 4) -> list[str]:
    dev = _device_of(model)
    previous = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs = []
    try:
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start : start + batch_size]
            messages = [[{"role": "user", "content": cfg["prompt"].format(sentence=sentence)}] for sentence in chunk]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            )
            if torch.is_tensor(encoded):
                encoded = {"input_ids": encoded}
            encoded = {key: value.to(dev) for key, value in encoded.items() if torch.is_tensor(value)}
            kwargs = {
                "max_new_tokens": int(cfg["max_len"]),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature is None:
                kwargs.update(num_beams=int(cfg["beam_size"]), do_sample=False)
            else:
                kwargs.update(do_sample=True, temperature=float(temperature), top_p=1.0, num_beams=1)
            with torch.no_grad():
                generated = model.generate(**encoded, **kwargs)
            width = encoded["input_ids"].shape[1]
            for row in generated:
                text = tokenizer.decode(row[width:], skip_special_tokens=True).strip()
                outputs.append(text or ".")
    finally:
        tokenizer.padding_side = previous
    return outputs
