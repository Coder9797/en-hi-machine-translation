"""Write plots and fill the results section of the README from metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import RESULTS_DIR, ROOT


def _done(models: dict) -> list[tuple[str, dict]]:
    rows = []
    for name, info in models.items():
        if "chrf" in info:
            rows.append((name, info))
    return rows


def render_tables(metrics: dict) -> str:
    done = _done(metrics.get("models", {}))
    if not done:
        return "No completed models are in `results/metrics.json` yet.\n"
    lines = [
        "| Model | chrF | BLEU | Parameters | Trainable | Train time | Inference s/sentence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, info in done:
        train = info.get("train_seconds")
        train_text = f"{train / 60:.1f} min" if isinstance(train, (int, float)) else "pending"
        lines.append(
            "| {name} | {chrf:.2f} | {bleu:.2f} | {parameters:,} | {trainable:,} | {train} | {speed:.3f} |".format(
                name=name,
                chrf=info["chrf"],
                bleu=info["bleu"],
                parameters=int(info["parameters"]),
                trainable=int(info.get("trainable_parameters", info["parameters"])),
                train=train_text,
                speed=float(info["inference_seconds_per_sentence"]),
            )
        )
    best = max(done, key=lambda item: (item[1]["chrf"], item[1]["bleu"], -int(item[1]["parameters"])))
    lines.append("")
    lines.append(
        f"Pre-registered rule: the best model is the highest chrF, then BLEU, then the smaller parameter count. "
        f"That model is **{best[0]}** (chrF {best[1]['chrf']:.2f})."
    )
    fertility = metrics.get("fertility") or {}
    if fertility:
        lines.extend(["", "### Tokenizer fertility", "", "| Model | Mean English tokens | Mean Hindi tokens |", "| --- | ---: | ---: |"])
        for name, info in fertility.items():
            lines.append(
                f"| {name} | {info['english_tokens']:.1f} | {info['hindi_tokens']:.1f} |"
            )
    temps = metrics.get("temperatures") or {}
    if temps:
        header = sorted({float(key) for info in temps.values() for key in info})
        lines.extend(["", "### Sampling temperature (chrF on 100 test sentences)", ""])
        lines.append("| Model | " + " | ".join(f"T={value:g}" for value in header) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in header) + " |")
        for name, info in temps.items():
            cells = []
            for value in header:
                raw = info.get(str(value))
                if raw is None:
                    raw = info.get(f"{value:g}")
                cells.append(f"{float(raw):.2f}" if raw is not None else "")
            lines.append("| " + name + " | " + " | ".join(cells) + " |")
    ethics = metrics.get("ethics") or []
    if ethics:
        models = []
        for row in ethics:
            if row["model"] not in models:
                models.append(row["model"])
        grouped: dict[str, dict] = {}
        for row in ethics:
            grouped.setdefault(row["id"], {"en": row["en"], "concern": row["concern"], "pred": {}})
            grouped[row["id"]]["pred"][row["model"]] = row["prediction"]
        lines.extend(["", "### Ethics probe translations", ""])
        lines.append("| Concern | English | " + " | ".join(models) + " |")
        lines.append("| --- | --- | " + " | ".join("---" for _ in models) + " |")
        for row in grouped.values():
            cells = [row["concern"], row["en"]] + [row["pred"].get(model, "") for model in models]
            cells = [cell.replace("|", "/").replace("\n", " ") for cell in cells]
            lines.append("| " + " | ".join(cells) + " |")
    pending = metrics.get("models", {}).get("llama", {})
    if pending.get("status") == "pending":
        lines.extend(["", "LLaMA is not in the table. Access was requested and is still awaiting approval. The same LoRA recipe is in `src/hf_experiment.py` and runs with `python -m src.pipeline --stages llama evaluate report` once access is granted."])
    return "\n".join(lines) + "\n"


def write_plots(metrics: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    done = _done(metrics.get("models", {}))
    if not done:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in done]
    chrf = [info["chrf"] for _, info in done]
    bleu = [info["bleu"] for _, info in done]
    figure, axis = plt.subplots(figsize=(7, 4))
    import numpy as np

    positions = np.arange(len(names))
    width = 0.36
    axis.bar(positions - width / 2, chrf, width, label="chrF")
    axis.bar(positions + width / 2, bleu, width, label="BLEU")
    axis.set_xticks(positions, names)
    axis.set_ylabel("score")
    axis.set_title("English–Hindi test scores")
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "scores.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    params = [info["parameters"] / 1e6 for _, info in done]
    axis.scatter(params, chrf)
    for name, param, score in zip(names, params, chrf):
        axis.annotate(name, (param, score), textcoords="offset points", xytext=(6, 4))
    axis.set_xlabel("parameters (millions)")
    axis.set_ylabel("chrF")
    axis.set_title("Quality against model size")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "params_vs_chrf.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    speeds = [info["inference_seconds_per_sentence"] for _, info in done]
    axis.bar(names, speeds)
    axis.set_ylabel("seconds per sentence")
    axis.set_title("Batch-1 inference time")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "inference_time.png", dpi=140)
    plt.close(figure)

    temps = metrics.get("temperatures") or {}
    if temps:
        figure, axis = plt.subplots(figsize=(6, 4))
        for name, info in temps.items():
            xs = sorted(float(key) for key in info)
            ys = [info[str(value)] if str(value) in info else info[f"{value:g}"] for value in xs]
            axis.plot(xs, ys, marker="o", label=name)
        axis.set_xlabel("temperature")
        axis.set_ylabel("chrF")
        axis.set_title("Sampling temperature")
        axis.legend()
        figure.tight_layout()
        figure.savefig(RESULTS_DIR / "temperature.png", dpi=140)
        plt.close(figure)

    history_path = RESULTS_DIR / "transformer_train.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        if history:
            figure, axis = plt.subplots(figsize=(6, 4))
            axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
            axis.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], label="validation")
            axis.set_xlabel("epoch")
            axis.set_ylabel("token loss")
            axis.set_title("From-scratch Transformer")
            axis.legend()
            figure.tight_layout()
            figure.savefig(RESULTS_DIR / "transformer_loss.png", dpi=140)
            plt.close(figure)


def splice_readme(tables: str) -> None:
    path = ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- AUTO-RESULTS:START -->", "<!-- AUTO-RESULTS:END -->"
    if start not in text or end not in text:
        return
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(before + start + "\n\n" + tables + "\n" + end + after, encoding="utf-8")


def report() -> None:
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        raise SystemExit("results/metrics.json is missing. Run evaluation first.")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    tables = render_tables(metrics)
    (RESULTS_DIR / "tables.md").write_text(tables, encoding="utf-8")
    write_plots(metrics)
    splice_readme(tables)
    print("updated README results and plots")


if __name__ == "__main__":
    report()
