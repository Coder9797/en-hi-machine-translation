"""Run the experiment stages in order."""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path

from src.config import PROCESSED_DIR, RESULTS_DIR, ensure_dirs, load_config


def _publish() -> None:
    working = Path("/kaggle/working")
    if not working.exists():
        return
    if RESULTS_DIR.exists():
        shutil.copytree(RESULTS_DIR, working / "results", dirs_exist_ok=True)
    if PROCESSED_DIR.exists():
        shutil.copytree(PROCESSED_DIR, working / "processed", dirs_exist_ok=True)
    checkpoints = Path("checkpoints")
    if checkpoints.exists():
        shutil.copytree(checkpoints, working / "checkpoints", dirs_exist_ok=True)


def _hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Apple MPS"
    except Exception:
        pass
    return "CPU"


def main() -> None:
    parser = argparse.ArgumentParser(description="English–Hindi translation experiment")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=["all"],
        help="data, transformer, mt5, gemma, llama, evaluate, report, or all",
    )
    args = parser.parse_args()
    stages = ["data", "transformer", "mt5", "gemma", "llama", "evaluate", "report"] if "all" in args.stages else args.stages
    ensure_dirs()
    cfg = load_config()
    (RESULTS_DIR / "hardware.json").write_text(
        json.dumps({"device": _hardware()}, indent=2),
        encoding="utf-8",
    )
    from src.hf_experiment import login_if_needed

    login_if_needed()
    for stage in stages:
        print(f"\n===== {stage} =====", flush=True)
        try:
            if stage == "data":
                from src.data import build_splits

                build_splits(cfg, force=not (PROCESSED_DIR / "test.jsonl").exists())
            elif stage == "transformer":
                from src.train_transformer import train

                train(cfg)
            elif stage == "mt5":
                from src.hf_experiment import train_mt5

                train_mt5(cfg)
            elif stage == "gemma":
                from src.hf_experiment import train_causal

                train_causal("gemma", cfg)
            elif stage == "llama":
                from src.hf_experiment import train_causal

                train_causal("llama", cfg)
            elif stage == "evaluate":
                from src.evaluate import evaluate

                evaluate(cfg)
            elif stage == "report":
                from src.report import report

                report()
            elif stage == "smoke":
                from src.train_transformer import smoke

                smoke()
            else:
                raise SystemExit(f"Unknown stage {stage}")
        except Exception:
            traceback.print_exc()
            _publish()
            raise
        _publish()


if __name__ == "__main__":
    main()
