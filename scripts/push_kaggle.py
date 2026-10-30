"""Submit a private Kaggle GPU script. Tokens stay on this machine and on Kaggle."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "kaggle_private"
REPO_URL = "https://github.com/Coder9797/en-hi-machine-translation.git"


def _kaggle_username() -> str:
    import json as _json
    import ssl
    import urllib.request

    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = ssl.create_default_context()
    token = Path.home().joinpath(".kaggle", "access_token").read_text(encoding="utf-8").strip()
    request = urllib.request.Request(
        "https://www.kaggle.com/api/v1/hello",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=30, context=context) as response:
        payload = _json.loads(response.read().decode())
    return payload["userName"]


def _hf_token() -> str:
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise SystemExit("Hugging Face is not logged in on this machine.")
    return token


def build(stages: list[str] | None = None) -> Path:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    username = _kaggle_username()
    encoded = base64.b64encode(_hf_token().encode()).decode()
    command = ["python", "-m", "src.pipeline"]
    if stages:
        command.extend(["--stages", *stages])
    command_literal = ", ".join(repr(part) for part in command)
    (PRIVATE / "train.py").write_text(
        textwrap.dedent(
            f"""
            import base64
            import os
            import shutil
            import subprocess
            from pathlib import Path

            os.environ["HF_TOKEN"] = base64.b64decode("{encoded}").decode()
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["PYTHONUNBUFFERED"] = "1"
            work = Path("/kaggle/working/project")
            if work.exists():
                shutil.rmtree(work)
            subprocess.check_call(["git", "clone", "--depth", "1", "{REPO_URL}", str(work)])
            os.chdir(work)
            subprocess.check_call(
                [
                    "pip", "install", "-q",
                    "sentencepiece", "sacrebleu>=2.4", "peft", "accelerate", "bitsandbytes",
                    "transformers>=4.44", "datasets", "pyyaml", "matplotlib", "pandas", "tqdm",
                ]
            )
            subprocess.check_call([{command_literal}])
            """
        ).lstrip(),
        encoding="utf-8",
    )
    metadata = {
        "id": f"{username}/en-hi-machine-translation",
        "title": "en-hi-machine-translation",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (PRIVATE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return PRIVATE


def push(stages: list[str] | None = None) -> None:
    folder = build(stages)
    token = Path.home().joinpath(".kaggle", "access_token").read_text(encoding="utf-8").strip()
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    subprocess.check_call(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(folder),
            "--accelerator",
            "NvidiaTeslaT4",
            "--timeout",
            "43200",
        ],
        env=env,
    )


if __name__ == "__main__":
    import sys

    push(sys.argv[1:] or None)
