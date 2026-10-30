"""SentencePiece vocabulary shared by the from-scratch Transformer."""

from __future__ import annotations

from pathlib import Path

from src.config import CHECKPOINT_DIR


class SentencePieceVocab:
    def __init__(self, model_path: Path):
        import sentencepiece as spm

        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    @property
    def vocab_size(self) -> int:
        return int(self.sp.vocab_size())

    def encode(self, text: str, max_len: int) -> list[int]:
        pieces = self.sp.encode(text, out_type=int)[: max(0, max_len - 2)]
        return [self.bos_id, *pieces, self.eos_id]

    def decode(self, ids: list[int]) -> str:
        cleaned = [token for token in ids if token not in {self.pad_id, self.bos_id, self.eos_id}]
        return self.sp.decode(cleaned).strip()

    def token_length(self, text: str) -> int:
        return len(self.sp.encode(text, out_type=int))


def train_sentencepiece(pairs: list[dict], vocab_size: int, model_path: Path | None = None) -> SentencePieceVocab:
    import sentencepiece as spm

    model_path = Path(model_path or CHECKPOINT_DIR / "spm")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = model_path.with_suffix(".corpus.txt")
    with open(corpus, "w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(row["en"].replace("\n", " ") + "\n")
            handle.write(row["hi"].replace("\n", " ") + "\n")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_path),
        vocab_size=int(vocab_size),
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        hard_vocab_limit=False,
        num_threads=4,
        train_extremely_large_corpus=False,
    )
    corpus.unlink(missing_ok=True)
    return SentencePieceVocab(model_path.with_suffix(".model"))
