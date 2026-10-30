"""Small pre-norm encoder–decoder Transformer for machine translation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        tensor = tensor.view(batch, length, self.nhead, self.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask: torch.Tensor | None):
        q = self._split(self.q(query))
        k = self._split(self.k(key))
        v = self._split(self.v(value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        weights = torch.nan_to_num(weights, nan=0.0)
        mixed = torch.matmul(weights, v).permute(0, 2, 1, 3).contiguous()
        mixed = mixed.view(query.size(0), query.size(1), -1)
        return self.out(mixed)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, h, mask))
        h = self.norm2(x)
        x = x + self.drop(self.ff(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask, cross_mask):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h, self_mask))
        h = self.norm2(x)
        x = x + self.drop(self.cross_attn(h, memory, memory, cross_mask))
        h = self.norm3(x)
        x = x + self.drop(self.ff(h))
        return x


class TransformerMT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 64,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model, max_len=max_len + 8, dropout=dropout)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embed.weight
        self.d_model = d_model

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.position(self.embed(tokens) * math.sqrt(self.d_model))

    @staticmethod
    def pad_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        return tokens.eq(pad_id).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_mask = self.pad_mask(src, self.pad_id)
        tgt_mask = self.causal_mask(tgt_in.size(1), tgt_in.device).unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_mask | self.pad_mask(tgt_in, self.pad_id)
        memory = self._embed(src)
        for layer in self.encoder:
            memory = layer(memory, src_mask)
        hidden = self._embed(tgt_in)
        for layer in self.decoder:
            hidden = layer(hidden, memory, tgt_mask, src_mask)
        return self.output(self.norm(hidden))

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        beam_size: int = 4,
        max_len: int | None = None,
    ) -> list[list[int]]:
        """Batch-1 beam search. Returns token ids for each source row, without BOS/EOS."""
        self.eval()
        max_len = max_len or self.max_len
        results = []
        for index in range(src.size(0)):
            results.append(self._beam_one(src[index : index + 1], bos_id, eos_id, beam_size, max_len))
        return results

    def _beam_one(self, src, bos_id, eos_id, beam_size, max_len) -> list[int]:
        device = src.device
        src_mask = self.pad_mask(src, self.pad_id)
        memory = self._embed(src)
        for layer in self.encoder:
            memory = layer(memory, src_mask)
        beams = [(0.0, [bos_id], False)]
        for _ in range(max_len - 1):
            candidates = []
            active = [beam for beam in beams if not beam[2]]
            if not active:
                break
            tgt = torch.tensor([beam[1] for beam in active], dtype=torch.long, device=device)
            memory_exp = memory.expand(tgt.size(0), -1, -1)
            src_mask_exp = src_mask.expand(tgt.size(0), -1, -1, -1)
            tgt_mask = self.causal_mask(tgt.size(1), device).unsqueeze(0).unsqueeze(0)
            hidden = self._embed(tgt)
            for layer in self.decoder:
                hidden = layer(hidden, memory_exp, tgt_mask, src_mask_exp)
            logits = self.output(self.norm(hidden[:, -1]))
            log_probs = F.log_softmax(logits, dim=-1)
            top_scores, top_ids = log_probs.topk(beam_size, dim=-1)
            next_beams = [beam for beam in beams if beam[2]]
            for beam_index, (score, tokens, _) in enumerate(active):
                for rank in range(beam_size):
                    token = int(top_ids[beam_index, rank])
                    next_score = score + float(top_scores[beam_index, rank])
                    done = token == eos_id
                    next_beams.append((next_score, tokens + [token], done))
            next_beams.sort(key=lambda item: item[0] / max(1, len(item[1]) - 1), reverse=True)
            beams = next_beams[:beam_size]
        best = max(beams, key=lambda item: item[0] / max(1, len(item[1]) - 1))
        ids = best[1][1:]
        if ids and ids[-1] == eos_id:
            ids = ids[:-1]
        return ids

    @torch.no_grad()
    def sample(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        temperature: float,
        max_len: int | None = None,
    ) -> list[list[int]]:
        self.eval()
        max_len = max_len or self.max_len
        outputs = []
        for index in range(src.size(0)):
            row = src[index : index + 1]
            src_mask = self.pad_mask(row, self.pad_id)
            memory = self._embed(row)
            for layer in self.encoder:
                memory = layer(memory, src_mask)
            tokens = [bos_id]
            for _ in range(max_len - 1):
                tgt = torch.tensor([tokens], dtype=torch.long, device=src.device)
                tgt_mask = self.causal_mask(tgt.size(1), src.device).unsqueeze(0).unsqueeze(0)
                hidden = self._embed(tgt)
                for layer in self.decoder:
                    hidden = layer(hidden, memory, tgt_mask, src_mask)
                logits = self.output(self.norm(hidden[:, -1])) / max(temperature, 1e-4)
                probs = torch.softmax(logits, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1).item())
                tokens.append(token)
                if token == eos_id:
                    break
            ids = tokens[1:]
            if ids and ids[-1] == eos_id:
                ids = ids[:-1]
            outputs.append(ids)
        return outputs
