# English–Hindi machine translation

Comparison of an encoder–decoder Transformer trained from scratch, fine-tuned mT5, and LoRA-adapted decoder-only models (Gemma now, LLaMA when access is granted) on the same English-to-Hindi test set.

## Problem statement and objective

The task is to build four machine-translation systems for one language pair and compare them under a shared data split and a shared decoding setup.

- **Basic Transformer:** encoder–decoder, trained from scratch.
- **mT5:** encoder–decoder, fine-tuned.
- **LLaMA:** decoder-only. The training code is included. Weights stay gated until the model authors approve access, so this model is not in the score table yet.
- **Gemma:** decoder-only, LoRA fine-tuned.

The question is not only which system scores highest. It is how architecture, pretraining, tokenizer fertility, parameter count, and adaptation method show up in translation quality, training cost, and inference cost.

Official quality uses beam search. A separate temperature sweep is reported because sampling temperature changes faithfulness.

## Dataset description and source link

The corpus is the IIT Bombay English–Hindi Parallel Corpus (Kunchukuttan, Mehta, and Bhattacharyya, LREC 2018).

- Corpus page: https://www.cfilt.iitb.ac.in/iitb_parallel/
- Download mirror: https://huggingface.co/datasets/cfilt/iitb-english-hindi
- License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). This project is a noncommercial excerpt. See `data/DATA_SOURCE.md`.

Cleaning drops empty pairs, exact duplicates, pairs that are not English–Hindi, pairs longer than 40 words or 240 characters, and pairs whose word-count ratio is above 4. That length-ratio filter removes many misaligned pairs that would teach a model to add or delete content.

The random seed is 42.

| Split | Source | Size |
| --- | --- | --- |
| Train | Official training split, shuffled | 8,000 |
| Validation | Next slice of that shuffled training split | 1,000 |
| Test | Official held-out test split, shuffled | 1,000 |

Every model trains on the same 8,000 pairs and is scored on the same 1,000 test sentences. The exact pairs are written to `data/processed/` when the data stage runs.

## Models

| Model | Architecture | Adaptation | Checkpoint name |
| --- | --- | --- | --- |
| Basic Transformer | 3-layer pre-norm encoder–decoder, 4 heads, d_model 256, shared SentencePiece vocabulary of 8,000 | Trained from scratch with token cross-entropy and label smoothing 0.1 | Implemented in `src/transformer.py` |
| mT5-small | Pretrained encoder–decoder (~300M) | Full fine-tune for 1 epoch on `translate English to Hindi: {sentence}` | `google/mt5-small` |
| Gemma 2 2B Instruct | Pretrained decoder-only | 4-bit QLoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj` (rank 8) for 1 epoch | `google/gemma-2-2b-it` |
| LLaMA 3.2 1B Instruct | Pretrained decoder-only | The same LoRA recipe and the same user prompt as Gemma | `meta-llama/Llama-3.2-1B-Instruct` |

The from-scratch model uses pre-norm residuals so a small network remains stable on this subset. It still has token embeddings, sinusoidal positions, scaled dot-product attention, causal masking in the decoder, and a tied output layer.

Gemma and LLaMA see the same user instruction: “Translate this English sentence into Hindi. Reply with only the Hindi translation.” Each model’s own chat template wraps that instruction. Loss is applied only to the Hindi answer tokens.

Reported training times and inference times are measured on one GPU. Inference time is batch size 1 on 100 test sentences after a 2-sentence warmup. Quality uses beam size 4 on all 1,000 test sentences.

The decision rule is fixed before looking at scores: highest chrF, then BLEU, then the smaller parameter count.

## Execution steps and dependencies

```bash
pip install -r requirements.txt
python -m src.pipeline
```

Stages can be run on their own: `data`, `transformer`, `mt5`, `gemma`, `llama`, `evaluate`, `report`.

Gemma is gated. Export `HF_TOKEN` for an account that has accepted the Gemma license before the Gemma stage. LLaMA uses the same variable. If LLaMA access is still pending, that stage records the status and the rest of the pipeline continues.

The run that produced the numbers below used a free GPU notebook with internet enabled. `scripts/push_kaggle.py` submits that notebook from a machine that already has a Kaggle token in `~/.kaggle/access_token` and a Hugging Face token in the local Hugging Face login. The token files are not part of this repository.

Useful outputs:

- `results/metrics.json` — chrF, BLEU, parameter counts, train time, inference time, fertility, temperature, ethics probes
- `results/samples.md` — 25 side-by-side translations
- `results/*.png` — score, size, speed, temperature, and Transformer loss plots
- `checkpoints/` — local weights after training (not committed; they are large and can be rebuilt)

## Results obtained

<!-- AUTO-RESULTS:START -->

The results block is filled by `python -m src.report` after evaluation.

<!-- AUTO-RESULTS:END -->

Plots, when present: `results/scores.png`, `results/params_vs_chrf.png`, `results/inference_time.png`, `results/temperature.png`, `results/transformer_loss.png`.

## Model comparison and evaluation

Three comparisons are kept separate so a tiny from-scratch network is not treated as the same kind of evidence as a pretrained model.

1. **Pretraining, same family.** The basic Transformer and mT5 are both encoder–decoder. mT5 has already been pretrained on a large multilingual corpus. The from-scratch model has not. A higher mT5 score is evidence about pretraining and scale, not about the encoder–decoder layout by itself.
2. **Encoder–decoder versus decoder-only after adaptation.** mT5 is fully fine-tuned. Gemma is adapted with LoRA on a much larger decoder-only model. Parameter count and Hindi tokens per sentence (fertility) are reported beside the scores because the tokenizers are not the same. mT5’s sentencepiece vocabulary covers Devanagari directly. A decoder-only model with a byte-heavy Hindi tokenization spends more of its context on script and less on words.
3. **Matched decoder-only pair.** LLaMA uses the same prompt, the same LoRA rank, and the same number of training pairs as Gemma. That pair is the clean architecture-family comparison, and it is run only after LLaMA access is approved.

chrF is the primary score because it is less harsh than BLEU on Hindi morphology. BLEU is the secondary score, computed by sacreBLEU with character tokenization so Devanagari is not forced through an English tokenizer. Both signatures are stored in `results/metrics.json`.

## Sampling-temperature analysis

Beam search (size 4, no sampling) is the official decoder. Temperature is measured separately on the first 100 test sentences at 0.3, 0.7, 1.0, and 1.3, with one sample per sentence and nucleus cutoff 1.0 so temperature is the only randomness control.

Translation should be faithful, so chrF is expected to fall as temperature rises. The deployed setting remains beam search. Higher temperature is the wrong control when the cost of an added or dropped fact is high. The curve is in `results/temperature.png` and in the results table above.

## Responsible and ethical use

The probes in `data/ethics_probes.json` are not part of the BLEU corpus. They ask each system about gender in occupations, an English idiom, a religious and cultural reference, a date and number, negation, a person’s name, a medical fact, a political sentence, and a short sentence that should not grow extra claims. The translations are listed in the results section.

Built-in controls in this project:

- The length-ratio filter removes many misaligned training pairs, which is a direct way to reduce learned additions and deletions.
- Official decoding is low-randomness beam search. The temperature table is the evidence for keeping sampling off in a translation deployment.
- Medical, legal, and political sentences are not a safe automatic output. A person should review them before they are used.
- Gendered occupation sentences are kept as a standing test, not as something BLEU will catch.
- Names and numbers are checked by reading the probe output, because a high corpus chrF can still drop “12 March 2024” or “102 degrees”.

A high chrF does not certify that a system is fair or safe. The probe set is small and qualitative.

## Conclusion

The best completed model is the one named by the pre-registered rule in the results section: highest chrF, then BLEU, then fewer parameters. LLaMA is excluded until access is approved. After approval, `python -m src.pipeline --stages llama evaluate report` reruns that model on the frozen split and rebuilds this section.

## Repository layout

```text
configs/experiment.yaml    shared sizes, seed, and decoding settings
src/transformer.py         from-scratch encoder–decoder
src/train_transformer.py   training loop for that model
src/hf_experiment.py       mT5 fine-tune and LoRA for Gemma and LLaMA
src/evaluate.py            chrF, BLEU, timing, temperature, probes
src/data.py                download, clean, and split
notebooks/analysis.ipynb   loads the saved metrics and plots
data/DATA_SOURCE.md        corpus citation and license
```
