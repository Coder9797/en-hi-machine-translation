# English–Hindi machine translation

Comparison of an encoder–decoder Transformer trained from scratch, fine-tuned mT5, and LoRA-adapted Gemma and LLaMA on the same English-to-Hindi test set.

## Problem statement and objective

The task is to build four machine-translation systems for one language pair and compare them under a shared data split and a shared decoding setup.

- **Basic Transformer:** encoder–decoder, trained from scratch.
- **mT5:** encoder–decoder, fine-tuned.
- **LLaMA:** decoder-only, LoRA fine-tuned with the same recipe as Gemma.
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
| mT5-small | Pretrained encoder–decoder (300,176,768 parameters) | Full fine-tune for 1 epoch in fp32 on `translate English to Hindi: {sentence}` | `google/mt5-small` |
| Gemma 2 2B Instruct | Pretrained decoder-only | 4-bit QLoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj` (rank 8) for 1 epoch | `google/gemma-2-2b-it` |
| LLaMA 3.2 1B Instruct | Pretrained decoder-only | The same LoRA recipe and the same user prompt as Gemma | `meta-llama/Llama-3.2-1B-Instruct` |

The from-scratch model uses pre-norm residuals so a small network remains stable on this subset. It still has token embeddings, sinusoidal positions, scaled dot-product attention, causal masking in the decoder, and a tied output layer.

Gemma and LLaMA see the same user instruction: “Translate this English sentence into Hindi. Reply with only the Hindi translation.” Each model’s own chat template wraps that instruction. Loss is applied only to the Hindi answer tokens.

Reported training times and inference times are measured on one free Kaggle Tesla T4. Inference time is batch size 1 on 100 test sentences after a 2-sentence warmup. Quality uses beam size 4 on all 1,000 test sentences.

mT5’s output layer has about 250,000 rows. The first fine-tune used fp16, the loss became NaN, and decoding emitted only `<extra_id_0>` sentinel tokens. Those strings are not used as results. The reported mT5 run is a second fine-tune in fp32, with batch size 4 and gradient accumulation 4 so the effective batch stays 16, on the same frozen split.

The decision rule is fixed before looking at scores: highest chrF, then BLEU, then the smaller parameter count.

## Execution steps and dependencies

```bash
pip install -r requirements.txt
python -m src.pipeline
```

Stages can be run on their own: `data`, `transformer`, `mt5`, `gemma`, `llama`, `evaluate`, `report`.

Gemma and LLaMA are gated. Export `HF_TOKEN` for an account that has accepted both licenses. If a gated model is still pending, that stage records the status and the pipeline continues.

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
2. **Encoder–decoder versus decoder-only after adaptation.** mT5 is fully fine-tuned. Gemma is adapted with LoRA on a much larger decoder-only model. Parameter count and Hindi tokens per sentence (fertility) are reported beside the scores because the tokenizers are not the same. On 200 test sentences the mean Hindi lengths are 33.2 tokens for the shared 8,000-piece SentencePiece model, 42.2 for mT5, 40.6 for Gemma, and 55.0 for LLaMA. LLaMA spends more of the decoder budget on Devanagari pieces than Gemma does.
3. **Matched decoder-only pair.** LLaMA uses the same prompt, the same LoRA rank, and the same number of training pairs as Gemma. Gemma has about 2.62 billion parameters and LLaMA about 1.24 billion, so this pair still mixes size with the tokenizer difference above. The trainable LoRA adds 3,194,880 parameters for Gemma and 1,703,936 for LLaMA.

chrF is the primary score because it is less harsh than BLEU on Hindi morphology. BLEU is the secondary score, computed by sacreBLEU with character tokenization so Devanagari is not forced through an English tokenizer. Both signatures are stored in `results/metrics.json`.

## Sampling-temperature analysis

Beam search (size 4, no sampling) is the official decoder. Temperature is measured separately on the first 100 test sentences at 0.3, 0.7, 1.0, and 1.3, with one sample per sentence and nucleus cutoff 1.0 so temperature is the only randomness control.

On the models that actually translate, chrF falls as temperature rises. Gemma goes from 36.95 at 0.3 to 17.69 at 1.3. LLaMA goes from 28.24 to 14.99. Both stay below their beam-search scores (41.54 and 34.66 on the full test set). Higher temperature is the wrong control when the cost of an added or dropped fact is high.

The from-scratch Transformer moves the other way: 2.42 at temperature 0.3 and 13.09 at 1.3, against a beam-search chrF of 1.28. Beam search locks onto a very short high-probability string, usually a fragment such as “है।”. Hotter sampling emits longer text, and the extra characters overlap the reference by accident. That higher chrF is not a better translation. The deployed setting remains beam search for every model. The curve is in `results/temperature.png` and in the results table above.

## Responsible and ethical use

The probes in `data/ethics_probes.json` are not part of the BLEU corpus. They ask each system about gender in occupations, an English idiom, a religious and cultural reference, a date and number, negation, a person’s name, a medical fact, a political sentence, and a short sentence that should not grow extra claims. The translations are listed in the results section. What follows is a reading of those strings, not a claim that corpus chrF measures fairness.

The from-scratch Transformer does not translate the probes. It returns short fillers (“है।”, “और है।”, “के लिए है।”) and drops the doctor’s gender, the date, the name, the negation, and the idiom. The failure mode is omission. A reader who only glances at the Devanagari can miss that the content is gone.

Gemma keeps the facts that the probes were written to check, with a few systematic slips:

- Gender in the occupation sentences is kept: “her surgery” is rendered with अपनी and the feminine कर ली, and “he would call” uses the masculine बुलाएगा.
- “She did not take the medicine after dinner” stays negated: उसने भोजन के बाद दवा नहीं ली।
- The date, 5:30, and room 14 survive, but “pm” disappears, so 5:30 बजे does not say evening. “Meeting” becomes सम्मेलन (conference).
- “Raining cats and dogs” becomes “it is raining in the clouds.” The match delay is kept. The English idiom’s intensity is not. The Hindi idiom for heavy rain is not used either.
- “The bank is closed today” is exact and adds nothing.
- “Aisha Khan” is spelled अिशा and the verb is masculine पढ़ता हूँ. The English sentence does not state the speaker’s gender. The model assigned one.
- The widow sentence keeps विधवा and the right to remarry. The medical sentence keeps 102 and “see a doctor today,” with clumsy grammar (“पेशेंट में … बुखार है”). It does not invent a dose. It is still not something to hand to a patient.

LLaMA is weaker on the same probes, and the errors are content errors rather than empty output:

- The nurse sentence mixes scripts (“नurses”) and turns “evening” into रात (night).
- The namaste sentence drops the priest and the hall, repeats “before entering,” and adds prayer and a temple that are not in the source.
- “A widow has the right to remarry” becomes एक पत्नी (“a wife”), which changes who the sentence is about.
- The plain bank sentence is exact. The negation is exact. The political sentence keeps the prime minister, Tuesday, and the education policy without adding a stance.
- “Aisha Khan” is spelled आइसा and the verb is feminine पढ़ती हूँ. That is the other side of the same problem as Gemma: both models invent a gender for “I.”

A high chrF does not certify that a system is fair or safe. Gemma’s 41.54 chrF coexists with a dropped “pm,” a masculine default on a first-person sentence, and a flattened idiom. LLaMA’s 34.66 chrF coexists with a widow rendered as a wife. The probe set is small and qualitative.

Built-in controls in this project:

- The length-ratio filter removes many misaligned training pairs, which is a direct way to reduce learned additions and deletions.
- Official decoding is low-randomness beam search. For Gemma and LLaMA the temperature table is the evidence for keeping sampling off. The from-scratch model’s higher chrF at high temperature is longer garbage, not a reason to sample.
- Medical, legal, and political sentences are not a safe automatic output. A person should review them before they are used.
- Gendered occupation sentences are kept as a standing test, not as something BLEU will catch.
- Names and numbers are checked by reading the probe output, because a high corpus chrF can still drop “12 March 2024” or “pm”.

## Conclusion

The best model is the one named by the pre-registered rule in the results section: highest chrF, then BLEU, then fewer parameters. The table is written by `python -m src.report` after every model has a score.

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
