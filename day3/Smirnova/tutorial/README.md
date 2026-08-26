# Day 3 tutorial — hands-on with language models (13:30–15:00, Nuvolos)

Anna Smirnova · AI for Economics and Finance, Torino, 26 August 2026

Four notebooks, ~90 minutes. Everything runs on CPU. Open each notebook, run top to bottom, stop at the questions.

| time | notebook | where | what you do |
|---|---|---|---|
| 13:30–13:55 | `00_warmup_language_models.ipynb` | `day3/Smirnova/tutorial/` | tokenize 150 real 10-K risk sections; build unigram/bigram/trigram models and compare perplexities (468 → 71 → 41); sample from them; read GPT-2's next-token table on the lecture sentence and vary temperature; **the rookie mistake**: raw BERT random-pair cosine 0.72 → 0.22 (Sentence-BERT) → 0.00 (all-but-the-top) |
| 13:55–14:20 | `01_babymodels.ipynb` | `day3/Smirnova/tutorial/` (copy of Leippold's `babymodels_executed`) | train three tiny models from scratch in 18 s — Bengio's neural LM, a word2vec-style pooled model, one transformer block — on a six-sentence corpus. Run §1–4, then §5.1 (distribution vs argmax) and §8 (read the attention weights). Skip §6.1 and §7. |
| 14:20–14:50 | `02_inside_bert.ipynb` | `day3/Smirnova/tutorial/` (copy of Leippold's `Inside_BERT_From_Intuition_to_Tensors`) | open BERT: WordPiece tokens, the three summed embeddings, attention heads with `bertviz` (Lecture 1, part 5), then §3.1–3.4 rebuild one layer by hand and check `== Hugging Face: True`. Finish with Part 4: 83 % sentiment accuracy from frozen features + logistic regression (Lecture 1, "what each one was used for"). Skip §2.3 and Part 5. |
| 14:50–15:00 | `03_rlhf_dpo_lab.ipynb` (optional) | `day3/Smirnova/tutorial/` (copy of Leippold's `rlhf_dpo_lab`) | exact RLHF/DPO on a toy world in 10 s: §1–4 (the Gibbs tilt, β prices the KL) and §5 (the Goodhart curve). Lecture 2, part 4, made numerical. Skip §7b/7c/§8. |

## Before class (lecturer)

- On the Nuvolos image, run `00_warmup` and `02_inside_bert` once so the model weights are cached (`gpt2`, `bert-base-uncased`, `distilbert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2`; ~1.2 GB). Cold, `Inside_BERT` spends 7 minutes downloading.
- Packages beyond the standard image: `transformers`, `sentence-transformers`, `bertviz`, `scikit-learn`. `00_warmup` needs no data download: the corpus is in `data/risk_factors_2020.csv.gz` (150 filings, Item 1A, from the EDGAR-CORPUS dataset, 2020 test split).
- Have `00_warmup` open on the projector at 13:30 with outputs already rendered; run live only the cells with questions.

## Not used, and why

- `00_climb_the_ladder.ipynb`, `01_calibration_changes_the_decision.ipynb` — need `teaching-data/*.csv` that are not in the repository (upstream link is dead). Outputs are saved, so they can be *shown*, not run.
- `bengio_rung_gpu.ipynb` — the ablation experiment needs a GPU; the CPU configuration fails in §7.
- `latent_state_rung.ipynb` — works (HMM on 10-Ks) but takes 3.5 minutes on the SEC download and is off the lecture's arc.
- `02_point_in_time_rag_sec.ipynb` — retrieval; Leippold's afternoon session.
- The `.html` labs are copied with clean names into `tutorial/labs/` (`shannon-bigram`, `rnn-machine`, `scaling-law-lab`, `lora-rank-lab`, `calibration-lab`, `dpo-preference-lab`). Self-contained, run offline, referenced from the lecture slides; open them in tabs before the lectures.

## If time runs short

Drop `rlhf_dpo_lab` first, then `babymodels` §8. `00_warmup` part 4 (anisotropy) and `Inside_BERT` Part 4 (the sentiment probe) are the two things participants will actually reuse; protect those.

## Regenerating the warm-up notebook

`make_warmup.py` writes `00_warmup_language_models.ipynb`; execute with `jupyter nbconvert --to notebook --execute --inplace 00_warmup_language_models.ipynb`.
