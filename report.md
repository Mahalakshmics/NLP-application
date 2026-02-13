
```md
# Statistical Machine Translation (EN→HI) — Mini Project Report

## Overview
This project implements a minimal phrase-based Statistical Machine Translation (SMT) system trained on the IIT Bombay English–Hindi parallel corpus. The system follows the classical SMT pipeline: sentence alignment verification, IBM Model 1 word alignment, phrase extraction, phrase translation probabilities, a Hindi n-gram language model, and noisy-channel decoding.

## Design Choices
### Parallel corpus and preprocessing
- Dataset: IIT Bombay English–Hindi parallel corpus.
- Sentence alignment: the dataset is line-aligned; the implementation verifies equal line counts and drops empty lines.
- Tokenization: lightweight whitespace tokenization with punctuation separation. This keeps the pipeline simple and transparent for educational purposes.

### Word alignment: IBM Model 1
- IBM Model 1 was chosen because it is the simplest probabilistic word alignment model and can be implemented from scratch using EM.
- The model learns lexical translation probabilities t(f|e), which are later used to generate Viterbi word alignments for phrase extraction.

### Phrase extraction and phrase table
- Phrase pairs are extracted using Koehn-style alignment consistency constraints (phrases up to length 5).
- Phrase translation probabilities φ(f|e) are computed via relative frequency counts from extracted phrase pairs.

### Language model
- A Hindi bigram language model with add-one smoothing is used.
- The LM encourages fluent Hindi token sequences during decoding, although bigram context is limited compared to higher-order LMs.

### Decoding: noisy channel model
- The decoder uses a monotone beam search (left-to-right segmentation), scoring hypotheses with:
  - Translation model (TM): log φ(f|e)
  - Language model (LM): λ · log P_LM(f)
- A copy-through fallback is used for OOV tokens to avoid `<UNK>` outputs.

## Challenges Faced
1. **Computational cost on large corpora:** Training IBM1 and extracting phrases on 500k pairs is expensive.
   - Mitigation: phrase extraction is performed on a subset (e.g., 300k pairs).
   - The frontend supports model save/load so retraining is not required each run.

2. **Word order differences (English vs Hindi):** Monotone decoding cannot reorder phrases, so long sentences can appear scrambled.
   - This is a known limitation of monotone decoders and motivates distortion/reordering models in full SMT systems.

3. **Noise and mixed-script tokens:** Some corpus segments contain named entities and transliterated words, which can remain in English.
   - The system still produces meaningful phrase substitutions, but grammatical fluency depends on LM strength and decoding constraints.

## Integration (SMT model + Frontend)
- The SMT pipeline is implemented in `smt.py`.
- The Streamlit UI (`app.py`) trains or loads a saved model and exposes:
  - corpus statistics (pairs, vocabulary sizes)
  - top lexical probabilities t(f|e)
  - interactive translation box using noisy-channel decoding
- The model is cached and can be saved as `models/smt_iitb_500k.pkl` for fast reuse.

## Conclusion
The project demonstrates an end-to-end SMT pipeline and shows how corpus size, noise, and domain mismatch affect translation quality. While output quality is limited by monotone decoding and a bigram LM, the system successfully reproduces the essential components of classical phrase-based SMT.
