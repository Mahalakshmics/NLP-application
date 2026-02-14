# Statistical Machine Translation (EN→HI) — IIT Bombay Corpus
**Phrase-based SMT (IBM Model 1 + Phrase Extraction + Bigram LM) with Streamlit Frontend**

This project implements a minimal end-to-end Statistical Machine Translation (SMT) pipeline trained on the **IIT Bombay English–Hindi parallel corpus**, along with a **Streamlit UI** to display corpus statistics, learned probabilities, and interactive translations.

---

## 1) Project Structure

Recommended folder structure:

```
NLP/
  .venv/
  src/
    data/
        train.en
        train.hi
        test.en
        test.hi
        test_short.en
    smt.py
    app.py
    prepare_dataset.py
    models/
      smt_iitb_500k.pkl   (created after saving)
```

> Run commands from the **NLP** folder unless specified.

---

## 2) Environment Setup

### Create and activate a virtual environment (if not already)

```bash
cd ~/Desktop/NLP
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -U pip
pip install datasets tqdm streamlit
```

---

## 3) Prepare the IIT Bombay EN–HI Corpus

This step downloads the dataset and creates the parallel text files:

- `data/train.en`, `data/train.hi`
- `data/test.en`, `data/test.hi`

Run:

```bash
cd src
python prepare_dataset.py
```

Verify files exist and line counts match:

```bash
wc -l data/train.en data/train.hi
wc -l data/test.en data/test.hi
```

### Create a short test set for nicer demo outputs (recommended)

```bash
awk 'NF>=3 && NF<=10 {print}' data/test.en | head -n 10 > data/test_short.en
```

---

## 4) Run the Backend (CLI) — Optional

You can run the SMT system from the terminal for logs/output:

```bash
cd ~/Desktop/NLP/src
python smt.py \
  --train-src data/train.en --train-tgt data/train.hi \
  --test-src data/test_short.en \
  --max-train 500000 --ibm-iters 10 \
  --phrase-build-pairs 300000 \
  --lambda-lm 1.2 --beam 20 --cand-per-phrase 10
```

This prints:

- Number of sentence pairs
- Vocabulary sizes
- Top learned lexical translation probabilities `t(hi|en)`
- Phrase-table size and top phrase translation probs `φ(hi|en)`
- Sample translations + TM/LM scores
- Sample word alignments

---

## 5) Run the Frontend (Streamlit UI)

Start the UI:

```bash
cd ~/Desktop/NLP/src
streamlit run app.py
```

Open the URL shown in the terminal (typically):

- `http://localhost:8501`

### Recommended UI workflow

1. Click **Train Model** (first time).
2. After training completes, click **Save Model** to:  
   `src/models/smt_iitb_500k.pkl`
3. Next time, click **Load Model** for instant startup (no retraining).

---

## 6) Notes / Common Issues

### “python: can't open file .../NLP/smt.py”
You must run from `src/` because `smt.py` is inside `src`:

```bash
cd ~/Desktop/NLP/src
python smt.py ...
```

### Training takes time
- IBM Model 1 + phrase extraction on 500k pairs is compute-heavy.
- The UI supports **save/load** so you train once and reuse.

### Translation quality
This is a baseline SMT:

- monotone decoding (no reordering)
- bigram LM

So long sentences may look ungrammatical; short sentences are better.

---

## 7) Suggested Screenshots for Report

1. `ls -lah data/` showing generated files
2. `streamlit run app.py` terminal output
3. UI before training (buttons visible)
4. UI after training: sentence pairs + vocab sizes
5. UI showing top `t(hi|en)` probabilities
6. UI translation box with output + scores
7. “Save Model” confirmation
8. “Load Model” confirmation (fast load)

---

## 8) Deliverables

- `src/smt.py` — SMT backend (IBM1 + phrase extraction + LM + decoder)
- `src/app.py` — Streamlit frontend
- `src/prepare_dataset.py` — downloads and prepares IITB corpus
- `README.md` — this file
- `report.md` — brief report (design choices + challenges + integration)
- `screenshots/` — flow screenshots for submission
