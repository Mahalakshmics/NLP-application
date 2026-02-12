#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smt.py — Minimal Phrase-Based SMT (EN→HI) pipeline
Implements:
- Sentence alignment (verification for line-aligned corpus)
- IBM Model 1 word alignment (EM)
- Phrase extraction (Koehn-style consistency)
- Phrase table probabilities φ(hi_phrase | en_phrase)
- Hindi bigram LM (add-one smoothing)
- Monotone noisy-channel decoding with beam search (TM + LM)

USAGE :
  python smt.py --train-src data/train.en --train-tgt data/train.hi \
                --test-src data/test.en --max-train 50000 --ibm-iters 8

"""
# ----------------------------
# Imports
# ----------------------------


import argparse
import math
import random
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable, Optional


# ----------------------------
# Tokenization (simple baseline)
# ----------------------------

_punct_re = re.compile(r"([.,!?;:\"()\[\]{}<>])")

def tokenize_en(s: str) -> List[str]:
    s = s.strip().lower()
    s = _punct_re.sub(r" \1 ", s)
    s = re.sub(r"\s+", " ", s)
    return s.split()

def tokenize_hi(s: str) -> List[str]:
    # Minimal: whitespace + light punctuation spacing
    s = s.strip()
    s = _punct_re.sub(r" \1 ", s)
    s = re.sub(r"\s+", " ", s)
    return s.split()

def detokenize(tokens: List[str]) -> str:
    # Simple detok: remove spaces before punctuation
    out = " ".join(tokens)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    out = re.sub(r"\(\s+", "(", out)
    out = re.sub(r"\s+\)", ")", out)
    out = re.sub(r"\[\s+", "[", out)
    out = re.sub(r"\s+\]", "]", out)
    return out


# ----------------------------
# Data loading + "sentence alignment" (verification)
# ----------------------------

def read_parallel(src_path: str, tgt_path: str, max_pairs: Optional[int]=None, seed: int=42) -> List[Tuple[str, str]]:
    with open(src_path, "r", encoding="utf-8") as fs:
        src_lines = [ln.rstrip("\n") for ln in fs]
    with open(tgt_path, "r", encoding="utf-8") as ft:
        tgt_lines = [ln.rstrip("\n") for ln in ft]

    n0, n1 = len(src_lines), len(tgt_lines)
    n = min(n0, n1)
    if n0 != n1:
        print(f"[WARN] Line count mismatch: src={n0}, tgt={n1}. Truncating to {n}.")

    pairs = []
    for i in range(n):
        s, t = src_lines[i].strip(), tgt_lines[i].strip()
        if not s or not t:
            continue
        pairs.append((s, t))

    # Optional subsampling for speed
    if max_pairs is not None and len(pairs) > max_pairs:
        random.seed(seed)
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]

    return pairs


# ----------------------------
# Vocabulary stats
# ----------------------------

def vocab_stats(pairs_tok: List[Tuple[List[str], List[str]]]) -> Tuple[int, int, Counter, Counter]:
    src_vocab = Counter()
    tgt_vocab = Counter()
    for src_toks, tgt_toks in pairs_tok:
        src_vocab.update(src_toks)
        tgt_vocab.update(tgt_toks)
    return len(src_vocab), len(tgt_vocab), src_vocab, tgt_vocab


# ----------------------------
# IBM Model 1 (t(f|e)) with EM
# ----------------------------

NULL = "<NULL>"

def build_cooc(pairs_tok: List[Tuple[List[str], List[str]]],
               max_cooc_per_e: int = 2000) -> Dict[str, set]:
    """
    Build co-occurrence sets: for each e, which f words co-occur with it.
    max_cooc_per_e caps memory.
    """
    cooc = defaultdict(set)
    for en_toks, hi_toks in pairs_tok:
        en = [NULL] + en_toks
        # Optionally unique target tokens for cooc set growth control
        for e in set(en):
            s = cooc[e]
            if len(s) >= max_cooc_per_e:
                continue
            for f in set(hi_toks):
                if len(s) >= max_cooc_per_e:
                    break
                s.add(f)
    return cooc

def init_t_from_cooc(cooc: Dict[str, set]) -> Dict[str, Dict[str, float]]:
    t = {}
    for e, fs in cooc.items():
        if not fs:
            continue
        uni = 1.0 / len(fs)
        t[e] = {f: uni for f in fs}
    return t

def ibm1_train(pairs_tok: List[Tuple[List[str], List[str]]],
               iters: int = 8,
               max_cooc_per_e: int = 2000,
               prune_min_prob: float = 1e-6,
               verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Train IBM Model 1 t(f|e) by EM.
    - Stores only observed co-occurrences from initialisation.
    - Prunes tiny probs after each iteration for memory control.
    """
    if verbose:
        print("[IBM1] Building co-occurrence sets...")
    cooc = build_cooc(pairs_tok, max_cooc_per_e=max_cooc_per_e)
    t = init_t_from_cooc(cooc)

    # Ensure NULL exists
    if NULL not in t:
        t[NULL] = {}

    for it in range(1, iters + 1):
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)

        # E-step
        for en_toks, hi_toks in pairs_tok:
            en = [NULL] + en_toks
            # For speed, work with unique tokens per sentence in EN to reduce repeated sums
            for f in hi_toks:
                # Compute normalization
                z = 0.0
                for e in en:
                    z += t.get(e, {}).get(f, 0.0)
                if z == 0.0:
                    # Backoff: if unseen, skip (or uniform among en)
                    continue
                for e in en:
                    tef = t.get(e, {}).get(f, 0.0)
                    if tef == 0.0:
                        continue
                    delta = tef / z
                    count[e][f] += delta
                    total[e] += delta

        # M-step
        for e, fdict in count.items():
            denom = total[e]
            if denom == 0.0:
                continue
            new_fdict = {}
            for f, c in fdict.items():
                p = c / denom
                if p >= prune_min_prob:
                    new_fdict[f] = p
            t[e] = new_fdict

        if verbose:
            # quick progress metric: avg support size
            avg_sup = sum(len(v) for v in t.values()) / max(1, len(t))
            print(f"[IBM1] Iter {it}/{iters} done. Entries={sum(len(v) for v in t.values())} | Avg support/e={avg_sup:.1f}")

    return t

def top_translations(t: Dict[str, Dict[str, float]], e: str, k: int=10) -> List[Tuple[str, float]]:
    cand = t.get(e, {})
    return sorted(cand.items(), key=lambda x: x[1], reverse=True)[:k]

def align_sentence_ibm1(en_toks: List[str], hi_toks: List[str], t: Dict[str, Dict[str, float]]) -> List[Tuple[int, int]]:
    """
    Returns alignment links as (i, j) where i indexes EN tokens (0..l-1), j indexes HI tokens (0..m-1)
    NULL is considered but not returned in links.
    """
    en_with_null = [NULL] + en_toks
    links = []
    for j, f in enumerate(hi_toks):
        best_i = 0
        best_p = t.get(NULL, {}).get(f, 0.0)
        for i in range(1, len(en_with_null)):
            e = en_with_null[i]
            p = t.get(e, {}).get(f, 0.0)
            if p > best_p:
                best_p = p
                best_i = i
        if best_i != 0:
            links.append((best_i - 1, j))
    return links


# ----------------------------
# Phrase extraction (Koehn-style consistency)
# ----------------------------

def phrase_extract(en_toks: List[str],
                   hi_toks: List[str],
                   links: List[Tuple[int, int]],
                   max_len: int = 5) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Returns list of phrase span pairs: ((i1,i2), (j1,j2)) inclusive spans
    Standard consistency:
      - all alignment points within source span map inside target span
      - all alignment points within target span map inside source span
    """
    l, m = len(en_toks), len(hi_toks)
    A = set(links)

    # build maps
    src_to_tgt = defaultdict(set)
    tgt_to_src = defaultdict(set)
    for i, j in A:
        src_to_tgt[i].add(j)
        tgt_to_src[j].add(i)

    extracted = []

    for i1 in range(l):
        for i2 in range(i1, min(l, i1 + max_len)):
            # collect all target positions aligned to any source in [i1,i2]
            js = set()
            for i in range(i1, i2 + 1):
                js |= src_to_tgt.get(i, set())
            if not js:
                continue
            j1, j2 = min(js), max(js)
            if (j2 - j1 + 1) > max_len:
                continue

            # consistency check 1: source span alignments stay within target span
            ok = True
            for i in range(i1, i2 + 1):
                for j in src_to_tgt.get(i, set()):
                    if j < j1 or j > j2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # consistency check 2: target span alignments stay within source span
            for j in range(j1, j2 + 1):
                for i in tgt_to_src.get(j, set()):
                    if i < i1 or i > i2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            extracted.append(((i1, i2), (j1, j2)))

    return extracted

def build_phrase_table(pairs_tok: List[Tuple[List[str], List[str]]],
                       t_ibm1: Dict[str, Dict[str, float]],
                       max_phrase_len: int = 5,
                       max_pairs: Optional[int] = None,
                       verbose: bool = True) -> Dict[str, List[Tuple[str, float]]]:
    """
    Builds φ(hi_phrase | en_phrase) using extracted phrase pairs.
    Returns: phrase_table[en_phrase] = [(hi_phrase, prob), ...] sorted desc
    """
    phrase_counts = defaultdict(Counter)

    if max_pairs is None:
        data = pairs_tok
    else:
        data = pairs_tok[:max_pairs]

    for idx, (en_toks, hi_toks) in enumerate(data, 1):
        links = align_sentence_ibm1(en_toks, hi_toks, t_ibm1)
        spans = phrase_extract(en_toks, hi_toks, links, max_len=max_phrase_len)
        for (i1, i2), (j1, j2) in spans:
            e_phrase = " ".join(en_toks[i1:i2+1])
            f_phrase = " ".join(hi_toks[j1:j2+1])
            phrase_counts[e_phrase][f_phrase] += 1

        if verbose and idx % 10000 == 0:
            print(f"[PHRASE] Processed {idx} sentence pairs...")

    # Convert counts to conditional probabilities φ(f|e)
    phrase_table = {}
    for e_phrase, ctr in phrase_counts.items():
        total = sum(ctr.values())
        if total == 0:
            continue
        items = [(f_phrase, c / total) for f_phrase, c in ctr.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        phrase_table[e_phrase] = items

    if verbose:
        print(f"[PHRASE] Phrase table entries (unique EN phrases): {len(phrase_table)}")
    return phrase_table


# ----------------------------
# Hindi Bigram LM (add-one smoothing)
# ----------------------------

BOS = "<s>"
EOS = "</s>"

class BigramLM:
    def __init__(self):
        self.unigram = Counter()
        self.bigram = Counter()
        self.vocab = set()

    def train(self, sentences_tok: List[List[str]]):
        for toks in sentences_tok:
            seq = [BOS] + toks + [EOS]
            for w in seq:
                self.unigram[w] += 1
                self.vocab.add(w)
            for i in range(len(seq) - 1):
                self.bigram[(seq[i], seq[i+1])] += 1
        self.vocab.add(BOS)
        self.vocab.add(EOS)

    def logprob_next(self, prev: str, w: str) -> float:
        V = len(self.vocab)
        num = self.bigram[(prev, w)] + 1
        den = self.unigram[prev] + V
        return math.log(num / den)

    def logprob_sentence(self, toks: List[str]) -> float:
        seq = [BOS] + toks + [EOS]
        lp = 0.0
        for i in range(len(seq) - 1):
            lp += self.logprob_next(seq[i], seq[i+1])
        return lp


# ----------------------------
# Decoder (monotone phrase-based beam search)
# ----------------------------

def get_candidates(phrase_table: Dict[str, List[Tuple[str, float]]],
                   e_phrase: str,
                   topn: int = 5) -> List[Tuple[List[str], float]]:
    """
    Returns candidate Hindi token lists with log φ(f|e).
    """
    items = phrase_table.get(e_phrase)
    if not items:
        return []
    out = []
    for f_phrase, p in items[:topn]:
        out.append((f_phrase.split(), math.log(max(p, 1e-12))))
    return out

def decode_monotone(en_toks: List[str],
                    phrase_table: Dict[str, List[Tuple[str, float]]],
                    lm: BigramLM,
                    max_phrase_len: int = 5,
                    beam: int = 8,
                    cand_per_phrase: int = 5,
                    lambda_lm: float = 0.5) -> Tuple[List[str], float, float, float]:
    """
    Returns (best_hi_tokens, tm_log, lm_log, total_score)
    total_score = tm_log + lambda_lm * lm_log
    """
    # Hypothesis: (pos, hi_tokens, tm_log, lm_log)
    hyps = [(0, [], 0.0, 0.0)]

    while True:
        # Check if all complete
        if all(pos == len(en_toks) for (pos, _, _, _) in hyps):
            break

        new_hyps = []
        for pos, out_toks, tm_log, lm_log in hyps:
            if pos == len(en_toks):
                # keep completed
                new_hyps.append((pos, out_toks, tm_log, lm_log))
                continue

            # Expand phrases of length 1..max_phrase_len
            expanded = False
            for L in range(1, max_phrase_len + 1):
                if pos + L > len(en_toks):
                    break
                e_phrase = " ".join(en_toks[pos:pos+L])
                cands = get_candidates(phrase_table, e_phrase, topn=cand_per_phrase)
                if not cands:
                    continue
                expanded = True
                for f_toks, logphi in cands:
                    # Update LM incrementally
                    # compute additional lm logprob for appending f_toks
                    inc_lm = incremental_lm_log(lm, out_toks, f_toks)
                    new_hyps.append((pos + L, out_toks + f_toks, tm_log + logphi, lm_log + inc_lm))

            if not expanded:
                # Fallback: copy source token or output UNK token (Hindi output baseline)
                # This keeps decoder moving even if phrase missing.
                # You can change "<UNK>" to the English token if desired.
                f_toks = [en_toks[pos]]
                inc_lm = incremental_lm_log(lm, out_toks, f_toks)
                new_hyps.append((pos + 1, out_toks + f_toks, tm_log + math.log(1e-12), lm_log + inc_lm))

        # Prune beam by total score
        new_hyps.sort(key=lambda h: h[2] + lambda_lm * h[3], reverse=True)
        hyps = new_hyps[:beam]

    # Finalize with EOS contribution
    best = max(hyps, key=lambda h: h[2] + lambda_lm * (h[3] + finalize_lm_eos(lm, h[1])))
    pos, out_toks, tm_log, lm_log = best
    lm_log = lm_log + finalize_lm_eos(lm, out_toks)
    total = tm_log + lambda_lm * lm_log
    return out_toks, tm_log, lm_log, total

def incremental_lm_log(lm: BigramLM, prefix: List[str], append: List[str]) -> float:
    """
    LM logprob added by appending tokens 'append' after 'prefix', without EOS.
    """
    if not append:
        return 0.0
    inc = 0.0
    prev = BOS if len(prefix) == 0 else prefix[-1]
    for w in append:
        inc += lm.logprob_next(prev, w)
        prev = w
    return inc

def finalize_lm_eos(lm: BigramLM, toks: List[str]) -> float:
    prev = BOS if len(toks) == 0 else toks[-1]
    return lm.logprob_next(prev, EOS)


# ----------------------------
# Utilities: show top translations, show phrase table samples
# ----------------------------

def print_top_ibm1(t: Dict[str, Dict[str, float]], words: List[str], k: int = 10):
    for w in words:
        w = w.lower()
        items = top_translations(t, w, k=k)
        if not items:
            continue
        print(f'\n[TOP t(hi|en)] "{w}":')
        for f, p in items:
            print(f"  {f}\t{p:.4f}")

def print_phrase_samples(phrase_table: Dict[str, List[Tuple[str, float]]], phrases: List[str], k: int = 5):
    for p in phrases:
        items = phrase_table.get(p.lower())
        if not items:
            continue
        print(f'\n[TOP φ(hi|en)] "{p}":')
        for f, prob in items[:k]:
            print(f"  {f}\t{prob:.4f}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-src", required=True, help="Path to train.en")
    ap.add_argument("--train-tgt", required=True, help="Path to train.hi")
    ap.add_argument("--test-src", default=None, help="Optional path to test.en for demo translations")
    ap.add_argument("--max-train", type=int, default=50000, help="Max training sentence pairs to use")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ibm-iters", type=int, default=8)
    ap.add_argument("--max-cooc-per-e", type=int, default=2000)
    ap.add_argument("--prune-min-prob", type=float, default=1e-6)
    ap.add_argument("--max-phrase-len", type=int, default=5)
    ap.add_argument("--phrase-build-pairs", type=int, default=30000, help="How many pairs to use for phrase extraction (subset for speed)")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--cand-per-phrase", type=int, default=5)
    ap.add_argument("--lambda-lm", type=float, default=0.5)
    ap.add_argument("--interactive", action="store_true", help="Interactive translation mode")
    args = ap.parse_args()

    random.seed(args.seed)

    # 1) Load and "sentence alignment" verification (line-aligned)
    pairs_raw = read_parallel(args.train_src, args.train_tgt, max_pairs=args.max_train, seed=args.seed)
    print("\n=== System Requirements Output ===")
    print(f"Number of sentence pairs (after cleaning): {len(pairs_raw)}")

    # Tokenize
    pairs_tok = []
    for en, hi in pairs_raw:
        en_toks = tokenize_en(en)
        hi_toks = tokenize_hi(hi)
        if en_toks and hi_toks:
            pairs_tok.append((en_toks, hi_toks))

    # 2) Vocab sizes
    v_en, v_hi, en_freq, hi_freq = vocab_stats(pairs_tok)
    print(f"Vocabulary size (EN): {v_en}")
    print(f"Vocabulary size (HI): {v_hi}")

    # 3) Train IBM1
    print("\n=== Training IBM Model 1 (Word Alignment) ===")
    t_ibm1 = ibm1_train(
        pairs_tok,
        iters=args.ibm_iters,
        max_cooc_per_e=args.max_cooc_per_e,
        prune_min_prob=args.prune_min_prob,
        verbose=True
    )

    # 4) Show top learned translation probabilities (required)
    # choose top frequent EN words excluding punctuation-like tokens
    common_en = [w for (w, _) in en_freq.most_common(50) if w.isalpha()]
    sample_words = common_en[:8] if len(common_en) >= 8 else common_en
    print("\n=== Top Learned Translation Probabilities (t(hi|en)) ===")
    print_top_ibm1(t_ibm1, sample_words, k=10)

    # 5) Build phrase table
    print("\n=== Phrase Extraction + Phrase Table (φ(hi|en)) ===")
    phrase_table = build_phrase_table(
        pairs_tok,
        t_ibm1,
        max_phrase_len=args.max_phrase_len,
        max_pairs=args.phrase_build_pairs,
        verbose=True
    )

    # Show phrase samples using common unigrams/bigrams from EN side
    common_phrases = []
    for w in sample_words[:5]:
        common_phrases.append(w)
    # Add a couple common bigrams
    if pairs_tok:
        bigram_ctr = Counter()
        for en_toks, _ in pairs_tok[:20000]:
            for i in range(len(en_toks)-1):
                bg = en_toks[i] + " " + en_toks[i+1]
                if all(tok.isalpha() for tok in bg.split()):
                    bigram_ctr[bg] += 1
        common_phrases += [bg for (bg, _) in bigram_ctr.most_common(3)]
    print_phrase_samples(phrase_table, common_phrases, k=5)

    # 6) Train Hindi Bigram LM
    print("\n=== Training Hindi Bigram Language Model (LM) ===")
    lm = BigramLM()
    hi_sents = [hi for _, hi in pairs_tok]
    lm.train(hi_sents)
    print(f"LM vocab size (HI + BOS/EOS): {len(lm.vocab)}")

    # 7) Demo translation (noisy channel: TM + λLM)
    def translate_sentence(en_sent: str):
        en_toks = tokenize_en(en_sent)
        out_toks, tm_log, lm_log, total = decode_monotone(
            en_toks=en_toks,
            phrase_table=phrase_table,
            lm=lm,
            max_phrase_len=args.max_phrase_len,
            beam=args.beam,
            cand_per_phrase=args.cand_per_phrase,
            lambda_lm=args.lambda_lm
        )
        print("\n--- Translation ---")
        print(f"SRC: {en_sent}")
        print(f"HYP: {detokenize(out_toks)}")
        print(f"TM log-score: {tm_log:.3f}")
        print(f"LM log-score: {lm_log:.3f}")
        print(f"Total score (TM + λLM, λ={args.lambda_lm}): {total:.3f}")

    print("\n=== Translation Demo (Noisy Channel Decoding) ===")
    if args.test_src:
        with open(args.test_src, "r", encoding="utf-8") as f:
            test_lines = [ln.strip() for ln in f if ln.strip()]
        for s in test_lines[:5]:
            translate_sentence(s)
    elif args.interactive:
        print("Interactive mode. Type an English sentence and press Enter. Ctrl+C to exit.")
        try:
            while True:
                s = input("\nEN> ").strip()
                if not s:
                    continue
                translate_sentence(s)
        except KeyboardInterrupt:
            print("\nBye.")
    else:
        # Default: translate a few simple fixed samples
        for s in ["i love water", "where is the school", "this is a book", "he is going home"]:
            translate_sentence(s)

    # 8) Show a couple word alignments for sample training pairs (optional but useful)
    print("\n=== Sample Word Alignments (IBM1 argmax) ===")
    for (en_toks, hi_toks) in pairs_tok[:3]:
        links = align_sentence_ibm1(en_toks, hi_toks, t_ibm1)
        # Pretty print as word pairs
        pairs_str = []
        for i, j in links[:20]:
            pairs_str.append(f"{en_toks[i]}↔{hi_toks[j]}")
        print(f"EN: {' '.join(en_toks)}")
        print(f"HI: {' '.join(hi_toks)}")
        print(f"ALN: {', '.join(pairs_str)}\n")


if __name__ == "__main__":
    main()
