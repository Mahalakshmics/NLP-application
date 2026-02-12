#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from datasets import load_dataset
from tqdm import tqdm
import re

devnagari = re.compile(r"[\u0900-\u097F]")
latin = re.compile(r"[A-Za-z]")

def is_reasonable_pair(en, hi):
    # must contain at least one Hindi (Devanagari) char
    if not devnagari.search(hi):
        return False
    # English should contain at least one letter
    if not latin.search(en):
        return False
    # very short lines often are UI fragments
    if len(en.split()) < 2 or len(hi.split()) < 2:
        return False
    return True

SEED = 42
OUT_DIR = "data"

# Adjust these for speed
MAX_TRAIN = 500000     # start with 20k–100k for IBM1
MAX_DEV   = 2000
MAX_TEST  = 2000

def safe_get_translation(example):
    """
    Handles common HF dataset formats:
      example["translation"] = {"en": "...", "hi": "..."}
    """
    if "translation" in example and isinstance(example["translation"], dict):
        en = (example["translation"].get("en") or "").strip()
        hi = (example["translation"].get("hi") or "").strip()
        return en, hi
    # Fallback: sometimes datasets store directly
    en = (example.get("en") or "").strip()
    hi = (example.get("hi") or "").strip()
    return en, hi

def write_bitext(pairs, src_path, tgt_path):
    with open(src_path, "w", encoding="utf-8") as fsrc, open(tgt_path, "w", encoding="utf-8") as ftgt:
        for en, hi in pairs:
            fsrc.write(en.replace("\n", " ") + "\n")
            ftgt.write(hi.replace("\n", " ") + "\n")

def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[INFO] Loading dataset: cfilt/iitb-english-hindi")
    ds = load_dataset("cfilt/iitb-english-hindi")

    print("[INFO] Available splits:", list(ds.keys()))
    # Try to use official splits if present
    if "train" in ds and ("validation" in ds or "test" in ds):
        train_data = ds["train"]
        val_data   = ds["validation"] if "validation" in ds else None
        test_data  = ds["test"] if "test" in ds else None

        # Show one example structure
        ex0 = train_data[0]
        print("[DEBUG] Sample keys:", list(ex0.keys()))
        if "translation" in ex0:
            print("[DEBUG] Sample translation keys:", list(ex0["translation"].keys()))

        def collect(split, max_n):
            out = []
            for ex in tqdm(split, desc=f"Collecting {max_n} pairs"):
                en, hi = safe_get_translation(ex)
                if en and hi and is_reasonable_pair(en, hi):
                    out.append((en, hi))
                if len(out) >= max_n:
                    break
            return out

        train_pairs = collect(train_data, MAX_TRAIN)

        dev_pairs = collect(val_data, MAX_DEV) if val_data is not None else []
        test_pairs = collect(test_data, MAX_TEST) if test_data is not None else []

        # If validation/test missing, sample from train tail
        if not dev_pairs or not test_pairs:
            print("[WARN] validation/test split missing or empty; sampling from train instead.")
            pool = train_pairs.copy()
            random.shuffle(pool)
            dev_pairs = pool[:MAX_DEV]
            test_pairs = pool[MAX_DEV:MAX_DEV+MAX_TEST]

    else:
        # Single split case: make our own train/dev/test split
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        data = ds[split_name]
        ex0 = data[0]
        print("[DEBUG] Sample keys:", list(ex0.keys()))
        if "translation" in ex0:
            print("[DEBUG] Sample translation keys:", list(ex0["translation"].keys()))

        pairs = []
        for ex in tqdm(data, desc="Collecting pairs"):
            en, hi = safe_get_translation(ex)
            if en and hi:
                pairs.append((en, hi))

        if len(pairs) == 0:
            raise RuntimeError("Collected 0 (en,hi) pairs. Check dataset fields printed above.")

        random.shuffle(pairs)
        train_pairs = pairs[:MAX_TRAIN]
        dev_pairs   = pairs[MAX_TRAIN:MAX_TRAIN+MAX_DEV]
        test_pairs  = pairs[MAX_TRAIN+MAX_DEV:MAX_TRAIN+MAX_DEV+MAX_TEST]

    # Write out
    train_en = os.path.join(OUT_DIR, "train.en")
    train_hi = os.path.join(OUT_DIR, "train.hi")
    dev_en   = os.path.join(OUT_DIR, "dev.en")
    dev_hi   = os.path.join(OUT_DIR, "dev.hi")
    test_en  = os.path.join(OUT_DIR, "test.en")
    test_hi  = os.path.join(OUT_DIR, "test.hi")

    write_bitext(train_pairs, train_en, train_hi)
    write_bitext(dev_pairs,   dev_en,   dev_hi)
    write_bitext(test_pairs,  test_en,  test_hi)

    print("\n[SUCCESS] Wrote files:")
    print(" -", train_en, "lines:", len(train_pairs))
    print(" -", train_hi, "lines:", len(train_pairs))
    print(" -", dev_en,   "lines:", len(dev_pairs))
    print(" -", dev_hi,   "lines:", len(dev_pairs))
    print(" -", test_en,  "lines:", len(test_pairs))
    print(" -", test_hi,  "lines:", len(test_pairs))

    # Quick sanity check: file existence
    for p in [train_en, train_hi]:
        if not os.path.exists(p):
            raise RuntimeError(f"Expected output missing: {p}")

if __name__ == "__main__":
    main()
