"""
CubeMind Generation Quality Test
=================================

Round-trip encode → decode on REAL sentences from Phase 1 data.
Tests whether the resonator can recover actual words from natural
language bundles against a 128K-word codebook.

This is the critical quality gate: synthetic 1000-word codebooks
are easy. Real language with 128K tokens, Zipfian frequency
distribution, and variable sentence lengths is the real test.

Metrics:
  - Per-token accuracy (exact match)
  - Per-sentence perfect recovery rate
  - Accuracy vs sentence length
  - Accuracy vs complexity
  - Explaining-away lift (with vs without)
  - Similarity score distributions
  - Per-realm accuracy breakdown
"""

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_sample(data_dir, n_per_file=5000, seed=42):
    """Load a stratified sample from the JSONL files."""
    rng = np.random.RandomState(seed)
    rows = []
    for fpath in sorted(Path(data_dir).glob("*.jsonl")):
        file_rows = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                file_rows.append(json.loads(line))
        # Shuffle and take n_per_file
        indices = rng.permutation(len(file_rows))[:n_per_file]
        for i in indices:
            rows.append(file_rows[i])
    return rows


def run_quality_test(data_dir="pretraining/phase1", n_samples=5000,
                     dim=10240, seed=42):
    """Run comprehensive generation quality analysis."""

    import grilly_core

    print("=" * 70)
    print("  CubeMind Generation Quality Test")
    print("=" * 70)
    print()

    # ── 1. Load sample data ───────────────────────────────────────────
    print("Loading sample data...")
    rows = load_sample(data_dir, n_per_file=n_samples, seed=seed)
    print(f"  Loaded {len(rows):,d} sentences")

    # Filter out very short sentences (< 3 tokens) — can't test generation
    rows = [r for r in rows if len(r.get("lemmas", [])) >= 3]
    print(f"  After filtering (>=3 tokens): {len(rows):,d}")
    print()

    # ── 2. Build vocabulary + codebook ────────────────────────────────
    print("Building vocabulary codebook...")
    vocab = set()
    for row in rows:
        for lemma in row.get("lemmas", []):
            vocab.add(lemma)
    vocab = sorted(vocab)
    print(f"  Unique tokens: {len(vocab):,d}")

    # Initialize Vulkan
    dev = grilly_core.Device()
    dev.load_shaders("shaders/spv")
    print(f"  GPU: {dev.device_name}")

    # Generate BLAKE3 bipolar vectors for entire vocabulary
    t0 = time.perf_counter()
    codebook_bipolar = np.zeros((len(vocab), dim), dtype=np.int8)
    for i, word in enumerate(vocab):
        codebook_bipolar[i] = grilly_core.blake3_role(f"filler_{word}", dim)
    elapsed = time.perf_counter() - t0
    print(f"  Codebook generated: {len(vocab):,d} entries ({elapsed:.2f}s)")

    # Load into resonator
    resonator = grilly_core.ResonatorNetwork(dev, dim=dim)
    resonator.load_codebook_bipolar(vocab, codebook_bipolar.ravel())

    # Load into text encoder
    encoder = grilly_core.TextEncoder(dim=dim)
    for i, word in enumerate(vocab):
        encoder.add_filler(word, codebook_bipolar[i])

    print(f"  Resonator codebook: {resonator.codebook_size:,d}")
    print(f"  Encoder vocab: {encoder.vocab_size():,d}")
    print()

    # ── 3. Round-trip test ────────────────────────────────────────────
    print("=" * 70)
    print("  Round-Trip Encode -> Decode")
    print("=" * 70)
    print()

    # Buckets for analysis
    results_by_length = defaultdict(lambda: {"correct": 0, "total": 0, "sims": []})
    results_by_realm = defaultdict(lambda: {"correct": 0, "total": 0, "perfect": 0, "count": 0})
    results_by_complexity = defaultdict(lambda: {"correct": 0, "total": 0})
    results_no_ea = {"correct": 0, "total": 0}  # without explaining away
    results_ea = {"correct": 0, "total": 0}     # with explaining away

    all_sims_correct = []
    all_sims_wrong = []
    perfect_sentences = 0
    total_sentences = 0
    total_tokens = 0

    t_start = time.perf_counter()

    for idx, row in enumerate(rows):
        lemmas = row.get("lemmas", [])
        deps = row.get("deps", [])
        slen = len(lemmas)

        if slen < 3 or slen > 50:
            continue

        positions = list(range(slen))
        realm = row.get("realm", "unknown")
        complexity = row.get("complexity", 0.5)

        # Encode
        encoded = encoder.encode_sentence(lemmas, deps, positions)
        bundle_packed = np.array(encoded["data"], dtype=np.uint32)

        # Decode WITH explaining away
        result_ea_list = resonator.generate_sentence(
            bundle_packed, deps, positions, explain_away=True)

        recovered_ea = [r["word"] for r in result_ea_list]
        sims_ea = [r["similarity"] for r in result_ea_list]

        # Count per-token accuracy
        correct_ea = sum(1 for r, s in zip(recovered_ea, lemmas) if r == s)
        results_ea["correct"] += correct_ea
        results_ea["total"] += slen
        total_tokens += slen

        # Perfect sentence recovery
        is_perfect = (correct_ea == slen)
        if is_perfect:
            perfect_sentences += 1
        total_sentences += 1

        # Similarity distributions
        for r, s, sim in zip(recovered_ea, lemmas, sims_ea):
            if r == s:
                all_sims_correct.append(sim)
            else:
                all_sims_wrong.append(sim)

        # By length bucket
        if slen <= 5:
            bucket = "3-5"
        elif slen <= 10:
            bucket = "6-10"
        elif slen <= 15:
            bucket = "11-15"
        elif slen <= 20:
            bucket = "16-20"
        elif slen <= 30:
            bucket = "21-30"
        else:
            bucket = "31-50"
        results_by_length[bucket]["correct"] += correct_ea
        results_by_length[bucket]["total"] += slen
        results_by_length[bucket]["sims"].extend(sims_ea)

        # By realm
        results_by_realm[realm]["correct"] += correct_ea
        results_by_realm[realm]["total"] += slen
        if is_perfect:
            results_by_realm[realm]["perfect"] += 1
        results_by_realm[realm]["count"] += 1

        # By complexity (buckets of 0.2)
        c_bucket = f"{int(complexity * 5) / 5:.1f}-{int(complexity * 5) / 5 + 0.2:.1f}"
        results_by_complexity[c_bucket]["correct"] += correct_ea
        results_by_complexity[c_bucket]["total"] += slen

        # Also test WITHOUT explaining away (every 10th sentence)
        if idx % 10 == 0:
            result_no_ea = resonator.generate_sentence(
                bundle_packed, deps, positions, explain_away=False)
            recovered_no = [r["word"] for r in result_no_ea]
            correct_no = sum(1 for r, s in zip(recovered_no, lemmas) if r == s)
            results_no_ea["correct"] += correct_no
            results_no_ea["total"] += slen

        # Progress
        if (idx + 1) % 1000 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / elapsed
            acc = results_ea["correct"] / results_ea["total"] * 100
            print(f"  [{idx + 1:>5,d}/{len(rows):,d}]  "
                  f"accuracy={acc:.1f}%  "
                  f"perfect={perfect_sentences}/{total_sentences} "
                  f"({perfect_sentences / total_sentences * 100:.1f}%)  "
                  f"{rate:.0f} sent/s")

    t_total = time.perf_counter() - t_start

    # ── 4. Results ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()

    # Overall accuracy
    acc_ea = results_ea["correct"] / results_ea["total"] * 100
    acc_no_ea = results_no_ea["correct"] / results_no_ea["total"] * 100 if results_no_ea["total"] else 0
    print(f"  Overall token accuracy (with EA):    {acc_ea:.1f}%  "
          f"({results_ea['correct']:,d}/{results_ea['total']:,d})")
    print(f"  Overall token accuracy (without EA): {acc_no_ea:.1f}%  "
          f"({results_no_ea['correct']:,d}/{results_no_ea['total']:,d})")
    print(f"  Explaining-away lift:                +{acc_ea - acc_no_ea:.1f}pp")
    print()
    print(f"  Perfect sentence recovery: {perfect_sentences:,d}/{total_sentences:,d} "
          f"({perfect_sentences / total_sentences * 100:.1f}%)")
    print(f"  Total tokens tested:       {total_tokens:,d}")
    print(f"  Codebook size:             {len(vocab):,d}")
    print(f"  Time:                      {t_total:.1f}s "
          f"({total_sentences / t_total:.0f} sent/s)")
    print()

    # Accuracy by sentence length
    print("  Accuracy by Sentence Length:")
    print("  " + "-" * 55)
    for bucket in ["3-5", "6-10", "11-15", "16-20", "21-30", "31-50"]:
        if bucket in results_by_length:
            r = results_by_length[bucket]
            acc = r["correct"] / r["total"] * 100 if r["total"] else 0
            avg_sim = np.mean(r["sims"]) if r["sims"] else 0
            print(f"    len {bucket:>5s}:  {acc:5.1f}%  "
                  f"({r['correct']:>6,d}/{r['total']:>6,d})  "
                  f"avg_sim={avg_sim:.4f}")
    print()

    # Accuracy by realm (top 8)
    print("  Accuracy by Realm:")
    print("  " + "-" * 55)
    realm_sorted = sorted(results_by_realm.items(),
                          key=lambda x: x[1]["total"], reverse=True)
    for realm, r in realm_sorted[:8]:
        acc = r["correct"] / r["total"] * 100 if r["total"] else 0
        perf = r["perfect"] / r["count"] * 100 if r["count"] else 0
        print(f"    {realm:>12s}:  {acc:5.1f}%  "
              f"perfect={perf:5.1f}%  "
              f"({r['total']:>6,d} tokens)")
    print()

    # Accuracy by complexity
    print("  Accuracy by Complexity:")
    print("  " + "-" * 55)
    for bucket in sorted(results_by_complexity.keys()):
        r = results_by_complexity[bucket]
        acc = r["correct"] / r["total"] * 100 if r["total"] else 0
        print(f"    complexity {bucket}:  {acc:5.1f}%  "
              f"({r['correct']:>6,d}/{r['total']:>6,d})")
    print()

    # Similarity score distributions
    if all_sims_correct:
        print("  Similarity Score Distribution:")
        print("  " + "-" * 55)
        print(f"    Correct tokens:  mean={np.mean(all_sims_correct):.4f}  "
              f"std={np.std(all_sims_correct):.4f}  "
              f"min={min(all_sims_correct):.4f}  "
              f"P5={np.percentile(all_sims_correct, 5):.4f}")
        if all_sims_wrong:
            print(f"    Wrong tokens:    mean={np.mean(all_sims_wrong):.4f}  "
                  f"std={np.std(all_sims_wrong):.4f}  "
                  f"max={max(all_sims_wrong):.4f}  "
                  f"P95={np.percentile(all_sims_wrong, 95):.4f}")
            # Separation gap
            gap = np.mean(all_sims_correct) - np.mean(all_sims_wrong)
            print(f"    Separation gap:  {gap:.4f} "
                  f"({'good' if gap > 0.05 else 'TIGHT'})")
        else:
            print("    Wrong tokens:    NONE (100% accuracy)")
    print()

    # Show some example recoveries
    print("  Example Sentences (first 10):")
    print("  " + "-" * 55)
    rng = np.random.RandomState(seed + 1)
    example_indices = rng.choice(len(rows), size=min(10, len(rows)), replace=False)
    for idx in example_indices:
        row = rows[idx]
        lemmas = row.get("lemmas", [])
        deps = row.get("deps", [])
        slen = len(lemmas)
        if slen < 3 or slen > 25:
            continue
        positions = list(range(slen))
        encoded = encoder.encode_sentence(lemmas, deps, positions)
        bundle_packed = np.array(encoded["data"], dtype=np.uint32)
        result = resonator.generate_sentence(
            bundle_packed, deps, positions, explain_away=True)
        recovered = [r["word"] for r in result]
        correct = sum(1 for r, s in zip(recovered, lemmas) if r == s)

        # Color-code: matching tokens normal, mismatches in brackets
        display = []
        for r, s in zip(recovered, lemmas):
            if r == s:
                display.append(s)
            else:
                display.append(f"[{r}!={s}]")

        status = "OK" if correct == slen else f"{correct}/{slen}"
        print(f"    ({status:>5s}) {' '.join(display[:20])}")

    print()
    print("=" * 70)

    return {
        "token_accuracy": acc_ea,
        "perfect_rate": perfect_sentences / total_sentences * 100,
        "codebook_size": len(vocab),
        "total_tokens": total_tokens,
    }


if __name__ == "__main__":
    run_quality_test(
        data_dir="pretraining/phase1",
        n_samples=5000,
        dim=10240,
    )
