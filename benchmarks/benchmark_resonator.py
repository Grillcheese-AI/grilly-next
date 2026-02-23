"""
Resonator Network Generation Benchmark
=======================================

End-to-end test of VSA sentence generation via bitpacked Hamming resonance.

Pipeline:
  1. Build a word codebook (BLAKE3 bipolar vectors for each word)
  2. Encode a sentence using TextEncoder (bind + bundle + bitpack)
  3. Decompose the bundle back into words via the ResonatorNetwork
  4. Measure: recovery accuracy, per-token latency, explaining-away effect

This proves that CubeMind can both ENCODE and DECODE sentences through
the VSA geometry — the complete round-trip for generation.

Tests:
  A. Simple decode (no explaining away)
  B. Decode with explaining away (noise cancellation)
  C. Noisy decode (corrupted bundle -> robustness)
  D. Latency benchmark (tokens/sec for generation)
"""

import time

import numpy as np


DEP_ROLES_POOL = [
    "nsubj", "ROOT", "dobj", "prep", "pobj",
    "det", "amod", "compound", "advmod", "aux",
    "cc", "conj",
]


def run_benchmark(vocab_size=1000, dim=10240, sentence_lengths=None):
    """Run the full resonator benchmark."""

    import grilly_core

    if sentence_lengths is None:
        sentence_lengths = [3, 5, 8, 12]

    print("=" * 64)
    print("  Resonator Network Generation Benchmark")
    print("=" * 64)
    print()

    # -- Initialize Vulkan -------------------------------------------------
    dev = grilly_core.Device()
    dev.load_shaders("shaders/spv")
    print(f"  GPU: {dev.device_name}")
    print(f"  VSA dim: {dim}")
    print(f"  Vocab size: {vocab_size}")
    print()

    # -- Build codebook: BLAKE3 bipolar vectors for each word ---------------
    print(f"Building word codebook ({vocab_size} words)...")
    words = [f"word_{i:04d}" for i in range(vocab_size)]

    # Generate bipolar vectors via BLAKE3 (deterministic per-word)
    codebook_bipolar = np.zeros((vocab_size, dim), dtype=np.int8)
    for i, word in enumerate(words):
        vec = grilly_core.blake3_role(f"filler_{word}", dim)
        codebook_bipolar[i] = vec

    # -- Initialize ResonatorNetwork and load codebook ----------------------
    resonator = grilly_core.ResonatorNetwork(dev, dim=dim)
    resonator.load_codebook_bipolar(words, codebook_bipolar.ravel())
    print(f"  Codebook loaded: {resonator.codebook_size} entries")

    # -- Initialize TextEncoder with same fillers ---------------------------
    encoder = grilly_core.TextEncoder(dim=dim)
    for i, word in enumerate(words):
        encoder.add_filler(word, codebook_bipolar[i])
    print(f"  TextEncoder vocab: {encoder.vocab_size()}")
    print()

    # ======================================================================
    # TEST A: Round-trip encode -> decode (no explaining away)
    # ======================================================================
    print("--- Test A: Round-Trip Accuracy (no explaining away) ---")

    for slen in sentence_lengths:
        rng = np.random.RandomState(42 + slen)
        sentence_indices = rng.choice(vocab_size, size=slen, replace=False)
        sentence_words = [words[i] for i in sentence_indices]
        dep_roles = DEP_ROLES_POOL[:slen]
        positions = list(range(slen))

        # Encode (three-way: word x role x position)
        encoded = encoder.encode_sentence(sentence_words, dep_roles, positions)
        bundle_packed = np.array(encoded["data"], dtype=np.uint32)

        # Decode — pass SAME dep_roles and positions used during encoding
        result = resonator.generate_sentence(
            bundle_packed, dep_roles, positions, explain_away=False)

        recovered = [r["word"] for r in result]
        correct = sum(1 for r, s in zip(recovered, sentence_words) if r == s)
        accuracy = correct / slen * 100

        sims = [r["similarity"] for r in result]
        print(f"  len={slen:>2d}:  accuracy={accuracy:5.1f}%  "
              f"avg_sim={np.mean(sims):.4f}  "
              f"min_sim={min(sims):.4f}")

    print()

    # ======================================================================
    # TEST B: Round-trip with explaining away
    # ======================================================================
    print("--- Test B: Round-Trip Accuracy (with explaining away) ---")

    for slen in sentence_lengths:
        rng = np.random.RandomState(42 + slen)
        sentence_indices = rng.choice(vocab_size, size=slen, replace=False)
        sentence_words = [words[i] for i in sentence_indices]
        dep_roles = DEP_ROLES_POOL[:slen]
        positions = list(range(slen))

        encoded = encoder.encode_sentence(sentence_words, dep_roles, positions)
        bundle_packed = np.array(encoded["data"], dtype=np.uint32)

        result = resonator.generate_sentence(
            bundle_packed, dep_roles, positions, explain_away=True)

        recovered = [r["word"] for r in result]
        correct = sum(1 for r, s in zip(recovered, sentence_words) if r == s)
        accuracy = correct / slen * 100

        sims = [r["similarity"] for r in result]
        print(f"  len={slen:>2d}:  accuracy={accuracy:5.1f}%  "
              f"avg_sim={np.mean(sims):.4f}  "
              f"min_sim={min(sims):.4f}")

    print()

    # ======================================================================
    # TEST C: Noisy decode (bit flips in bundle)
    # ======================================================================
    print("--- Test C: Robustness to Noise (5% bit corruption) ---")

    for slen in [5, 8, 12]:
        rng = np.random.RandomState(42 + slen)
        sentence_indices = rng.choice(vocab_size, size=slen, replace=False)
        sentence_words = [words[i] for i in sentence_indices]
        dep_roles = DEP_ROLES_POOL[:slen]
        positions = list(range(slen))

        encoded = encoder.encode_sentence(sentence_words, dep_roles, positions)
        bundle_packed = np.array(encoded["data"], dtype=np.uint32)

        # Corrupt 5% of bits
        words_per_vec = len(bundle_packed)
        n_flip = max(1, int(dim * 0.05) // 32)
        flip_indices = rng.choice(words_per_vec, size=n_flip, replace=False)
        noisy_packed = bundle_packed.copy()
        for fi in flip_indices:
            noisy_packed[fi] ^= rng.randint(0, 0xFFFFFFFF, dtype=np.uint32)

        result_clean = resonator.generate_sentence(
            bundle_packed, dep_roles, positions, explain_away=True)
        result_noisy = resonator.generate_sentence(
            noisy_packed, dep_roles, positions, explain_away=True)

        clean_acc = sum(1 for r, s in zip(
            [r["word"] for r in result_clean], sentence_words) if r == s)
        noisy_acc = sum(1 for r, s in zip(
            [r["word"] for r in result_noisy], sentence_words) if r == s)

        print(f"  len={slen:>2d}:  clean={clean_acc}/{slen} "
              f"({clean_acc / slen * 100:.0f}%)  "
              f"noisy={noisy_acc}/{slen} "
              f"({noisy_acc / slen * 100:.0f}%)")

    print()

    # ======================================================================
    # TEST D: Generation latency
    # ======================================================================
    print("--- Test D: Generation Latency ---")

    for slen in [5, 10, 20]:
        rng = np.random.RandomState(42 + slen)
        sentence_indices = rng.choice(vocab_size, size=slen, replace=False)
        sentence_words = [words[i] for i in sentence_indices]
        dep_roles = (DEP_ROLES_POOL * 3)[:slen]
        positions = list(range(slen))

        encoded = encoder.encode_sentence(sentence_words, dep_roles, positions)
        bundle_packed = np.array(encoded["data"], dtype=np.uint32)

        # Warmup
        _ = resonator.generate_sentence(
            bundle_packed, dep_roles, positions, explain_away=True)

        # Timed run (10 iterations for averaging)
        n_iter = 10
        t0 = time.perf_counter()
        for _ in range(n_iter):
            result = resonator.generate_sentence(
                bundle_packed, dep_roles, positions, explain_away=True)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) / n_iter
        per_token = elapsed / slen * 1000

        print(f"  len={slen:>2d}:  total={elapsed * 1000:.3f}ms  "
              f"per_token={per_token:.3f}ms  "
              f"tokens/sec={slen / elapsed:.0f}")

    print()

    # ======================================================================
    # TEST E: Single resonation timing
    # ======================================================================
    print("--- Test E: Single Resonation Timing ---")

    # Build a query
    query_bipolar = codebook_bipolar[0]
    query_packed = np.zeros(dim // 32, dtype=np.uint32)
    for d in range(dim):
        if query_bipolar[d] > 0:
            query_packed[d // 32] |= (1 << (d % 32))

    # Warmup
    _ = resonator.resonate(query_packed)

    # Timed run
    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        res = resonator.resonate(query_packed)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / n_iter * 1000

    print(f"  Codebook: {resonator.codebook_size} entries")
    print(f"  Avg resonation: {avg_ms:.3f} ms")
    print(f"  Best match: {res['best_word']} (sim={res['best_similarity']:.4f})")
    print(f"  Resonations/sec: {n_iter / (t1 - t0):.0f}")
    print()

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("=" * 64)
    print("  Resonator Benchmark Summary")
    print("=" * 64)
    print(f"  GPU:               {dev.device_name}")
    print(f"  VSA dimension:     {dim}")
    print(f"  Codebook size:     {vocab_size}")
    print(f"  Resonation time:   {avg_ms:.3f} ms")
    print(f"  Total resonations: {resonator.total_resonations}")
    print("=" * 64)

    return {
        "resonation_ms": avg_ms,
        "codebook_size": vocab_size,
    }


if __name__ == "__main__":
    run_benchmark(vocab_size=1000, dim=10240,
                  sentence_lengths=[3, 5, 8, 12])
