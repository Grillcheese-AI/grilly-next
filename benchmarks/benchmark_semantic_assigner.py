"""
SemanticAssigner Memoization Benchmark
======================================

Measures the speedup from bitpacked memoization caching under Zipfian
token distributions — the key optimization for LLM pretraining throughput.

Problem: Projecting a 300D FastText vector through a 300x10240 Gaussian
matrix costs ~3M FLOPs per word. At 10K tokens/sec, this chokes the CPU.

Solution: Cache the BitpackedVec result per unique word. Because natural
language follows Zipf's law (top 1000 words = ~80% of tokens), the cache
hit rate converges quickly.

Benchmarks:
  1. Python uncached  — numpy matmul + sign every token (baseline)
  2. C++ uncached     — SemanticAssigner.project_to_bipolar() every token
  3. C++ cached       — SemanticAssigner.get_semantic_filler() with memoization
  4. C++ prewarmed    — Cache fully pre-populated, pure lookup speed

Reports: tokens/sec, cache hit rate, speedup vs baseline.
"""

import time

import numpy as np


# == Zipfian Token Stream Generator =========================================


def generate_zipfian_stream(vocab_size, num_tokens, alpha=1.07, seed=123):
    """Generate a token stream following Zipf's law.

    In natural language, word frequency follows:
        f(rank) ~ rank^(-alpha)

    With alpha~1.07 (empirical for English), the top 100 words account
    for ~50% of all tokens, and the top 1000 cover ~80%.

    Returns:
        tokens: list of token strings
        vocab: list of unique token strings
    """
    rng = np.random.RandomState(seed)

    # Build Zipf probability distribution
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = ranks ** (-alpha)
    probs /= probs.sum()

    # Generate token indices
    indices = rng.choice(vocab_size, size=num_tokens, p=probs)

    # Create vocabulary (simulates word tokens)
    vocab = [f"word_{i:05d}" for i in range(vocab_size)]
    tokens = [vocab[idx] for idx in indices]

    return tokens, vocab


def generate_random_embeddings(vocab, embed_dim=300, seed=42):
    """Generate random float32 embeddings for each vocabulary word.

    In production these would be FastText vectors. For benchmarking,
    random vectors preserve the projection cost characteristics.
    """
    rng = np.random.RandomState(seed)
    embeddings = {}
    for word in vocab:
        vec = rng.randn(embed_dim).astype(np.float32)
        # L2 normalize (like real FastText embeddings)
        vec /= max(np.linalg.norm(vec), 1e-9)
        embeddings[word] = vec
    return embeddings


# == Python Baseline (numpy) ================================================


class PythonProjector:
    """Python reference: numpy matmul + sign for each token. No caching."""

    def __init__(self, embed_dim=300, vsa_dim=10240):
        self.embed_dim = embed_dim
        self.vsa_dim = vsa_dim
        rng = np.random.RandomState(42)
        self.projection = rng.randn(embed_dim, vsa_dim).astype(np.float32)

    def project(self, vec):
        """Project one vector: 300 x 10240 matmul + sign. ~3M FLOPs."""
        projected = vec @ self.projection
        bipolar = np.sign(projected).astype(np.int8)
        bipolar[bipolar == 0] = 1
        return bipolar


# == Benchmark Functions ====================================================


def bench_python_uncached(tokens, embeddings, projector):
    """Baseline: Python numpy projection for every token (no caching)."""
    t0 = time.perf_counter()
    for token in tokens:
        _ = projector.project(embeddings[token])
    elapsed = time.perf_counter() - t0
    return elapsed


def bench_cpp_uncached(tokens, embeddings, sa):
    """C++ projection for every token, bypassing cache."""
    t0 = time.perf_counter()
    for token in tokens:
        _ = sa.project_to_bipolar(embeddings[token])
    elapsed = time.perf_counter() - t0
    return elapsed


def bench_cpp_cached(tokens, sa):
    """C++ SemanticAssigner with memoization cache (the production path)."""
    sa.reset_stats()
    t0 = time.perf_counter()
    for token in tokens:
        _ = sa.get_semantic_filler(token)
    elapsed = time.perf_counter() - t0
    return elapsed


def bench_cpp_prewarmed(tokens, sa):
    """C++ SemanticAssigner with fully pre-warmed cache (all hits)."""
    sa.reset_stats()
    t0 = time.perf_counter()
    for token in tokens:
        _ = sa.get_semantic_filler(token)
    elapsed = time.perf_counter() - t0
    return elapsed


# == Cache Hit Rate Analysis ================================================


def analyze_cache_convergence(tokens, sa, checkpoints=None):
    """Track cache hit rate as tokens stream in.

    Shows how Zipfian distribution causes rapid convergence:
    after seeing ~2000 tokens, hit rate is typically >95%.
    """
    if checkpoints is None:
        checkpoints = [100, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]

    sa.reset_stats()
    results = []

    for i, token in enumerate(tokens):
        _ = sa.get_semantic_filler(token)
        count = i + 1
        if count in checkpoints:
            results.append({
                "tokens_seen": count,
                "cache_size": sa.cache_size,
                "hits": sa.cache_hits,
                "misses": sa.cache_misses,
                "hit_rate": sa.hit_rate,
            })

    # Final checkpoint
    results.append({
        "tokens_seen": len(tokens),
        "cache_size": sa.cache_size,
        "hits": sa.cache_hits,
        "misses": sa.cache_misses,
        "hit_rate": sa.hit_rate,
    })

    return results


# == Main Benchmark =========================================================


def run_benchmark(vocab_size=10000, num_tokens=100000, embed_dim=300,
                  vsa_dim=10240, alpha=1.07):
    """Run the full memoization benchmark suite."""

    import grilly_core

    print("=" * 64)
    print("  SemanticAssigner Memoization Benchmark")
    print("=" * 64)
    print()

    # -- Generate Zipfian token stream -------------------------------------
    print(f"Generating Zipfian token stream...")
    print(f"  Vocab size:    {vocab_size:,}")
    print(f"  Num tokens:    {num_tokens:,}")
    print(f"  Zipf alpha:    {alpha}")
    print(f"  Embed dim:     {embed_dim}")
    print(f"  VSA dim:       {vsa_dim}")

    tokens, vocab = generate_zipfian_stream(vocab_size, num_tokens, alpha=alpha)

    # Count unique tokens actually used
    unique_used = len(set(tokens))
    print(f"  Unique tokens: {unique_used:,} (of {vocab_size:,} vocab)")

    # Show top-10 token frequencies
    from collections import Counter
    freq = Counter(tokens)
    top10 = freq.most_common(10)
    total = len(tokens)
    cumulative = 0
    print(f"\n  Top 10 tokens (Zipf distribution):")
    for word, count in top10:
        pct = count / total * 100
        cumulative += pct
        print(f"    {word}: {count:>6,} ({pct:5.2f}%, cumul: {cumulative:5.1f}%)")

    # -- Generate embeddings -----------------------------------------------
    print(f"\nGenerating {vocab_size:,} random float32 embeddings ({embed_dim}D)...")
    embeddings = generate_random_embeddings(vocab, embed_dim=embed_dim)

    # -- Initialize engines ------------------------------------------------
    print(f"\nInitializing engines...")
    projector = PythonProjector(embed_dim=embed_dim, vsa_dim=vsa_dim)
    sa = grilly_core.SemanticAssigner(dim=vsa_dim, ft_dim=embed_dim)

    # Register all embeddings in the SemanticAssigner
    print(f"  Loading {len(embeddings):,} float vectors into SemanticAssigner...")
    t_load = time.perf_counter()
    for word, vec in embeddings.items():
        sa.add_float_vector(word, vec)
    t_load = time.perf_counter() - t_load
    print(f"  Loaded in {t_load:.3f}s ({len(embeddings) / t_load:.0f} vectors/sec)")
    print(f"  Float vocab: {sa.float_vocab_size:,}")
    print(f"  Projection matrix: {embed_dim} x {vsa_dim} = "
          f"{embed_dim * vsa_dim:,} floats "
          f"({embed_dim * vsa_dim * 4 / 1024 / 1024:.1f} MB)")

    # ======================================================================
    # BENCHMARK 1: Python uncached (numpy matmul + sign)
    # ======================================================================
    print(f"\n--- Benchmark 1: Python Uncached (numpy) ---")
    # Use fewer tokens for Python (it's slow)
    py_tokens = tokens[:10000]
    t_py = bench_python_uncached(py_tokens, embeddings, projector)
    py_tps = len(py_tokens) / t_py
    print(f"  {len(py_tokens):,} tokens in {t_py:.3f}s")
    print(f"  Throughput: {py_tps:,.0f} tokens/sec")

    # ======================================================================
    # BENCHMARK 2: C++ uncached (project_to_bipolar every time)
    # ======================================================================
    print(f"\n--- Benchmark 2: C++ Uncached (project_to_bipolar) ---")
    cpp_tokens = tokens[:10000]
    t_cpp = bench_cpp_uncached(cpp_tokens, embeddings, sa)
    cpp_tps = len(cpp_tokens) / t_cpp
    print(f"  {len(cpp_tokens):,} tokens in {t_cpp:.3f}s")
    print(f"  Throughput: {cpp_tps:,.0f} tokens/sec")
    speedup_cpp_vs_py = cpp_tps / py_tps if py_tps > 0 else 0
    print(f"  Speedup vs Python: {speedup_cpp_vs_py:.1f}x")

    # ======================================================================
    # BENCHMARK 3: C++ cached with cold start (Zipfian convergence)
    # ======================================================================
    print(f"\n--- Benchmark 3: C++ Cached, Cold Start (Zipfian) ---")
    # Fresh assigner for clean cache stats
    sa_cold = grilly_core.SemanticAssigner(dim=vsa_dim, ft_dim=embed_dim)
    for word, vec in embeddings.items():
        sa_cold.add_float_vector(word, vec)

    t_cold = bench_cpp_cached(tokens, sa_cold)
    cold_tps = len(tokens) / t_cold
    print(f"  {len(tokens):,} tokens in {t_cold:.3f}s")
    print(f"  Throughput: {cold_tps:,.0f} tokens/sec")
    print(f"  Cache hits:   {sa_cold.cache_hits:,}")
    print(f"  Cache misses: {sa_cold.cache_misses:,}")
    print(f"  Hit rate:     {sa_cold.hit_rate * 100:.2f}%")
    speedup_cold_vs_py = cold_tps / py_tps if py_tps > 0 else 0
    print(f"  Speedup vs Python uncached: {speedup_cold_vs_py:.1f}x")

    # ======================================================================
    # BENCHMARK 4: C++ prewarmed (100% cache hits)
    # ======================================================================
    print(f"\n--- Benchmark 4: C++ Prewarmed Cache (100% hits) ---")
    print(f"  Pre-warming cache...")
    t_prewarm = time.perf_counter()
    prewarmed = sa_cold.prewarm()
    t_prewarm = time.perf_counter() - t_prewarm
    print(f"  Pre-warmed {prewarmed:,} entries in {t_prewarm:.3f}s "
          f"({prewarmed / t_prewarm:.0f} entries/sec)")

    t_warm = bench_cpp_prewarmed(tokens, sa_cold)
    warm_tps = len(tokens) / t_warm
    print(f"  {len(tokens):,} tokens in {t_warm:.3f}s")
    print(f"  Throughput: {warm_tps:,.0f} tokens/sec")
    print(f"  Cache hits:   {sa_cold.cache_hits:,}")
    print(f"  Cache misses: {sa_cold.cache_misses:,}")
    print(f"  Hit rate:     {sa_cold.hit_rate * 100:.2f}%")
    speedup_warm_vs_py = warm_tps / py_tps if py_tps > 0 else 0
    speedup_warm_vs_cpp = warm_tps / cpp_tps if cpp_tps > 0 else 0
    print(f"  Speedup vs Python uncached:  {speedup_warm_vs_py:.0f}x")
    print(f"  Speedup vs C++ uncached:     {speedup_warm_vs_cpp:.0f}x")

    # ======================================================================
    # CACHE CONVERGENCE ANALYSIS
    # ======================================================================
    print(f"\n--- Cache Hit Rate Convergence (Zipf alpha={alpha}) ---")
    sa_conv = grilly_core.SemanticAssigner(dim=vsa_dim, ft_dim=embed_dim)
    for word, vec in embeddings.items():
        sa_conv.add_float_vector(word, vec)

    checkpoints = [100, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000]
    checkpoints = [c for c in checkpoints if c <= num_tokens]
    convergence = analyze_cache_convergence(tokens, sa_conv, checkpoints)

    print(f"  {'Tokens':>10s}  {'Cache Size':>10s}  {'Hit Rate':>10s}  "
          f"{'Hits':>10s}  {'Misses':>10s}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for r in convergence:
        print(f"  {r['tokens_seen']:>10,}  {r['cache_size']:>10,}  "
              f"{r['hit_rate'] * 100:>9.2f}%  "
              f"{r['hits']:>10,}  {r['misses']:>10,}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print(f"\n{'=' * 64}")
    print(f"  SUMMARY: SemanticAssigner Memoization Benchmark")
    print(f"{'=' * 64}")
    print(f"  Vocab: {vocab_size:,} words, Stream: {num_tokens:,} tokens")
    print(f"  Projection: {embed_dim} -> {vsa_dim} ({embed_dim * vsa_dim:,} FLOPs/word)")
    print()
    print(f"  {'Method':<30s}  {'tokens/sec':>12s}  {'Speedup':>8s}")
    print(f"  {'-' * 30}  {'-' * 12}  {'-' * 8}")
    print(f"  {'Python uncached (numpy)':<30s}  {py_tps:>12,.0f}  {'1.0x':>8s}")
    print(f"  {'C++ uncached (project)':<30s}  {cpp_tps:>12,.0f}  "
          f"{speedup_cpp_vs_py:>7.1f}x")
    print(f"  {'C++ cached (cold start)':<30s}  {cold_tps:>12,.0f}  "
          f"{speedup_cold_vs_py:>7.1f}x")
    print(f"  {'C++ cached (prewarmed)':<30s}  {warm_tps:>12,.0f}  "
          f"{speedup_warm_vs_py:>7.0f}x")
    print()
    print(f"  Final cache hit rate: {sa_cold.hit_rate * 100:.2f}%")
    print(f"  Cache entries: {sa_cold.cache_size:,}")
    print(f"  Memory per entry: {(vsa_dim + 31) // 32 * 4} bytes (bitpacked)")
    cache_mb = sa_cold.cache_size * ((vsa_dim + 31) // 32 * 4) / 1024 / 1024
    print(f"  Total cache memory: {cache_mb:.2f} MB")
    print(f"{'=' * 64}")

    return {
        "py_tps": py_tps,
        "cpp_tps": cpp_tps,
        "cold_tps": cold_tps,
        "warm_tps": warm_tps,
        "speedup_cached_vs_python": speedup_warm_vs_py,
        "speedup_cached_vs_cpp": speedup_warm_vs_cpp,
        "final_hit_rate": sa_cold.hit_rate,
        "cache_entries": sa_cold.cache_size,
    }


if __name__ == "__main__":
    run_benchmark(
        vocab_size=10000,
        num_tokens=100000,
        embed_dim=300,
        vsa_dim=10240,
        alpha=1.07,
    )
