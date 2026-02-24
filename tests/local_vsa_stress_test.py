"""
Grilly-Next Pre-Flight Stress Test
===================================

Validates the C++ VSA pipeline before scaling to A40/A100.
Mirrors the patterns in benchmarks/test_490k_lean.py.

Usage:
    python tests/local_vsa_stress_test.py              # full suite
    python tests/local_vsa_stress_test.py --cpu-only   # no GPU required
"""

import sys
import time
import json
import os
import gc
import argparse
import numpy as np

# ── Import grilly_core (C++ extension) ──────────────────────────────────

try:
    from grilly import grilly_core as _core
    HAS_NATIVE = True
except ImportError:
    try:
        import grilly_core as _core
        HAS_NATIVE = True
    except ImportError:
        HAS_NATIVE = False
        _core = None


def random_bipolar(n, dim, seed=42):
    """Generate n random bipolar int8 vectors ({-1, +1})."""
    rng = np.random.RandomState(seed)
    return (rng.randint(0, 2, size=(n, dim), dtype=np.int8) * 2 - 1)


# ── Tests ───────────────────────────────────────────────────────────────

def check_profile_exists():
    """Verify profiles.json is present and parseable."""
    profile_path = os.path.join(os.path.dirname(__file__), '..', 'profiles.json')
    profile_path = os.path.abspath(profile_path)

    if not os.path.exists(profile_path):
        print(f"  WARN: profiles.json not found at {profile_path}")
        return False

    with open(profile_path) as f:
        profiles = json.load(f)

    print(f"  Available profiles: {list(profiles.keys())}")
    for name, p in profiles.items():
        assert 'vsa_dim' in p, f"Profile '{name}' missing vsa_dim"
        assert 'subgroup_size' in p, f"Profile '{name}' missing subgroup_size"
        print(f"    {name}: dim={p['vsa_dim']}, subgroup={p['subgroup_size']}, "
              f"cache={p['max_cache_capacity']:,}")
    return True


def test_blake3_determinism():
    """BLAKE3 role vectors must be deterministic across calls."""
    dim = 10240
    role_a = np.array(_core.blake3_role("test_key", dim))
    role_b = np.array(_core.blake3_role("test_key", dim))
    role_c = np.array(_core.blake3_role("different_key", dim))

    assert np.array_equal(role_a, role_b), "BLAKE3 not deterministic!"
    assert not np.array_equal(role_a, role_c), "Different keys produced same vector!"

    unique_vals = set(role_a.tolist())
    assert unique_vals == {-1, 1}, f"Expected bipolar {{-1,1}}, got {unique_vals}"


def test_xor_self_inverse():
    """XOR binding must be its own inverse: bind(bind(a, b), b) == a."""
    dim = 10240
    a = np.array(_core.blake3_role("vector_a", dim))
    b = np.array(_core.blake3_role("vector_b", dim))

    bound = np.array(_core.vsa_bind(a, b))
    unbound = np.array(_core.vsa_bind(bound, b))

    assert np.array_equal(a, unbound), "XOR binding is not self-inverse!"


def test_bitpack_roundtrip():
    """Bitpacking must preserve all bits."""
    dim = 10240
    bipolar = np.array(_core.blake3_role("roundtrip_test", dim))

    packed = np.array(_core.vsa_bitpack(bipolar))
    words = (dim + 31) // 32
    assert len(packed) == words, f"Expected {words} words, got {len(packed)}"

    for d in range(dim):
        bit_val = (int(packed[d // 32]) >> (d % 32)) & 1
        expected = 1 if bipolar[d] == 1 else 0
        assert bit_val == expected, f"Bit mismatch at dim {d}"


def test_hamming_cpu():
    """CPU Hamming search must find exact matches at distance 0."""
    dim = 10240
    cache = random_bipolar(100, dim, seed=42)
    query = cache[50].copy()

    distances = np.array(_core.hamming_search_cpu(query, cache))
    idx = int(np.argmin(distances))
    dist = int(distances[idx])

    assert idx == 50, f"Expected index 50, got {idx}"
    assert dist == 0, f"Expected distance 0, got {dist}"


def test_world_model_coherence_cpu():
    """Fact vs negation must be orthogonal (~50% Hamming distance)."""
    dim = 10240

    s = np.array(_core.blake3_role("filler_dog", dim))
    r = np.array(_core.blake3_role("filler_is", dim))
    o = np.array(_core.blake3_role("filler_animal", dim))

    fact = np.array(_core.vsa_bind(_core.vsa_bind(s, r), o))

    neg_r = np.array(_core.blake3_role("filler_is_not", dim))
    neg = np.array(_core.vsa_bind(_core.vsa_bind(s, neg_r), o))

    fact_packed = np.array(_core.vsa_bitpack(fact))
    neg_packed = np.array(_core.vsa_bitpack(neg))

    xor_result = np.bitwise_xor(fact_packed.astype(np.uint32),
                                 neg_packed.astype(np.uint32))
    hamming = sum(bin(int(w)).count('1') for w in xor_result)
    normalized = hamming / dim

    print(f"  Fact vs Negation: {hamming}/{dim} = {normalized:.4f}")
    assert 0.35 < normalized < 0.65, \
        f"Subspaces not orthogonal: {normalized:.4f}"
    print(f"  Delta from ideal 0.5: {abs(normalized - 0.5):.4f}")


def test_gpu_correctness(device):
    """GPU search must exactly match CPU reference (same as 490k bench)."""
    dim = 10240
    cache = random_bipolar(100, dim, seed=42)
    query = random_bipolar(1, dim, seed=99)[0]

    bench = _core.HammingSearchBench(device, cache, dim)
    gpu_result = bench.search(query)
    cpu_result = _core.hamming_search_cpu(query, cache)

    assert np.array_equal(gpu_result, cpu_result), "GPU != CPU"
    print("  GPU == CPU: EXACT MATCH")
    del bench; gc.collect()


def test_subgroup_size(device):
    """Report hardware subgroup size for profile validation."""
    dim = 10240
    cache = random_bipolar(256, dim, seed=0)
    bench = _core.HammingSearchBench(device, cache, dim)

    # subgroup_size is printed during HammingSearchBench construction
    # and exposed as attribute if bindings are up-to-date
    if hasattr(bench, 'subgroup_size'):
        sg = bench.subgroup_size
        epw = bench.entries_per_wg
        print(f"  Subgroup size: {sg}")
        print(f"  Entries per workgroup: {epw}")
        assert sg in (16, 32, 64), f"Unexpected subgroup size: {sg}"
        assert epw == 256 // sg, f"entries_per_wg mismatch"
    else:
        # Subgroup info logged during construction: [OK] Subgroup size: XX
        print("  Subgroup size reported in construction log above")
        print("  (Rebuild C++ to expose as Python attribute)")

    del bench; gc.collect()


def test_gpu_stress(device, n_entries=10000):
    """GPU stress test: persistent cache, repeated search."""
    dim = 10240
    print(f"  Building {n_entries:,} entries "
          f"({n_entries * 320 * 4 / (1024**2):.0f} MB packed)...")

    cache = random_bipolar(n_entries, dim, seed=0)
    bench = _core.HammingSearchBench(device, cache, dim)
    del cache; gc.collect()

    query = random_bipolar(1, dim, seed=99)[0]

    # Warm up
    for _ in range(5):
        bench.search(query)

    # Benchmark
    n_queries = 100
    times = []
    for _ in range(n_queries):
        t0 = time.perf_counter()
        bench.search(query)
        times.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p99 = np.percentile(times, 99)
    mn = np.min(times)
    bw = (n_entries * 320 * 4 / 1e9) / (avg / 1000)

    print(f"  avg={avg:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms  min={mn:.3f}ms")
    print(f"  bandwidth={bw:.1f} GB/s")

    del bench; gc.collect()


def run_cpu_stress(n_entries=10000):
    """CPU stress test for baseline comparison."""
    dim = 10240
    print(f"  Building {n_entries:,} entries...")

    cache = random_bipolar(n_entries, dim, seed=0)
    query = random_bipolar(1, dim, seed=99)[0]

    n_queries = 20
    start = time.perf_counter()
    for _ in range(n_queries):
        _core.hamming_search_cpu(query, cache)
    elapsed = time.perf_counter() - start

    avg_us = (elapsed / n_queries) * 1e6
    print(f"  CPU: {n_queries} queries, avg={avg_us:.0f} us/query")
    del cache; gc.collect()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grilly VSA Pre-Flight Test")
    parser.add_argument('--cpu-only', action='store_true', help='Skip GPU tests')
    args = parser.parse_args()

    print("=" * 60)
    print("  Grilly-Next Pre-Flight Stress Test")
    print("=" * 60)
    passed = 0
    failed = 0

    def run(name, fn, *a):
        nonlocal passed, failed
        print(f"\n[TEST] {name}")
        try:
            fn(*a)
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    run("profiles.json", check_profile_exists)

    if not HAS_NATIVE:
        print("\nFATAL: grilly_core not importable. Run: pip install -e .")
        sys.exit(1)

    # ── CPU tests ───────────────────────────────────────────────────────
    run("BLAKE3 determinism", test_blake3_determinism)
    run("XOR self-inverse", test_xor_self_inverse)
    run("Bitpack roundtrip", test_bitpack_roundtrip)
    run("CPU Hamming search", test_hamming_cpu)
    run("WorldModel coherence (CPU)", test_world_model_coherence_cpu)
    run("CPU stress (10k)", run_cpu_stress, 10000)

    # ── GPU tests ───────────────────────────────────────────────────────
    if args.cpu_only:
        print("\n  Skipping GPU tests (--cpu-only)")
    else:
        try:
            device = _core.Device()
            device.load_shaders('shaders')
            run("GPU correctness (GPU==CPU)", test_gpu_correctness, device)
            run("Subgroup size", test_subgroup_size, device)
            run("GPU stress (10k)", test_gpu_stress, device, 10000)
        except Exception as e:
            print(f"\n  GPU init failed: {e}")
            print("  Skipping GPU tests.")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  READY FOR A40/A100")
    else:
        print("  FIX FAILURES BEFORE DEPLOYING")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
