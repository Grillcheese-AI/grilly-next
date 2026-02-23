"""
Benchmark: grilly C++ Vulkan backend vs Python-Vulkan backend vs NumPy.

Measures latency per linear() call across different matrix sizes.
Expected: C++ path 5-20x faster for small/mid tensors (dispatch overhead
dominates), 1.5-3x for large tensors (GPU compute dominates).
"""

import os
import sys
import time

import numpy as np

# ── Try importing backends ──────────────────────────────────────────────────

# C++ backend
try:
    # Add parent dir to path so grilly_core.pyd can be found
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import grilly_core

    cpp_device = grilly_core.Device()
    # Load shaders from sibling grilly repo
    shader_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "shaders",
    )
    if not os.path.isdir(shader_dir):
        # Try sibling grilly repo
        shader_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "grilly", "shaders", "spv",
        )
    cpp_device.load_shaders(shader_dir)
    HAS_CPP = True
    print(f"[OK] C++ backend: {cpp_device.device_name}")
except Exception as e:
    HAS_CPP = False
    print(f"[SKIP] C++ backend not available: {e}")

# Python-Vulkan backend
try:
    from grilly.backend.compute import VulkanCompute

    py_backend = VulkanCompute()
    HAS_PYTHON_VK = True
    print("[OK] Python-Vulkan backend loaded")
except Exception as e:
    HAS_PYTHON_VK = False
    print(f"[SKIP] Python-Vulkan backend not available: {e}")

# ── Benchmark parameters ────────────────────────────────────────────────────

SIZES = [
    (64, 512, 512),
    (128, 768, 768),
    (256, 1024, 1024),
    (512, 2048, 2048),
    (1024, 4096, 4096),
]
WARMUP = 10
ITERS = 100


def bench_numpy(x, w, bias=None):
    """NumPy CPU baseline."""
    out = np.matmul(x, w.T)
    if bias is not None:
        out = out + bias
    return out


def bench_cpp(x, w, bias=None):
    """C++ Vulkan backend."""
    return grilly_core.linear(cpp_device, x, w, bias)


def bench_python_vk(x, w, bias=None):
    """Python-Vulkan backend."""
    return py_backend.fnn.linear(x, w, bias)


def bench_eigen_cpu(x, w, bias=None):
    """Eigen CPU reference (through C++ bindings)."""
    return grilly_core.linear_cpu(x, w, bias)


def time_fn(fn, x, w, bias, warmup, iters):
    """Time a function, returning mean ms per call."""
    for _ in range(warmup):
        fn(x, w, bias)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, w, bias)
    elapsed = time.perf_counter() - t0
    return (elapsed / iters) * 1000  # ms


# ── Run benchmarks ──────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 80)
    print(f"{'Size':>20s} | {'NumPy':>10s} | {'Eigen':>10s} | {'Py-Vk':>10s} | {'C++-Vk':>10s} | {'Speedup':>8s}")
    print("-" * 80)

    for B, M, N in SIZES:
        x = np.random.randn(B, M).astype(np.float32)
        w = np.random.randn(N, M).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)

        # NumPy baseline
        np_ms = time_fn(bench_numpy, x, w, b, WARMUP, ITERS)

        # Eigen CPU
        eigen_ms = time_fn(bench_eigen_cpu, x, w, b, WARMUP, ITERS) if HAS_CPP else float("nan")

        # Python-Vulkan
        py_ms = time_fn(bench_python_vk, x, w, b, WARMUP, ITERS) if HAS_PYTHON_VK else float("nan")

        # C++ Vulkan
        cpp_ms = time_fn(bench_cpp, x, w, b, WARMUP, ITERS) if HAS_CPP else float("nan")

        # Speedup: Python-Vulkan / C++-Vulkan
        if HAS_CPP and HAS_PYTHON_VK and cpp_ms > 0:
            speedup = f"{py_ms / cpp_ms:.1f}x"
        else:
            speedup = "N/A"

        size_str = f"({B},{M},{N})"
        print(
            f"{size_str:>20s} | {np_ms:>8.3f}ms | {eigen_ms:>8.3f}ms | "
            f"{py_ms:>8.3f}ms | {cpp_ms:>8.3f}ms | {speedup:>8s}"
        )

        # Correctness check: C++ vs NumPy
        if HAS_CPP:
            ref = bench_numpy(x, w, b)
            cpp_out = bench_cpp(x, w, b)
            max_diff = np.max(np.abs(ref - cpp_out))
            if max_diff > 1e-2:
                print(f"  WARNING: max diff = {max_diff:.6f}")
            else:
                print(f"  PASS (max diff = {max_diff:.6f})")

    print("=" * 80)

    # Print pool/cache stats
    if HAS_CPP:
        print(f"\nBuffer pool stats: {cpp_device.pool_stats()}")
        print(f"Pipeline cache stats: {cpp_device.cache_stats()}")


if __name__ == "__main__":
    main()
