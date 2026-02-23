"""
Benchmark: raw dispatch latency overhead.

Measures the fixed cost of a single GPU dispatch (without meaningful compute).
This isolates the Python→C boundary crossing overhead that grilly-cpp eliminates.

Uses a tiny 1x1 linear to minimize GPU compute time — what remains is pure
dispatch overhead: buffer alloc/upload, pipeline bind, descriptor set, submit,
fence wait, download.
"""

import os
import sys
import time

import numpy as np

# ── Import backends ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import grilly_core
    cpp_device = grilly_core.Device()
    shader_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "shaders",
    )
    if not os.path.isdir(shader_dir):
        shader_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "grilly", "shaders", "spv",
        )
    cpp_device.load_shaders(shader_dir)
    HAS_CPP = True
except Exception as e:
    HAS_CPP = False
    print(f"[SKIP] C++ backend: {e}")

try:
    from grilly.backend.compute import VulkanCompute
    py_backend = VulkanCompute()
    HAS_PY = True
except Exception as e:
    HAS_PY = False
    print(f"[SKIP] Python backend: {e}")


def main():
    # Tiny tensors to isolate dispatch overhead
    x = np.ones((1, 16), dtype=np.float32)
    w = np.ones((16, 16), dtype=np.float32)

    WARMUP = 50
    ITERS = 1000

    print(f"\nRaw dispatch latency ({ITERS} iterations, 1x16 @ 16x16 linear)")
    print("-" * 50)

    if HAS_PY:
        for _ in range(WARMUP):
            py_backend.fnn.linear(x, w)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            py_backend.fnn.linear(x, w)
        py_us = (time.perf_counter() - t0) / ITERS * 1e6
        print(f"Python-Vulkan:  {py_us:>8.1f} us/dispatch")

    if HAS_CPP:
        for _ in range(WARMUP):
            grilly_core.linear(cpp_device, x, w)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            grilly_core.linear(cpp_device, x, w)
        cpp_us = (time.perf_counter() - t0) / ITERS * 1e6
        print(f"C++ Vulkan:     {cpp_us:>8.1f} us/dispatch")

    if HAS_PY and HAS_CPP:
        print(f"Speedup:        {py_us / cpp_us:>8.1f}x")

    # Memory leak check: run many iterations and check pool stats
    if HAS_CPP:
        print(f"\nMemory leak test (10,000 dispatches)...")
        for _ in range(10_000):
            grilly_core.linear(cpp_device, x, w)
        stats = cpp_device.pool_stats()
        print(f"  Pool stats: {stats}")
        # If no leak, total_acquired should approximately equal total_released
        # (within the warm pool size)
        diff = stats["total_acquired"] - stats["total_released"]
        if diff > 100:
            print(f"  WARNING: Possible leak: {diff} unreleased buffers")
        else:
            print(f"  PASS: No leak detected (delta={diff})")


if __name__ == "__main__":
    main()
