# grilly-next: AMD-Native Sub-Symbolic AI Pretraining

[![Vulkan Compute](https://img.shields.io/badge/Vulkan-Compute-red.svg)](#) [![C++17](https://img.shields.io/badge/C++-17-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

**grilly-next** is a high-performance, post-Transformer machine learning framework built entirely from scratch in C++. It features a native Vulkan compute backend optimized for AMD hardware and a custom, tape-based automatic differentiation engine.

At the core of `grilly-next` is **CubeMind**, a Vector Symbolic Architecture (VSA) retrieval system that uses Rubik's Cube geometric priors to bypass the extreme API costs and memory bottlenecks of standard dense continuous embeddings (like `ada-002`).



---

## The Problem: The Embeddings Wall
Standard Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) pipelines suffer from three existential bottlenecks driven by dense float embeddings:
1. **The API Tax:** Relying on continuous float embeddings scales linearly and prohibitively with corpus size.
2. **The Memory Bloat:** Dense `float32` FAISS indices require massive server-grade VRAM (e.g., 38GB+ for 490k entries).
3. **The Compute Trap:** Finding semantic similarity requires heavy Cosine Similarity calculus across thousands of floating-point dimensions, bottlenecking GPU memory bandwidth.
4. **Superposition Collapse:** Standard embeddings mash grammar and meaning into a single "vibe," losing the structural reality of complex sentences.

## The Solution: CubeMind Architecture

`grilly-next` solves this by projecting semantic meaning and grammatical structure into strict bipolar hypervectors ($\{-1, +1\}$), fundamentally altering the pretraining loop.



* **Hardware-Native Bitwise Math:** By bitpacking our VSA cache, semantic similarity search degrades from heavy continuous calculus (Cosine) into single-cycle hardware intrinsics (`XOR` + `bitCount`).
* **Zero-Cost Local Retrieval:** The $5.4\text{GB}$ bitpacked `IndexBinaryFlat` cache fits entirely on commodity AMD RDNA 2 hardware (e.g., RX 6750 XT).
* **Surprise-Momentum Optimizer:** A biologically-inspired, active-learning autograd node. The framework queries the geometric memory cache in `<3ms` during the forward pass, aggressively scaling the learning rate only when it encounters structurally novel grammar.

---

## Empirical Validation: MS MARCO Retrieval

CubeMind was benchmarked on the full **MS MARCO** passage ranking dataset--a gold standard dataset consisting of real Bing search queries--using a lightweight 384d semantic embedder projected into our 10240d strict bipolar VSA space.

By replacing $O(N)$ float32 Cosine Similarity with Vulkan hardware-intrinsic Hamming search (`bitCount` + Subgroup Reductions), `grilly-next` achieves state-of-the-art semantic retrieval with zero API costs and a sub-3ms P99 latency.

| Metric | Target | Achieved (CubeMind VSA) |
| :--- | :--- | :--- |
| **Hit Rate @ 5** | > 75% | **87.6%** |
| **Hit Rate @ 10** | -- | **98.6%** |
| **MRR** | -- | **0.5534** |
| **NDCG @ 10** | -- | **0.6564** |
| **P99 Latency** | < 3ms | **2.094 ms** |
| **GPU Shader Time**| -- | **0.029 ms (29us)** |

*Note: The 29us Vulkan compute shader execution completely eliminates the retrieval bottleneck during LLM forward passes, allowing the VSA cache to operate inline with the autograd engine.*

---

## SemanticAssigner: Memoized LSH Projection Cache

To do semantic assignment in C++ at the speeds required for LLM pretraining, the `SemanticAssigner` solves a massive CPU bottleneck. Projecting a 300d dense embedding through a Gaussian random matrix to produce a 10240d bipolar VSA vector costs **3,072,000 FLOPs per word**. At 10K tokens/sec, this would completely saturate the CPU.

Because natural language follows a **Zipfian distribution** (the top 1,000 words cover ~80% of all tokens), the `SemanticAssigner` uses lazy evaluation with a bitpacked memoization cache. Each unique word is projected exactly once, then the GPU-ready `BitpackedVec` is returned from an O(1) hash table lookup on every subsequent access.

### Architecture

```
FAST PATH: shared_lock -> unordered_map lookup -> return BitpackedVec      (~1us)
SLOW PATH: project_to_bipolar() -> vsaBitpack() -> unique_lock -> insert   (~8ms, once per word)
```

Thread-safe via `std::shared_mutex` -- N concurrent readers on the fast path, exclusive lock only when a genuinely new word appears.

### Memoization Benchmark (10K vocab, 100K token stream, Zipf alpha=1.07)

| Method | tokens/sec | Speedup |
| :--- | :--- | :--- |
| Python uncached (numpy) | 5,206 | 1.0x |
| C++ uncached (scalar) | 121 | -- |
| C++ cached (cold start, 92% hit rate) | 1,474 | 0.3x |
| **C++ cached (prewarmed, 100% hits)** | **933,353** | **179x** |

### Cache Hit Rate Convergence (Zipf's Law)

| Tokens Seen | Cache Size | Hit Rate |
| :--- | :--- | :--- |
| 1,000 | 418 | 58.2% |
| 10,000 | 2,307 | 76.9% |
| 50,000 | 5,866 | 88.3% |
| 100,000 | 7,760 | **92.2%** |

Memory footprint: **1,280 bytes per entry** (10240 bits packed into 320 uint32 words). A 100K vocabulary cache occupies just **122 MB**.

---

## Surprise-Momentum Autograd Integration

The `CubeMindSurpriseNode` wires the VSA cache directly into the C++ backward pass via the Wengert list autograd engine. During the forward pass, the `SemanticAssigner` encodes the input and queries the VSA cache. The normalized Hamming distance (surprise) is stored inline in the autograd node on the `TapeArena` bump allocator.

During `loss.backward()`, the Wengert list walk encounters the surprise node and modulates the gradient:

```
grad_modulated = grad_output * (1.0 + alpha * surprise)
```

* **High surprise** (novel sentence) -> multiplier > 1.0 -> learn aggressively
* **Low surprise** (cached, known pattern) -> multiplier approaches floor -> save FLOPs

The full retrieval-to-gradient pipeline runs in under 100us per sentence:

| Stage | Time |
| :--- | :--- |
| Token -> BitpackedVec (SemanticAssigner) | ~1us (cached) |
| vsaBind + vsaBundle (sentence encoding) | ~microseconds |
| VSA cache lookup (Vulkan Hamming search) | 29us |
| Surprise node allocation (TapeArena bump) | O(1) |

---

## Training Pipeline: Producer-Consumer Architecture

The `TrainingPipeline` decouples text parsing from GPU execution using the classic producer-consumer pattern. A background `std::thread` runs the `TextEncoder` (powered by the `SemanticAssigner` cache) to produce `BitpackedVec` payloads, which are pushed to a bounded `ThreadSafeQueue`. The main training thread pops payloads, queries the Vulkan `VSACache`, and executes the forward/backward pass.

```
Producer Thread                     Consumer Thread (GPU)
--------------                     --------------------
ParsedDocument                      pop(TrainingPayload)
  -> TextEncoder.encode_sentence()    -> VSACache.lookup() [29us GPU]
  -> push(TrainingPayload)            -> CubeMindSurpriseNode [TapeArena]
                                      -> LLM forward/backward
                                      -> optimizer.step()
```

### Pipeline Benchmark (10K docs, 115K tokens, RX 6750 XT)

| Stage | Throughput |
| :--- | :--- |
| **Producer** (background encoding) | **1,659 docs/sec** (~19,101 tokens/sec) |
| **Consumer** (pop latency P99) | **1.248 ms** |
| **First payload** latency | **11.1 ms** |
| **GPU cache lookup** | **1,079 lookups/sec** |
| **Producer utilization** | **99.5%** (CPU-bound encoding, zero idle) |

The bounded queue (depth=512) provides automatic backpressure: the producer never outruns the consumer's GPU by more than 512 payloads (~660 KB), preventing unbounded memory growth.

---

## Tech Stack & Implementation Details



* **Vulkan Compute:** Custom GLSL shaders (`hamming-topk.glsl`) using subgroup reductions and Infinity Fabric L1 Local Data Share (LDS) caching to maximize GDDR6 bandwidth.
* **C++ Memory Arena:** An $O(1)$ bump allocator (`TapeArena`) completely bypasses the C++ heap and Python GIL for microsecond graph generation and teardown.
* **Wengert List Autograd:** Sequential allocation order IS the topological order. Backward pass is a simple `prev_in_tape` pointer chase -- no DFS, no BFS, no priority queue.
* **FastText + LSH:** Locality Sensitive Hashing (LSH) random Gaussian projections (seed 42) seamlessly bridge dense semantic gradients into strict bitpacked geometric topology.
* **SemanticAssigner:** Thread-safe memoization cache with `std::shared_mutex` for lock-free concurrent reads. Zipfian convergence eliminates 92%+ of projection overhead.
* **Training Pipeline:** Producer-consumer `ThreadSafeQueue` with bounded capacity and backpressure. Background `std::thread` encodes text while the main thread drives the GPU.

---

## Quick Start

### Prerequisites
* CMake >= 3.20
* Vulkan SDK (optimized for AMD RDNA 2+ and Nvidia Ampere)
* Python 3.10+ (for PyBind11 data loaders)
* Boost 1.82+ (header-only + atomic)

### Installation
Clone the repository and build the C++ core:
```bash
git clone https://github.com/Grillcheese-AI/grilly-next.git
cd grilly-next
git submodule update --init
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Python API Usage
Use the PyBind11 wrapper to feed your LLM pretraining loop:

```python
import grilly_core
import numpy as np

# Initialize Vulkan device and load shaders
dev = grilly_core.Device()
dev.load_shaders("shaders/spv")

# SemanticAssigner: memoized LSH projection cache
assigner = grilly_core.SemanticAssigner(dim=10240, ft_dim=300)

# Register dense embeddings (FastText, MiniLM, etc.)
for token, vec in pretrained_embeddings.items():
    assigner.add_float_vector(token, vec)

# Pre-warm cache for deterministic first-query timing
assigner.prewarm()

# Encode a sentence using TextEncoder (BLAKE3 roles + vsaBind + vsaBundle)
encoder = grilly_core.TextEncoder(dim=10240)
bitpacked_state = encoder.encode_sentence(
    tokens=["vulkan", "compute", "shaders", "bypass", "bottlenecks"],
    dependency_roles=["nsubj", "compound", "nsubj", "ROOT", "dobj"],
    positions=[0, 1, 2, 3, 4],
)

# 29us hardware GPU query -- VSA cache lookup
cache = grilly_core.VSACache(dev, initial_capacity=2048, max_capacity=500_000, dim=10240)
result = cache.lookup(dev, bitpacked_state["data"], top_k=1)
surprise = result["surprise"]
print(f"Surprise: {surprise:.4f} (novelty signal for autograd)")
```

---

## Roadmap

- [x] Core C++ Autograd Engine (TapeArena + Wengert List)
- [x] Bitpacked VSA Encoding & FastText LSH Projection
- [x] Vulkan `atomicMin` Hamming Search Shaders
- [x] MS MARCO empirical validation (87.6% Hit@5, 29us GPU)
- [x] SemanticAssigner memoization cache (933K tokens/sec)
- [x] Surprise-Momentum gradient modulation (CubeMindSurpriseNode)
- [x] Producer-Consumer Training Pipeline (ThreadSafeQueue + DataLoader)
- [ ] JEPA Predictive World Model integration for look-ahead planning
- [ ] Multi-GPU distributed VSA cache synchronization

---

## Contributing
We welcome contributions! Please see our Contributing Guidelines for details on our C++ formatting standards and Vulkan shader testing protocols.

## License
MIT License. See LICENSE for details.
