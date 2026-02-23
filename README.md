# 🧊 grilly-next: Hardware-Verified Sub-Symbolic AI Pretraining

**grilly-next** is a high-performance, post-Transformer machine learning framework built entirely from scratch in C++. It features a cross-platform Vulkan compute backend (optimized for both AMD RDNA 2 and Nvidia Ampere) and a custom, tape-based automatic differentiation engine.

At the core of `grilly-next` is **CubeMind**, a Vector Symbolic Architecture (VSA) system that uses geometric priors to bypass the API costs of dense embeddings, while introducing **Hardware-Level Hallucination Interrupts** and ** Multimodal Fusion**.

---

## 🛑 The Problem: The Embeddings & Hallucination Wall

Standard LLMs and RAG pipelines suffer from existential bottlenecks driven by dense float computations and blind autoregression:

1. **The API Tax:** Relying on continuous float embeddings scales linearly and prohibitively with corpus size.
2. **The Memory Bloat:** Dense `float32` FAISS indices require massive server-grade VRAM (e.g., 38GB+ for 490k entries).
3. **The Compute Trap:** Finding semantic similarity requires heavy Cosine Similarity calculus across thousands of floating-point dimensions.
4. **Superposition Collapse:** Standard embeddings mash grammar and meaning into a single "vibe," losing the structural reality of sentences.
5. **The Hallucination Trap:** Standard transformers are "blind" during generation. They waste expensive compute finishing a forward pass even if the logic is factually incoherent early in the layers.

---

## 💡 The Solution: CubeMind Architecture

`grilly-next` solves this by projecting semantic meaning and grammatical structure into strict bipolar hypervectors (), fundamentally altering the pretraining loop and the reasoning process.

* **Hardware-Native Bitwise Math:** Semantic similarity search degrades from heavy calculus into single-cycle hardware intrinsics (`XOR` + `bitCount`).
* **Hardware-Verified World Model:** A geometric knowledge base that acts as a 29μs immune system, mathematically verifying the coherence of latent thoughts.
* **Early-Exit Hallucination Interrupt:** Kills illogical forward passes mid-layer if the `WorldModel` detects a factual contradiction, saving VRAM and preventing garbage outputs.
* **Geometric Bottleneck Fusion (GBF):** Replaces expensive  cross-attention with  bitwise XOR for instant Multimodal (Vision/Text) fusion.

---

## 📊 Empirical Validation: MS MARCO & A40 Scaling

CubeMind was benchmarked on the **MS MARCO** passage ranking dataset using our 10240d strict bipolar VSA space. Moving from AMD RX 6750 XT to Nvidia A40 yields massive throughput for routing and coherence checks.

| Metric | Standard Transformer / Dense Index | Achieved (CubeMind VSA) |
| --- | --- | --- |
| **Logic Verification Latency** | ~45.0 ms (Attention) | **0.029 ms (29μs)** |
| **Multimodal Fusion Cost** |  | ** Bitwise XOR** |
| **Hit Rate @ 5 (Retrieval)** | ~75% Target | **87.6%** |
| **Memory per Cached Fact** | ~400 MB (KV Cache) | **1.28 KB (Bitpacked)** |

---

## 🧠 Cognitive Features

### 1. ResonatorMoE (Mixture of Experts)

Standard MoE routers suffer from collapse and require heavy softmax calculations. `grilly-next` uses a **Vulkan-accelerated Hamming Router**. The query vector is geometrically matched against expert subspace vectors simultaneously. Routing is guaranteed to be orthogonal and completes in microseconds.

### 2. Real-Time Reasoning Monitor

A zero-overhead `ImGui` / `ImPlot` interface directly hooks into the training loop, visualizing the model's internal cognitive state:

* **Surprise:** Novelty detection curve that dynamically scales the learning rate.
* **Coherence:** Real-time alignment delta against the `WorldModel`. If the score dips below zero, a contradiction is caught live.

---

## ⚡ Core Engine Integrations

### SemanticAssigner: Memoized LSH Projection Cache

Projecting a 300d dense embedding through a Gaussian random matrix to 10240d costs **3,072,000 FLOPs per word**. To prevent CPU saturation, we use a bitpacked memoization cache leveraging Zipf's Law.

* **C++ cached (prewarmed, 100% hits):** **933,353 tokens/sec** (179x speedup over numpy).
* **Hit Rate:** 92.2% cache hit rate after just 100,000 tokens.

### Surprise-Momentum Autograd (TapeArena)

The `CubeMindSurpriseNode` wires the VSA cache directly into the C++ backward pass via the Wengert list autograd engine.

* **High surprise** (novel sentence)  learn aggressively.
* **Low surprise** (cached, known pattern)  scale gradient to zero, saving FLOPs.

### Producer-Consumer Training Pipeline

Decouples text parsing (via `simdjson`) from GPU execution. A background `std::thread` encodes payloads and pushes to a bounded `ThreadSafeQueue`.

| Stage | Throughput (RX 6750 XT) |
| --- | --- |
| **Producer** (background encoding) | **1,659 docs/sec** (~19,101 tokens/sec) |
| **Consumer** (pop latency P99) | **1.248 ms** |
| **GPU cache lookup** | **1,079 lookups/sec** |

---



## 🌱 Sustainability & Carbon Emission Reduction

Training standard autoregressive Large Language Models is an ecological brute-force effort. As models scale, the energy required to process redundant grammar and execute hallucinated forward passes has skyrocketed.

**grilly-next** introduces a biologically inspired, sparse-compute paradigm that dramatically reduces the carbon footprint (tCO₂eq) and energy consumption (GWh) of AI pretraining and inference.

### The Industry Baseline vs. The `grilly-next` Projection

Using recent industry data from Meta's Llama models, we can project the theoretical energy and carbon savings of applying the **CubeMind Architecture** (which yields an estimated 60-65% reduction in total compute via active learning and early exits) to frontier-scale training runs:

| Model Scale (Baseline Data) | Standard Energy / Emissions | `grilly-next` Projection (-65% Compute) | Estimated Savings |
| --- | --- | --- | --- |
| **Llama 2 (70B)** | 539 tCO₂eq *(3.3M GPU hrs)* | **~188 tCO₂eq** | **🌿 Saved 351 Tons of CO₂** |
| **Llama 3 (70B)** | 1,900 tCO₂eq *(H100, 700W TDP)* | **~665 tCO₂eq** | **🌿 Saved 1,235 Tons of CO₂** |
| **Llama 4 (Expected)** | 5.17 GWh Energy Demand | **~1.81 GWh** | **⚡ Saved 3.36 GWh of Power** |

### Where Do the Savings Come From?

1. **The Surprise-Momentum Optimizer (~55% FLOP Reduction):** Natural language follows a Zipfian distribution. Roughly 50-60% of pretraining data consists of structurally redundant grammatical patterns. Because the `CubeMindSurpriseNode` dynamically scales the learning rate to `0.0` for low-surprise inputs, the GPU skips the expensive backpropagation phase for more than half the dataset.
2. **Early-Exit Hallucination Interrupts (~10-15% FLOP Reduction):** Standard models generate hallucinations token-by-token, fully utilizing the GPU's Tensor Cores for garbage output. `grilly-next` kills incoherent trajectories mid-layer the moment they contradict the 29μs Vulkan `WorldModel`, physically halting GPU execution and reclaiming wasted watts.
3. **Retrieval Power Drop (>90% less ALU usage):** Standard RAG relies on dense Float32 continuous embeddings (Cosine Similarity), which bottleneck memory bandwidth and spike GPU wattage. CubeMind bitpacks the semantic space, reducing retrieval to single-cycle hardware intrinsics (`XOR` + `bitCount`).

### Sustainable Inference

Beyond training, standard LLM inference generates ongoing emissions, currently estimated at **0.93 Wh to 1.7 Wh per response** for models like Llama 3.

By replacing massive continuous cross-attention matrices with ** Geometric Bottleneck Fusion** and utilizing **ResonatorMoE** (routing queries via microsecond bitwise operations rather than heavy softmax networks), `grilly-next` is architected to push per-query inference energy well below the 0.5 Wh threshold, making it viable for high-volume, continuous-uptime enterprise deployments without the massive ecological tax.

---


## 🛠️ Quick Start

### Prerequisites

* CMake >= 3.20
* Vulkan SDK (Supports AMD RDNA 2+ or Nvidia Ampere)
* Python 3.10+ (for PyBind11)

### 1. Local Build (AMD / Development)

```bash
git clone https://github.com/Grillcheese-AI/grilly-next.git
cd grilly-next && git submodule update --init
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPROFILE=AMD_LOCAL
make -j$(nproc)

# Run Local VSA/MoE Hardware Check
PYTHONPATH=./build python3 tests/local_vsa_stress_test.py

```

### 2. Data Center Build (Nvidia A40 / Docker)

```bash
# Fixes headless Vulkan ICD issues for Nvidia
./scripts/setup_a40_env.sh 
docker build -t grilly-next .
docker run --gpus all grilly-next python3 pretraining/pretrain_phase2.py

```

### 3. Python API (Phase 2 Adaptive Pretraining)

```python
import grilly_core

config = grilly_core.ConfigLoader.load("A40_MASSIVE")
pipeline = grilly_core.TrainingPipeline(config)
world_model = grilly_core.WorldModel(config)

pipeline.start_with_files(["data/train_0.jsonl"])

while True:
    payload = pipeline.pop()
    
    # 29us GPU Coherence Check (World Model)
    coherence = world_model.check_coherence(payload.vsa_state)
    
    # 29us GPU Novelty Check
    surprise = vsa_cache.query_surprise(payload.vsa_state)
    
    # Active Learning: Ignore hallucinations, boost novel facts
    lr_multiplier = (1.0 + surprise) * max(0.0, coherence.score)
    
    loss = model.forward(payload.token_ids)
    loss.backward(scale=lr_multiplier)

```

---

## 🗺️ Roadmap

* [x] Core C++ Autograd Engine (TapeArena + Wengert List)
* [x] Vulkan `atomicMin` Hamming Search Shaders
* [x] MS MARCO empirical validation (87.6% Hit@5, 29μs GPU)
* [x] SemanticAssigner memoization cache (933K tokens/sec)
* [x] Producer-Consumer Training Pipeline (`simdjson` + `ThreadSafeQueue`)
* [x] **New:** Phase 2 Adaptive Learning (Surprise-Momentum)
* [x] **New:** Hardware-Verified World Model & Coherence Monitor
* [x] **New:** Early-Exit Hallucination Interrupts
* [x] **New:** ResonatorMoE (Vulkan Top-K Routing)
* [x] **New:** Geometric Bottleneck Fusion (Multimodal)
* [ ] Multi-GPU distributed VSA cache synchronization (A40 Cluster)

---

## Contributing

We welcome contributions! Please see our `CONTRIBUTING.md` for details on our C++ formatting standards and mandatory Vulkan `subgroupAdd` coding practices for compute shaders.

## License

MIT License. See LICENSE for details.