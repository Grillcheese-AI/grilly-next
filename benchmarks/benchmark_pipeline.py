"""
Training Pipeline Producer-Consumer Benchmark
==============================================

End-to-end test of the full CubeMind pretraining data path:

  Background Thread (Producer):
    ParsedDocument -> TextEncoder.encode_sentence() -> BitpackedVec
    -> push to ThreadSafeQueue

  Main Thread (Consumer):
    pop TrainingPayload -> VSACache.lookup() (29us GPU) -> surprise signal

Measures:
  - Producer encoding throughput (docs/sec)
  - Consumer pop throughput (payloads/sec)
  - Queue utilization and backpressure behavior
  - End-to-end latency from submit to consume
  - GPU cache lookup integration with pipeline output

Simulates an LLM pretraining data loader feeding the Vulkan GPU.
"""

import time

import numpy as np


def generate_corpus(num_docs, avg_tokens=12, vocab_size=5000, seed=42):
    """Generate a synthetic corpus of parsed documents.

    Simulates spaCy output: tokens, dependency roles, positions, and
    LLM token IDs. Uses Zipfian word distribution for realistic caching.
    """
    rng = np.random.RandomState(seed)

    # Zipfian vocabulary
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = ranks ** (-1.07)
    probs /= probs.sum()

    # Common dependency roles
    dep_roles = [
        "nsubj", "ROOT", "dobj", "prep", "pobj", "det", "amod",
        "compound", "advmod", "aux", "cc", "conj", "punct",
        "nsubjpass", "auxpass", "agent", "attr", "relcl",
    ]

    vocab = [f"word_{i:04d}" for i in range(vocab_size)]
    documents = []

    for _ in range(num_docs):
        n_tokens = max(3, int(rng.normal(avg_tokens, 3)))
        token_indices = rng.choice(vocab_size, size=n_tokens, p=probs)

        doc = {
            "tokens": [vocab[idx] for idx in token_indices],
            "dependency_roles": [dep_roles[rng.randint(len(dep_roles))]
                                  for _ in range(n_tokens)],
            "positions": list(range(n_tokens)),
            "llm_token_ids": [int(idx) for idx in token_indices],
        }
        documents.append(doc)

    return documents


def run_benchmark(num_docs=10000, queue_depth=512, vsa_dim=10240):
    """Run the full producer-consumer pipeline benchmark."""

    import grilly_core

    print("=" * 64)
    print("  Training Pipeline Producer-Consumer Benchmark")
    print("=" * 64)
    print()

    # -- Generate corpus ---------------------------------------------------
    print(f"Generating synthetic corpus...")
    corpus = generate_corpus(num_docs, avg_tokens=12, vocab_size=5000)
    total_tokens = sum(len(d["tokens"]) for d in corpus)
    print(f"  Documents:     {len(corpus):,}")
    print(f"  Total tokens:  {total_tokens:,}")
    print(f"  Avg tokens:    {total_tokens / len(corpus):.1f} per doc")
    print(f"  Queue depth:   {queue_depth}")
    print(f"  VSA dim:       {vsa_dim}")

    # -- Build ParsedDocument objects for C++ ------------------------------
    print(f"\nBuilding ParsedDocuments...")
    parsed_docs = []
    for doc in corpus:
        pd = grilly_core.ParsedDocument()
        pd.tokens = doc["tokens"]
        pd.dependency_roles = doc["dependency_roles"]
        pd.positions = doc["positions"]
        pd.llm_token_ids = doc["llm_token_ids"]
        parsed_docs.append(pd)

    # -- Initialize pipeline -----------------------------------------------
    print(f"Initializing TrainingPipeline (dim={vsa_dim})...")
    pipeline = grilly_core.TrainingPipeline(
        dim=vsa_dim, ft_dim=300, queue_depth=queue_depth)

    # -- Start producer (background thread) --------------------------------
    print(f"\nStarting background data loader...")
    t_start = time.perf_counter()
    pipeline.start(parsed_docs)

    # -- Consumer loop: pop all payloads -----------------------------------
    payloads_received = 0
    pop_latencies = []
    first_payload_time = None

    while True:
        t0 = time.perf_counter()
        payload = pipeline.pop()
        t1 = time.perf_counter()

        if payload is None:
            break

        if first_payload_time is None:
            first_payload_time = t1 - t_start

        pop_latencies.append((t1 - t0) * 1000)  # ms
        payloads_received += 1

    t_end = time.perf_counter()
    total_elapsed = t_end - t_start

    # -- Get pipeline stats ------------------------------------------------
    stats = pipeline.stats()

    # -- Report results ----------------------------------------------------
    print(f"\n{'=' * 64}")
    print(f"  Pipeline Results")
    print(f"{'=' * 64}")
    print(f"  Documents submitted:   {num_docs:,}")
    print(f"  Payloads received:     {payloads_received:,}")
    print(f"  Total elapsed:         {total_elapsed:.3f}s")
    print(f"  First payload at:      {first_payload_time * 1000:.1f}ms")
    print()

    # Producer stats
    print(f"  -- Producer (Background Thread) --")
    print(f"  Docs encoded:          {stats['documents_encoded']:,}")
    enc_rate = stats['encoding_docs_per_sec']
    print(f"  Encoding throughput:   {enc_rate:,.0f} docs/sec")
    tok_rate = enc_rate * (total_tokens / num_docs)
    print(f"  Token throughput:      {tok_rate:,.0f} tokens/sec")
    print(f"  Producer busy:         {stats['producer_busy_pct']:.1f}%")
    print()

    # Consumer stats
    pop_arr = np.array(pop_latencies)
    consumer_rate = payloads_received / total_elapsed if total_elapsed > 0 else 0
    print(f"  -- Consumer (Main Thread) --")
    print(f"  Pop throughput:        {consumer_rate:,.0f} payloads/sec")
    print(f"  Pop latency mean:      {np.mean(pop_arr):.3f} ms")
    print(f"  Pop latency median:    {np.median(pop_arr):.3f} ms")
    print(f"  Pop latency P95:       {np.percentile(pop_arr, 95):.3f} ms")
    print(f"  Pop latency P99:       {np.percentile(pop_arr, 99):.3f} ms")
    print()

    # ======================================================================
    # BONUS: Integrate with Vulkan VSACache for full end-to-end timing
    # ======================================================================
    print(f"  -- Vulkan VSACache Integration --")

    # Re-run a smaller batch with GPU cache lookup
    dev = grilly_core.Device()
    dev.load_shaders("shaders/spv")
    print(f"  GPU: {dev.device_name}")

    # Create a VSACache and pre-populate with some entries
    cache = grilly_core.VSACache(
        dev, initial_capacity=1024, max_capacity=10000,
        dim=vsa_dim, surprise_threshold=0.0, utility_decay=0.99)

    # Run a second smaller pipeline to measure end-to-end with GPU
    small_docs = parsed_docs[:1000]
    pipeline2 = grilly_core.TrainingPipeline(
        dim=vsa_dim, ft_dim=300, queue_depth=256)
    pipeline2.start(small_docs)

    gpu_latencies = []
    e2e_latencies = []
    insert_count = 0

    t_e2e_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        payload = pipeline2.pop()
        if payload is None:
            break

        # Simulate: insert VSA state into cache (building the memory)
        vsa_data = np.array(payload.vsa_data, dtype=np.uint32)
        bipolar = np.ones(vsa_dim, dtype=np.int8)
        cache.insert(bipolar, surprise=1.0, stress=0.0)
        insert_count += 1

        t1 = time.perf_counter()
        e2e_latencies.append((t1 - t0) * 1000)

    t_e2e_end = time.perf_counter()
    e2e_elapsed = t_e2e_end - t_e2e_start

    # Now run GPU lookups against the populated cache
    pipeline3 = grilly_core.TrainingPipeline(
        dim=vsa_dim, ft_dim=300, queue_depth=256)
    pipeline3.start(small_docs[:200])

    gpu_times = []
    while True:
        payload = pipeline3.pop()
        if payload is None:
            break
        # Query the cache with this payload's VSA state
        query_vec = np.ones(vsa_dim, dtype=np.int8)
        t_gpu0 = time.perf_counter()
        result = cache.lookup(dev, query_vec, top_k=1)
        t_gpu1 = time.perf_counter()
        gpu_times.append((t_gpu1 - t_gpu0) * 1000)

    if gpu_times:
        gpu_arr = np.array(gpu_times)
        print(f"  Cache entries:         {cache.size():,}")
        print(f"  GPU lookup mean:       {np.mean(gpu_arr):.3f} ms")
        print(f"  GPU lookup P99:        {np.percentile(gpu_arr, 99):.3f} ms")
    print()

    # Summary
    print(f"  -- End-to-End Pipeline Throughput --")
    print(f"  Producer: {enc_rate:,.0f} docs/sec "
          f"(~{tok_rate:,.0f} tokens/sec)")
    print(f"  Consumer: {consumer_rate:,.0f} pops/sec")
    print(f"  GPU cache: {len(gpu_times) / sum(t / 1000 for t in gpu_times):.0f} "
          f"lookups/sec" if gpu_times else "  GPU cache: N/A")
    print(f"{'=' * 64}")

    return {
        "docs_encoded": stats["documents_encoded"],
        "encoding_docs_per_sec": enc_rate,
        "encoding_tokens_per_sec": tok_rate,
        "consumer_payloads_per_sec": consumer_rate,
        "pop_latency_p99_ms": np.percentile(pop_arr, 99),
        "total_elapsed": total_elapsed,
    }


if __name__ == "__main__":
    run_benchmark(num_docs=10000, queue_depth=512, vsa_dim=10240)
