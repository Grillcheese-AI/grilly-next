"""
CubeMind Phase 2 Pretraining: Adaptive Gradient Loop
=====================================================

Phase 2 treats the VSA as an Active Controller for the LLM. The 29us
Hamming search produces two signals that modulate the TapeArena gradients:

  1. Surprise:  How novel is this data?  (0 = redundant, 1 = never seen)
  2. Coherence: Does it align with known facts? (1 = valid, -1 = contradiction)

The adaptive learning multiplier combines both:

  lr_multiplier = (1 + surprise) * max(0, coherence)

This means:
  - Redundant data   -> surprise ~0 -> multiplier ~0 -> skip update
  - Garbage/conflict  -> coherence <0 -> multiplier  0 -> ignore
  - Novel + coherent  -> multiplier ~2x -> learn faster

Pipeline (all C++, zero Python allocation):
  1. C++ reads JSONL files via start_with_files() (nlohmann/json parser)
  2. TextEncoder lazily caches BLAKE3 fillers on first token miss
  3. Producer pushes TrainingPayload to ThreadSafeQueue
  4. Consumer pops payload, queries VSACache for surprise
  5. (Phase 2b) WorldModel coherence check modulates gradient

Usage:
  cd grilly-next
  python pretraining/pretrain_phase2.py
  python pretraining/pretrain_phase2.py --data-dir pretraining/phase1 --max-docs 10000
"""

import argparse
import time
from pathlib import Path

import numpy as np


def load_svc_facts(data_dir, max_facts=50000):
    """Extract SVC triples from JSONL data for WorldModel seeding.

    Each JSONL row has an 'svc' field: {"s": "dog", "v": "be", "c": "animal"}
    We extract these as (subject, verb, complement) facts.
    """
    import json

    facts = []
    seen = set()
    for fpath in sorted(Path(data_dir).glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if len(facts) >= max_facts:
                    break
                try:
                    row = json.loads(line)
                    svc = row.get("svc", {})
                    s = svc.get("s", "").strip()
                    v = svc.get("v", "").strip()
                    c = svc.get("c", "").strip()
                    if s and v and c:
                        key = (s, v, c)
                        if key not in seen:
                            seen.add(key)
                            facts.append(key)
                except (json.JSONDecodeError, AttributeError):
                    continue
        if len(facts) >= max_facts:
            break
    return facts


def run_phase2(data_dir, dim=10240, queue_depth=2048, max_docs=None,
               report_interval=10000, cache_capacity=500_000,
               max_facts=50000, use_world_model=True):
    """Run Phase 2 adaptive gradient pretraining."""

    import grilly_core

    print("=" * 70)
    print("  CubeMind Phase 2: Adaptive Gradient Pretraining")
    print("=" * 70)
    print()

    # -- 1. Discover JSONL files -------------------------------------------
    data_path = Path(data_dir)
    jsonl_files = sorted(data_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"  ERROR: No .jsonl files found in {data_dir}")
        return

    file_paths = [str(f) for f in jsonl_files]
    print(f"  Data directory: {data_dir}")
    print(f"  JSONL files:    {[f.name for f in jsonl_files]}")
    print()

    # -- 2. Initialize Vulkan + VSACache -----------------------------------
    print("Initializing Vulkan device...")
    dev = grilly_core.Device()
    dev.load_shaders("shaders/spv")
    print(f"  GPU: {dev.device_name}")
    print()

    # VSACache: stores seen VSA states for surprise computation.
    # On RX 6750 XT: 500K entries. On A40: 4M+ entries.
    print(f"Initializing VSACache (capacity={cache_capacity:,d})...")
    vsa_cache = grilly_core.VSACache(dev, dim=dim, max_capacity=cache_capacity)
    print(f"  Cache VRAM: {cache_capacity * (dim // 8) / 1e6:.1f} MB")
    print()

    # -- 2b. Initialize WorldModel ----------------------------------------
    world_model = None
    if use_world_model:
        print("Initializing WorldModel...")
        world_model = grilly_core.WorldModel(dev, dim=dim)

        # Seed with SVC facts from training data
        print("  Extracting SVC facts from training data...")
        t0 = time.perf_counter()
        facts = load_svc_facts(data_dir, max_facts=max_facts)
        elapsed = time.perf_counter() - t0
        print(f"  Found {len(facts):,d} unique SVC triples ({elapsed:.2f}s)")

        print("  Loading facts into WorldModel...")
        t0 = time.perf_counter()
        for s, v, c in facts:
            world_model.add_fact(s, v, c)
        elapsed = time.perf_counter() - t0
        print(f"  Facts: {world_model.fact_count:,d}  "
              f"Constraints: {world_model.constraint_count:,d}  "
              f"({elapsed:.2f}s)")
        print()

    # -- 3. Start C++ streaming pipeline -----------------------------------
    #
    # Zero Python allocation mode: C++ reads JSONL, parses with nlohmann/json,
    # lazy-caches BLAKE3 fillers, encodes via TextEncoder, pushes to queue.
    # Python only touches the TrainingPayload (a thin pybind11 wrapper).
    #
    print("Starting C++ streaming pipeline...")
    pipeline = grilly_core.TrainingPipeline(
        dim=dim, ft_dim=300, queue_depth=queue_depth)
    pipeline.start_with_files(file_paths)
    print(f"  Queue depth:  {queue_depth}")
    print(f"  VSA dim:      {dim}")
    print(f"  Filler mode:  lazy BLAKE3 (zero startup cost)")
    print()

    # -- 4. Consumer loop: Adaptive Gradient Training ----------------------
    print("=" * 70)
    print("  Training Loop (Adaptive Gradient Consumer)")
    print("=" * 70)
    print()

    consumed = 0
    total_surprise = 0.0
    total_coherence = 0.0
    skipped = 0          # docs where multiplier < threshold (skip update)
    high_value = 0       # docs where multiplier > 1.5 (boosted learning)
    contradictions = 0   # docs where coherence < 0

    # Latency tracking
    pop_times = []
    surprise_times = []
    coherence_times = []
    t_start = time.perf_counter()
    t_last_report = t_start

    while True:
        # -- Pop next payload (blocks until available) ---------------------
        t_pop = time.perf_counter()
        payload = pipeline.pop()
        pop_ms = (time.perf_counter() - t_pop) * 1000

        if payload is None:
            break

        pop_times.append(pop_ms)
        consumed += 1

        if max_docs and consumed > max_docs:
            break

        # -- A. Query VSACache for Surprise (GPU) --------------------------
        #
        # This is the core CubeMind signal: how "novel" is this VSA state?
        #   surprise ~0: model has seen this geometric structure before
        #   surprise ~1: completely new structure, worth learning from
        #
        vsa_data = np.array(payload.vsa_data, dtype=np.uint32)

        t_s = time.perf_counter()
        if vsa_cache.size() > 0:
            lookup_result = vsa_cache.lookup_packed(dev, vsa_data, top_k=1)
            surprise = lookup_result["surprise"]
        else:
            surprise = 1.0  # Empty cache -> everything is novel
        surprise_ms = (time.perf_counter() - t_s) * 1000
        surprise_times.append(surprise_ms)

        # -- B. Coherence Check (WorldModel GPU) ---------------------------
        #
        # The WorldModel compares (S, V, C) triples against known facts.
        # SVC triples flow through from C++ JsonlReader → TrainingPayload.
        # When a payload has SVC data, we encode it as a fact vector and
        # check Hamming distance against both known_facts and constraints.
        #
        # Score = support - violation  (range [-1, +1])
        #   > 0.3:  coherent with known facts → boost learning
        #   < 0.0:  contradicts known facts   → block gradient
        #
        t_c = time.perf_counter()
        if world_model is not None and world_model.fact_count > 0 and payload.has_svc:
            coh_result = world_model.check_coherence(
                dev, payload.svc_subject, payload.svc_verb, payload.svc_complement)
            coherence = coh_result.score
        else:
            coherence = 1.0  # No SVC data or no WorldModel → pass through
        coherence_ms = (time.perf_counter() - t_c) * 1000
        coherence_times.append(coherence_ms)

        # -- C. Adaptive Learning Multiplier -------------------------------
        #
        # lr_multiplier = (1 + surprise) * max(0, coherence)
        #
        # The "Emotional Optimizer" equation:
        #   - Redundant data:  surprise ~0 -> multiplier ~1 (normal)
        #   - Novel + valid:   surprise ~1 -> multiplier ~2 (boosted)
        #   - Contradictory:   coherence <0 -> multiplier  0 (blocked)
        #
        lr_multiplier = (1.0 + surprise) * max(0.0, coherence)

        total_surprise += surprise
        total_coherence += coherence
        if lr_multiplier < 0.1:
            skipped += 1
        elif lr_multiplier > 1.5:
            high_value += 1
        if coherence < 0:
            contradictions += 1

        # -- D. LLM Forward + Backward ------------------------------------
        #
        # TODO: When the LLM is integrated:
        #   logits = model.forward(payload.llm_input_tokens)
        #   loss = compute_loss(logits, targets)
        #   loss.backward(scale=lr_multiplier)
        #   model.optimizer_step()

        # -- E. Insert into VSACache (build surprise memory) ---------------
        #
        # Store this state so future encounters of similar structures
        # register as low-surprise. The cache's surprise threshold
        # filters redundant entries automatically.
        #
        vsa_cache.insert_packed(vsa_data, surprise=surprise, stress=0.0)

        # -- Periodic reporting --------------------------------------------
        now = time.perf_counter()
        if consumed % report_interval == 0 or (now - t_last_report) > 10.0:
            elapsed = now - t_start
            rate = consumed / elapsed if elapsed > 0 else 0
            stats = pipeline.stats()
            avg_surprise = total_surprise / consumed if consumed else 0
            avg_coherence = total_coherence / consumed if consumed else 0
            vocab = pipeline.encoder().vocab_size()
            cache_size = vsa_cache.size()

            sorted_pops = sorted(pop_times[-1000:])
            p50 = sorted_pops[len(sorted_pops) // 2] if sorted_pops else 0
            p99 = sorted_pops[int(len(sorted_pops) * 0.99)] if sorted_pops else 0

            avg_s_ms = np.mean(surprise_times[-1000:]) if surprise_times else 0
            avg_c_ms = np.mean(coherence_times[-1000:]) if coherence_times else 0

            print(f"  [{consumed:>7,d}]  "
                  f"rate={rate:,.0f}/s  "
                  f"surprise={avg_surprise:.3f}  "
                  f"coherence={avg_coherence:.3f}  "
                  f"skip={skipped}  "
                  f"boost={high_value}  "
                  f"contra={contradictions}  "
                  f"cache={cache_size:,d}  "
                  f"vocab={vocab:,d}  "
                  f"pop_p50={p50:.2f}ms  "
                  f"lookup={avg_s_ms:.3f}ms  "
                  f"coher={avg_c_ms:.3f}ms  "
                  f"busy={stats['producer_busy_pct']:.0f}%")
            t_last_report = now

    # -- 5. Summary --------------------------------------------------------
    t_end = time.perf_counter()
    total_elapsed = t_end - t_start
    pipeline.stop()   # Signal producer to stop (don't wait for all 491K if max_docs hit)
    pipeline.join()
    final = pipeline.stats()

    print()
    print("=" * 70)
    print("  Phase 2 Summary")
    print("=" * 70)
    print(f"  Documents processed:  {consumed:,d}")
    print(f"  Total time:           {total_elapsed:.1f}s")
    print(f"  Consumer throughput:  {consumed / total_elapsed:,.0f} docs/sec")
    print(f"  Producer throughput:  {final['encoding_docs_per_sec']:,.0f} docs/sec")
    print(f"  Producer utilization: {final['producer_busy_pct']:.1f}%")
    print(f"  Lazy vocab size:      {pipeline.encoder().vocab_size():,d}")
    print(f"  VSACache entries:     {vsa_cache.size():,d}")
    if world_model is not None:
        print(f"  WorldModel facts:     {world_model.fact_count:,d}")
        print(f"  WorldModel constr:    {world_model.constraint_count:,d}")
    print()
    if consumed:
        print(f"  Surprise (avg):       {total_surprise / consumed:.4f}")
        print(f"  Coherence (avg):      {total_coherence / consumed:.4f}")
        print(f"  Skipped (low value):  {skipped:,d} ({skipped/consumed*100:.1f}%)")
        print(f"  Boosted (high value): {high_value:,d} ({high_value/consumed*100:.1f}%)")
        print(f"  Contradictions:       {contradictions:,d} ({contradictions/consumed*100:.1f}%)")
    print()

    if pop_times:
        sorted_all = sorted(pop_times)
        print(f"  Pop latency P50:      {sorted_all[len(sorted_all) // 2]:.3f} ms")
        print(f"  Pop latency P99:      {sorted_all[int(len(sorted_all) * 0.99)]:.3f} ms")

    if surprise_times:
        sorted_s = sorted(surprise_times)
        print(f"  Surprise query P50:   {sorted_s[len(sorted_s) // 2]:.3f} ms")
        print(f"  Surprise query P99:   {sorted_s[int(len(sorted_s) * 0.99)]:.3f} ms")

    if coherence_times:
        sorted_c = sorted(coherence_times)
        print(f"  Coherence check P50:  {sorted_c[len(sorted_c) // 2]:.3f} ms")
        print(f"  Coherence check P99:  {sorted_c[int(len(sorted_c) * 0.99)]:.3f} ms")

    print("=" * 70)

    return {
        "documents": consumed,
        "elapsed_s": total_elapsed,
        "docs_per_sec": consumed / total_elapsed if total_elapsed > 0 else 0,
        "avg_surprise": total_surprise / consumed if consumed else 0,
        "avg_coherence": total_coherence / consumed if consumed else 0,
        "contradictions": contradictions,
        "vocab_size": pipeline.encoder().vocab_size(),
        "cache_size": vsa_cache.size(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CubeMind Phase 2: Adaptive Gradient Pretraining")
    parser.add_argument("--data-dir",
                        default="pretraining/phase1",
                        help="Directory containing .jsonl files")
    parser.add_argument("--dim", type=int, default=10240,
                        help="VSA dimension")
    parser.add_argument("--queue-depth", type=int, default=2048,
                        help="Producer-consumer queue depth")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Stop after N documents (for quick testing)")
    parser.add_argument("--report-interval", type=int, default=10000,
                        help="Report every N documents")
    parser.add_argument("--cache-capacity", type=int, default=500_000,
                        help="VSACache capacity (entries)")
    parser.add_argument("--max-facts", type=int, default=50_000,
                        help="Max SVC facts to load into WorldModel")
    parser.add_argument("--no-world-model", action="store_true",
                        help="Disable WorldModel coherence checks")
    args = parser.parse_args()

    run_phase2(
        data_dir=args.data_dir,
        dim=args.dim,
        queue_depth=args.queue_depth,
        max_docs=args.max_docs,
        report_interval=args.report_interval,
        cache_capacity=args.cache_capacity,
        max_facts=args.max_facts,
        use_world_model=not args.no_world_model,
    )
