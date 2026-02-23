"""
CubeMind Phase 1 Pretraining
============================

Loads 491K pre-parsed sentences from JSONL (conversations + instruct),
feeds them through the C++ TrainingPipeline (producer-consumer), and
runs the VSA encoding loop on the RX 6750 XT.

Data format (per JSONL line):
  {
    "text": "Give three tips for staying healthy.",
    "svc":  {"s": "...", "v": "Give", "c": "..."},
    "lemmas": ["give", "three", "tip", "for", "stay", "healthy"],
    "deps":   ["ROOT", "nummod", "dobj", "prep", "pcomp", "acomp"],
    "pos":    ["VERB", "NUM", "NOUN", "ADP", "VERB", "ADJ"],
    "root_verb": "give",
    "realm": "health",
    "complexity": 0.2,
    ...
  }

Pipeline:
  1. Load JSONL → ParsedDocument (lemmas=tokens, deps=dependency_roles)
  2. Register BLAKE3 fillers for all unique tokens (or FastText if available)
  3. Start TrainingPipeline (background C++ thread encodes via TextEncoder)
  4. Consumer loop: pop TrainingPayload, feed to GPU

Usage:
  cd grilly-next
  PYTHONPATH=. python pretraining/pretrain_phase1.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_jsonl(path, max_rows=None):
    """Load JSONL file into list of dicts."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            docs.append(json.loads(line))
    return docs


def jsonl_to_parsed_documents(rows, grilly_core):
    """Convert JSONL rows to C++ ParsedDocument objects."""
    documents = []
    for row in rows:
        doc = grilly_core.ParsedDocument()
        doc.tokens = row.get("lemmas", [])
        doc.dependency_roles = row.get("deps", [])
        doc.positions = list(range(len(doc.tokens)))
        doc.llm_token_ids = []  # Phase 2: add BPE token IDs
        documents.append(doc)
    return documents


def collect_vocab(rows):
    """Collect unique tokens from all rows."""
    vocab = set()
    for row in rows:
        for lemma in row.get("lemmas", []):
            vocab.add(lemma)
    return sorted(vocab)


def register_blake3_fillers(pipeline, vocab, dim):
    """Register BLAKE3 deterministic fillers for all vocabulary tokens."""
    import grilly_core

    encoder = pipeline.encoder()
    registered = 0
    for token in vocab:
        if not encoder.has_filler(token):
            bipolar = grilly_core.blake3_role(f"filler_{token}", dim)
            encoder.add_filler(token, bipolar)
            registered += 1
    return registered


def run_pretraining(data_dir, max_rows=None, dim=10240, queue_depth=2048,
                    report_interval=10000):
    """Run the full Phase 1 pretraining pipeline."""

    import grilly_core

    print("=" * 70)
    print("  CubeMind Phase 1 Pretraining")
    print("=" * 70)
    print()

    # ── 1. Load data ──────────────────────────────────────────────────
    data_path = Path(data_dir)
    jsonl_files = sorted(data_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"  ERROR: No .jsonl files found in {data_dir}")
        return

    print(f"  Data directory: {data_dir}")
    print(f"  JSONL files: {[f.name for f in jsonl_files]}")
    if max_rows:
        print(f"  Max rows per file: {max_rows}")
    print()

    all_rows = []
    for fpath in jsonl_files:
        t0 = time.perf_counter()
        rows = load_jsonl(fpath, max_rows=max_rows)
        elapsed = time.perf_counter() - t0
        all_rows.extend(rows)
        print(f"  Loaded {len(rows):>7,d} rows from {fpath.name}  ({elapsed:.1f}s)")

    print(f"  Total rows: {len(all_rows):,d}")
    print()

    # ── 2. Analyze vocabulary ─────────────────────────────────────────
    t0 = time.perf_counter()
    vocab = collect_vocab(all_rows)
    elapsed = time.perf_counter() - t0
    print(f"  Unique tokens: {len(vocab):,d}  (collected in {elapsed:.2f}s)")

    # Sentence length stats
    lengths = [len(r.get("lemmas", [])) for r in all_rows]
    print(f"  Sentence lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={int(np.median(lengths))}")

    # Realm distribution
    from collections import Counter
    realms = Counter(r.get("realm", "unknown") for r in all_rows)
    print(f"  Realms: {dict(realms.most_common(5))} ...")

    # Complexity distribution
    complexities = [r.get("complexity", 0) for r in all_rows]
    print(f"  Complexity: mean={np.mean(complexities):.2f}, "
          f"std={np.std(complexities):.2f}")
    print()

    # ── 3. Initialize pipeline ────────────────────────────────────────
    print("Initializing C++ TrainingPipeline...")
    pipeline = grilly_core.TrainingPipeline(
        dim=dim, ft_dim=300, queue_depth=queue_depth)

    # Register BLAKE3 fillers for all vocabulary tokens
    t0 = time.perf_counter()
    n_registered = register_blake3_fillers(pipeline, vocab, dim)
    elapsed = time.perf_counter() - t0
    print(f"  Registered {n_registered:,d} BLAKE3 fillers ({elapsed:.2f}s)")
    print(f"  Encoder vocab size: {pipeline.encoder().vocab_size()}")
    print()

    # ── 4. Convert to ParsedDocuments ─────────────────────────────────
    print("Converting to ParsedDocuments...")
    t0 = time.perf_counter()
    documents = jsonl_to_parsed_documents(all_rows, grilly_core)
    elapsed = time.perf_counter() - t0
    print(f"  Converted {len(documents):,d} documents ({elapsed:.2f}s)")
    print()

    # Free raw JSON data
    del all_rows

    # ── 5. Start pipeline (background encoding thread) ────────────────
    print("Starting producer-consumer pipeline...")
    print(f"  Queue depth: {queue_depth}")
    print(f"  VSA dimension: {dim}")
    print()

    pipeline.start(documents)

    # ── 6. Consumer loop: pop payloads and process ────────────────────
    print("=" * 70)
    print("  Training Loop (Consumer)")
    print("=" * 70)
    print()

    consumed = 0
    total_vsa_bytes = 0
    pop_times = []
    t_start = time.perf_counter()
    t_last_report = t_start

    # Latency tracking
    p50_window = []
    p99_window = []

    while True:
        t_pop_start = time.perf_counter()
        payload = pipeline.pop()
        t_pop_end = time.perf_counter()

        if payload is None:
            break

        pop_ms = (t_pop_end - t_pop_start) * 1000
        pop_times.append(pop_ms)
        p50_window.append(pop_ms)
        if len(p50_window) > 1000:
            p50_window.pop(0)

        consumed += 1
        total_vsa_bytes += len(payload.vsa_data) * 4  # uint32 → bytes

        # ── Training step would go here ──────────────────────────────
        # In Phase 2, this is where we'd:
        #   1. Upload vsa_state to GPU VSACache
        #   2. Compute Hamming distance for surprise signal
        #   3. Forward pass LLM with llm_input_tokens
        #   4. CubeMindSurpriseNode backward pass
        #   5. Optimizer step

        # Periodic reporting
        now = time.perf_counter()
        if consumed % report_interval == 0 or (now - t_last_report) > 10.0:
            elapsed = now - t_start
            rate = consumed / elapsed if elapsed > 0 else 0
            stats = pipeline.stats()
            q_size = stats.get("queue_current_size", 0)
            producer_rate = stats.get("encoding_docs_per_sec", 0)
            producer_busy = stats.get("producer_busy_pct", 0)

            sorted_pops = sorted(p50_window)
            p50 = sorted_pops[len(sorted_pops) // 2] if sorted_pops else 0
            p99 = sorted_pops[int(len(sorted_pops) * 0.99)] if sorted_pops else 0

            print(f"  [{consumed:>7,d}/{len(documents):,d}]  "
                  f"consumer={rate:,.0f} docs/s  "
                  f"producer={producer_rate:,.0f} docs/s  "
                  f"queue={q_size:>4d}  "
                  f"pop_p50={p50:.2f}ms  "
                  f"pop_p99={p99:.2f}ms  "
                  f"busy={producer_busy:.0f}%")
            t_last_report = now

    # ── 7. Summary ────────────────────────────────────────────────────
    t_end = time.perf_counter()
    total_elapsed = t_end - t_start
    pipeline.join()
    final_stats = pipeline.stats()

    print()
    print("=" * 70)
    print("  Phase 1 Pretraining Summary")
    print("=" * 70)
    print(f"  Documents processed:   {consumed:,d}")
    print(f"  Total time:            {total_elapsed:.1f}s")
    print(f"  Consumer throughput:   {consumed / total_elapsed:,.0f} docs/sec")
    print(f"  Producer throughput:   {final_stats.get('encoding_docs_per_sec', 0):,.0f} docs/sec")
    print(f"  Producer utilization:  {final_stats.get('producer_busy_pct', 0):.1f}%")
    print(f"  VSA data encoded:      {total_vsa_bytes / 1e6:.1f} MB")
    print(f"  Encoder vocab:         {pipeline.encoder().vocab_size():,d}")

    if pop_times:
        sorted_all = sorted(pop_times)
        print(f"  Pop latency P50:       {sorted_all[len(sorted_all) // 2]:.3f} ms")
        print(f"  Pop latency P99:       {sorted_all[int(len(sorted_all) * 0.99)]:.3f} ms")
        print(f"  Pop latency max:       {max(pop_times):.3f} ms")

    print("=" * 70)

    return {
        "documents": consumed,
        "elapsed_s": total_elapsed,
        "consumer_docs_per_sec": consumed / total_elapsed,
        "vsa_bytes": total_vsa_bytes,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CubeMind Phase 1 Pretraining")
    parser.add_argument("--data-dir",
                        default="pretraining/phase1",
                        help="Directory containing .jsonl files")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows per file (for quick testing)")
    parser.add_argument("--dim", type=int, default=10240,
                        help="VSA dimension")
    parser.add_argument("--queue-depth", type=int, default=2048,
                        help="Producer-consumer queue depth")
    parser.add_argument("--report-interval", type=int, default=10000,
                        help="Report every N documents")
    args = parser.parse_args()

    run_pretraining(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        dim=args.dim,
        queue_depth=args.queue_depth,
        report_interval=args.report_interval,
    )
