"""
Phase 1: Train VSA model on SVC triples, then generate with Multiverse.

Loads JSONL data → encodes SVC triples as VSA states → trains with
VSATrainingLoop (STDP + hippocampal consolidation) → generates
trajectories across cognitive universes (Annex H).

Usage:
  cd grilly-next
  uv run python pretraining/phase1/train_and_generate.py
  uv run python pretraining/phase1/train_and_generate.py --max-rows 500
"""

import json
import time
from pathlib import Path

import numpy as np
import grilly_core

from grilly_next.training import VSATrainingLoop
from grilly_next.multiverse import (
    CognitiveUniverse,
    MultiverseGenerator,
)


def load_svc_triples(jsonl_path, max_rows=None):
    """Load SVC triples from JSONL, filtering to rows with non-empty S+V+C."""
    triples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            row = json.loads(line)
            svc = row.get("svc", {})
            s = svc.get("s", "").strip()
            v = svc.get("v", "").strip()
            c = svc.get("c", "").strip()
            if v:  # Require at least a verb
                triples.append({
                    "s": s, "v": v, "c": c,
                    "text": row.get("text", ""),
                    "realm": row.get("realm", "unknown"),
                })
    return triples


def encode_triple(world_model, s, v, c):
    """Encode an SVC triple into a bitpacked VSA state."""
    return world_model.encode_triple(s if s else "_", v, c if c else "_")


def run_training(
    data_dir="pretraining/phase1",
    max_rows=None,
    vsa_dim=256,
    d_model=32,
    K=8,
    train_steps=200,
    dream_interval=50,
    report_interval=50,
):
    """Run training + generation pipeline."""

    shader_dir = str(Path(__file__).parent.parent.parent / "shaders" / "spv")

    print("=" * 70)
    print("  Hilbert Multiverse: Train + Generate")
    print("=" * 70)
    print()

    # ── 1. Load data ──────────────────────────────────────────────────
    data_path = Path(data_dir)
    jsonl_files = sorted(data_path.glob("*.jsonl"))

    all_triples = []
    for fpath in jsonl_files:
        t0 = time.perf_counter()
        triples = load_svc_triples(fpath, max_rows=max_rows)
        elapsed = time.perf_counter() - t0
        all_triples.extend(triples)
        print(f"  Loaded {len(triples):>7,d} SVC triples from {fpath.name}  ({elapsed:.1f}s)")

    print(f"  Total triples: {len(all_triples):,d}")
    print()

    if len(all_triples) < 2:
        print("  ERROR: Need at least 2 triples for training")
        return

    # ── 2. Initialize GPU + components ────────────────────────────────
    print("  Initializing GPU + VSA components...")
    device = grilly_core.Device()
    device.load_shaders(shader_dir)

    world_model = grilly_core.WorldModel(device, dim=vsa_dim)
    vsa_cache = grilly_core.VSACache(device, dim=vsa_dim, max_capacity=50000)
    model = grilly_core.VSAHypernetwork(
        device, d_model=d_model, vsa_dim=vsa_dim, K=K, seed=42
    )

    # ── 3. Encode SVC triples → VSA states ────────────────────────────
    print("  Encoding SVC triples to VSA states...")
    t0 = time.perf_counter()

    states = []
    texts = []
    for triple in all_triples:
        state = encode_triple(world_model, triple["s"], triple["v"], triple["c"])
        states.append(state)
        texts.append(triple["text"][:128] if triple["text"] else f'{triple["v"]}({triple["s"]}, {triple["c"]})')
        print(len(states))

        # Register SVC triple as a fact (enables coherence checking)
        s = triple["s"] if triple["s"] else "_"
        v = triple["v"]
        c = triple["c"] if triple["c"] else "_"
        world_model.add_fact(s, v, c)

    elapsed = time.perf_counter() - t0
    print(f"  Encoded {len(states):,d} states ({elapsed:.2f}s)")
    print(f"  State shape: {states[0].shape}, dtype: {states[0].dtype}")
    print(f"  WorldModel facts: {world_model.fact_count}")
    print(f"  WorldModel constraints: {world_model.constraint_count}")
    print()

    # ── 4. Populate VSACache for decoding ─────────────────────────────
    print("  Populating VSACache for multiverse decoding...")
    gen = MultiverseGenerator(
        device, model, world_model, vsa_cache,
        vsa_dim=vsa_dim, K=K,
    )

    inserted = 0
    for state, text in zip(states, texts):
        ok = gen.insert_with_text(text, state)
        if ok:
            inserted += 1
    print(f"  Inserted {inserted:,d} states into VSACache")
    print()

    # ── 5. Train with VSATrainingLoop ─────────────────────────────────
    print("=" * 70)
    print("  Training (STDP + Hippocampal Consolidation)")
    print("=" * 70)
    print()

    loop = VSATrainingLoop(
        device, d_model=d_model, vsa_dim=vsa_dim, K=K,
        dream_interval=dream_interval, seed=42,
    )

    actual_steps = min(train_steps, len(states) - 1)
    losses = []
    t_train_start = time.perf_counter()

    for i in range(actual_steps):
        state_t = states[i]
        state_t1 = states[i + 1]
        result = loop.step(state_t, state_t1)
        losses.append(result.loss)

        if (i + 1) % report_interval == 0 or i == actual_steps - 1:
            recent = losses[-report_interval:]
            elapsed = time.perf_counter() - t_train_start
            rate = (i + 1) / elapsed
            dream_info = ""
            if result.dream_report is not None:
                dream_info = (
                    f"  DREAM: {result.dream_report.episodes_replayed} eps, "
                    f"{result.dream_report.new_rules_extracted} rules"
                )
            print(
                f"  [{i+1:>5d}/{actual_steps}]  "
                f"loss={np.mean(recent):.4f}  "
                f"stdp_norm={result.stdp_weight_norm:.2f}  "
                f"wm_facts={result.world_model_fact_count}  "
                f"rate={rate:.0f} steps/s"
                f"{dream_info}"
            )

    t_train_end = time.perf_counter()
    print()
    print(f"  Training complete: {actual_steps} steps in {t_train_end - t_train_start:.1f}s")
    print(f"  Final loss: {np.mean(losses[-20:]):.4f}")
    print()

    # ── 6. Generate with Multiverse ───────────────────────────────────
    print("=" * 70)
    print("  Multiverse Generation (Annex H)")
    print("=" * 70)
    print()

    # Pick a seed state from the training data
    seed_idx = len(states) // 2
    seed_state = states[seed_idx]
    seed_text = texts[seed_idx]
    print(f"  Seed: \"{seed_text}\"")
    print()

    universes_to_test = [
        CognitiveUniverse.ANALYTICAL,
        CognitiveUniverse.EMPATHETIC,
        CognitiveUniverse.SKEPTICAL,
    ]

    for universe in universes_to_test:
        print(f"  -- {universe.name} (phi={universe.value}) --")
        t0 = time.perf_counter()
        trajectory = gen.generate_warped(
            seed_state,
            target_universe=universe,
            max_steps=5,
        )
        elapsed = time.perf_counter() - t0

        for j, step in enumerate(trajectory):
            print(
                f"    step {j}: \"{step.decoded_text}\"  "
                f"(coherence={step.coherence_score:.1f})"
            )
        print(f"    [{len(trajectory)} steps in {elapsed*1000:.1f}ms]")
        print()

    # ── 7. Summary ────────────────────────────────────────────────────
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Triples encoded:     {len(states):,d}")
    print(f"  Training steps:      {actual_steps}")
    print(f"  VSACache size:       {vsa_cache.size()}")
    print(f"  WorldModel facts:    {world_model.fact_count}")
    print(f"  WorldModel constraints: {world_model.constraint_count}")
    print(f"  Universes tested:    {len(universes_to_test)}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hilbert Multiverse: Train + Generate"
    )
    parser.add_argument(
        "--data-dir", default="pretraining/phase1",
        help="Directory containing .jsonl files",
    )
    parser.add_argument(
        "--max-rows", type=int, default=500000,
        help="Max rows per JSONL file (default 500 for quick test)",
    )
    parser.add_argument("--vsa-dim", type=int, default=10240)
    parser.add_argument("--d-model", type=int, default=768)
    # K=32 fits 12GB VRAM (~2GB weights). Use K=64 for 24GB, K=128 for 48GB.
    parser.add_argument("--K", type=int, default=48)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--dream-interval", type=int, default=50)
    parser.add_argument("--report-interval", type=int, default=50)
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        vsa_dim=args.vsa_dim,
        d_model=args.d_model,
        K=args.K,
        train_steps=args.train_steps,
        dream_interval=args.dream_interval,
        report_interval=args.report_interval,
    )
