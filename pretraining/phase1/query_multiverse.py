"""
Hilbert Multiverse Interactive Query REPL
=========================================

Loads Phase 1 JSONL data, builds the VSA knowledge base, then drops
into an interactive loop where you can query across cognitive universes.

Usage:
  cd grilly-next
  uv run python pretraining/phase1/query_multiverse.py
  uv run python pretraining/phase1/query_multiverse.py --max-rows 1000

Commands inside the REPL:
  <any text>              Query in current universe (with context carry-forward)
  /universe <name>        Switch universe (analytical, empathetic, skeptical, poetic, assertive)
  /all <text>             Query all universes at once
  /steps <n>              Set max generation steps
  /clear                  Clear conversational context
  /fasttext [top_k]       Load FastText fillers from HuggingFace
  /stats                  Show WorldModel and cache stats
  /help                   Show this help
  /quit                   Exit
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import grilly_core

from grilly_next.multiverse import (
    CognitiveUniverse,
    MultiverseGenerator,
)


UNIVERSE_ALIASES = {
    "analytical": CognitiveUniverse.ANALYTICAL,
    "empathetic": CognitiveUniverse.EMPATHETIC,
    "skeptical": CognitiveUniverse.SKEPTICAL,
    "poetic": CognitiveUniverse.POETIC,
    "assertive": CognitiveUniverse.ASSERTIVE,
    # Short aliases
    "a": CognitiveUniverse.ANALYTICAL,
    "e": CognitiveUniverse.EMPATHETIC,
    "s": CognitiveUniverse.SKEPTICAL,
    "p": CognitiveUniverse.POETIC,
    "x": CognitiveUniverse.ASSERTIVE,
}


def load_and_build(data_dir, max_rows, vsa_dim, d_model, K):
    """Load JSONL data, encode, and build the multiverse generator."""
    shader_dir = str(Path(__file__).parent.parent.parent / "shaders" / "spv")
    data_path = Path(data_dir)
    jsonl_files = sorted(data_path.glob("*.jsonl"))

    # Filter out this script and the train script
    jsonl_files = [f for f in jsonl_files if f.suffix == ".jsonl"]

    print("  Loading data...")
    all_triples = []
    for fpath in jsonl_files:
        count = 0
        with open(fpath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                row = json.loads(line)
                svc = row.get("svc", {})
                v = svc.get("v", "").strip()
                if v:
                    all_triples.append({
                        "s": svc.get("s", "").strip(),
                        "v": v,
                        "c": svc.get("c", "").strip(),
                        "text": row.get("text", ""),
                    })
                    count += 1
        print(f"    {fpath.name}: {count:,d} triples")

    print(f"  Total: {len(all_triples):,d} triples")
    print()

    # Initialize GPU
    print("  Initializing GPU...")
    device = grilly_core.Device()
    device.load_shaders(shader_dir)

    world_model = grilly_core.WorldModel(device, dim=vsa_dim)
    vsa_cache = grilly_core.VSACache(device, dim=vsa_dim, max_capacity=100000)
    model = grilly_core.VSAHypernetwork(
        device, d_model=d_model, vsa_dim=vsa_dim, K=K, seed=42
    )

    gen = MultiverseGenerator(
        device, model, world_model, vsa_cache,
        vsa_dim=vsa_dim, K=K,
    )

    # Encode and populate
    print("  Encoding SVC triples...")
    t0 = time.perf_counter()
    inserted = 0
    for triple in all_triples:
        s = triple["s"] if triple["s"] else "_"
        v = triple["v"]
        c = triple["c"] if triple["c"] else "_"
        state = world_model.encode_triple(s, v, c)
        world_model.add_fact(s, v, c)
        text = triple["text"][:100] if triple["text"] else f"{v}({s}, {c})"
        if gen.insert_with_text(text, state):
            inserted += 1

    elapsed = time.perf_counter() - t0
    print(f"  Encoded {len(all_triples):,d} triples in {elapsed:.2f}s")
    print(f"  VSACache: {inserted:,d} entries")
    print(f"  WorldModel: {world_model.fact_count} facts, {world_model.constraint_count} constraints")
    print()

    return gen


def print_trajectory(trajectory, universe_name):
    """Pretty-print a generation trajectory."""
    if not trajectory:
        print(f"    [{universe_name}] (no trajectory generated)")
        return

    for i, step in enumerate(trajectory):
        coherence = f"coh={step.coherence_score:.0f}" if step.coherence_score > 0 else "coh=OK"
        print(f"    {i}: {step.decoded_text}")
    print(f"    [{len(trajectory)} steps, {coherence}]")


def repl(gen, max_steps):
    """Interactive query loop with conversational context carry-forward."""
    current_universe = CognitiveUniverse.ANALYTICAL
    context_state = None  # Carries forward from last generation

    print("=" * 60)
    print("  Hilbert Multiverse Query REPL")
    print("  Type a sentence to query. /help for commands.")
    print(f"  Universe: {current_universe.name} (phi={current_universe.value})")
    print("  Context: XOR carry-forward enabled (use /clear to reset)")
    print("=" * 60)
    print()

    while True:
        try:
            ctx_marker = "*" if context_state is not None else " "
            prompt = f"[{current_universe.name}]{ctx_marker}> "
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        # --- Commands ---
        if line.lower() in ("/quit", "/exit", "/q"):
            print("Bye.")
            break

        if line.lower() == "/help":
            print("  Commands:")
            print("    <text>              Query with context carry-forward")
            print("    /universe <name>    Switch (analytical|empathetic|skeptical|poetic|assertive)")
            print("    /all <text>         Query all 5 universes at once")
            print("    /steps <n>          Set max generation steps")
            print("    /clear              Clear conversational context")
            print("    /fasttext [top_k]   Load FastText fillers from HuggingFace")
            print("    /stats              Show WorldModel/cache stats")
            print("    /quit               Exit")
            print()
            print("  The * after the universe name means context is active.")
            print("  Each query XOR-binds with the prior step's state.")
            print()
            continue

        if line.lower() == "/clear":
            context_state = None
            print("  Context cleared.")
            print()
            continue

        if line.lower().startswith("/fasttext"):
            parts = line.split()
            top_k = 50000
            if len(parts) > 1:
                try:
                    top_k = int(parts[1])
                except ValueError:
                    print("  Usage: /fasttext [top_k]")
                    print()
                    continue

            print(f"  Loading FastText fillers from HuggingFace (top {top_k:,d})...")
            print("  This may take a minute on first download...")
            try:
                t0 = time.perf_counter()
                count = gen.load_fasttext_from_hf(top_k=top_k)
                elapsed = time.perf_counter() - t0
                print(f"  Loaded {count:,d} fillers in {elapsed:.1f}s")
                print(f"  TextEncoder vocab: {gen._text_encoder.vocab_size():,d}")
            except ImportError as e:
                print(f"  {e}")
            except Exception as e:
                print(f"  Error: {e}")
            print()
            continue

        if line.lower().startswith("/universe ") or line.lower().startswith("/u "):
            parts = line.split(None, 1)
            if len(parts) < 2:
                print("  Usage: /universe <name>")
                continue
            name = parts[1].strip().lower()
            if name in UNIVERSE_ALIASES:
                current_universe = UNIVERSE_ALIASES[name]
                print(f"  Switched to {current_universe.name} (phi={current_universe.value})")
            else:
                print(f"  Unknown universe: {name}")
                print(f"  Available: {', '.join(sorted(set(k for k in UNIVERSE_ALIASES if len(k) > 1)))}")
            print()
            continue

        if line.lower().startswith("/steps "):
            try:
                max_steps = int(line.split()[1])
                print(f"  Max steps set to {max_steps}")
            except (IndexError, ValueError):
                print("  Usage: /steps <number>")
            print()
            continue

        if line.lower() == "/stats":
            print(f"  WorldModel facts:       {gen.world_model.fact_count}")
            print(f"  WorldModel constraints: {gen.world_model.constraint_count}")
            print(f"  VSACache size:          {gen.vsa_cache.size()}")
            cache_stats = gen.vsa_cache.stats()
            print(f"  Cache lookups:          {cache_stats.get('total_lookups', 0)}")
            print(f"  Last lookup:            {cache_stats.get('last_lookup_ms', 0):.3f}ms")
            print(f"  TextEncoder vocab:      {gen._text_encoder.vocab_size()}")
            print(f"  Current universe:       {current_universe.name} (phi={current_universe.value})")
            print(f"  Context active:         {'yes' if context_state is not None else 'no'}")
            print(f"  Max steps:              {max_steps}")
            print()
            continue

        if line.lower().startswith("/all "):
            query_text = line[5:].strip()
            if not query_text:
                print("  Usage: /all <text>")
                print()
                continue

            print(f'  Query: "{query_text}"')
            print()

            t0 = time.perf_counter()
            results = gen.query_all_universes(query_text, max_steps=max_steps)
            elapsed = time.perf_counter() - t0

            for universe, trajectory in results.items():
                print(f"  -- {universe.name} (phi={universe.value}) --")
                print_trajectory(trajectory, universe.name)
                print()

            print(f"  [{elapsed*1000:.1f}ms total]")
            print()
            continue

        # --- Default: query with generate_from_prompt (context carry-forward) ---
        t0 = time.perf_counter()
        trajectory = gen.generate_from_prompt(
            line,
            context_state=context_state,
            target_universe=current_universe,
            max_steps=max_steps,
        )
        elapsed = time.perf_counter() - t0

        # Carry forward the last step's state as context for next query
        if trajectory:
            context_state = trajectory[-1].state

        print()
        print_trajectory(trajectory, current_universe.name)
        print(f"  [{elapsed*1000:.1f}ms]")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hilbert Multiverse Interactive Query"
    )
    parser.add_argument(
        "--data-dir", default="pretraining/phase1",
        help="Directory containing .jsonl files",
    )
    parser.add_argument(
        "--max-rows", type=int, default=2000,
        help="Max rows per JSONL file",
    )
    parser.add_argument("--vsa-dim", type=int, default=10240)
    parser.add_argument("--d-model", type=int, default=768)
    # K=32 fits 12GB VRAM (~2GB weights). Use K=64 for 24GB, K=128 for 48GB.
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    print()
    gen = load_and_build(
        args.data_dir, args.max_rows,
        args.vsa_dim, args.d_model, args.K,
    )
    repl(gen, args.steps)


if __name__ == "__main__":
    main()
