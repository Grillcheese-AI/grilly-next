#!/usr/bin/env python3
"""CubeMind VSA Ingestion Pipeline.

Takes preprocessed JSONL singles (from grilly_preprocess_v2.py) and produces
training-ready artifacts for the CubeMind architecture:

  1. Merge + quality-filter JSONL inputs
  2. Group sentences by source document for consecutive transition pairs
  3. Encode via C++ TrainingPipeline (TextEncoder: token ⊗ role ⊗ position)
  4. Register SVC triples in WorldModel (coherence constraints)
  5. Build ResonatorNetwork codebook from corpus vocabulary
  6. Save: encoded states (.npy), codebook (.grly), pairs index, manifest

Output directory structure:
  <output_dir>/
    states.npy           # (N, words_per_vec) uint32 — bitpacked VSA states
    pairs_index.npy      # (M, 2) uint32 — transition pair indices
    codebook.grly        # ResonatorNetwork checkpoint
    vocab.json           # token -> index mapping
    manifest.json        # metadata (dims, counts, stats)

Usage:
  cd grilly-next
  uv run python scripts/ingest_cubemind.py \
    --inputs E:/RAW_TEXTS/processed/realm_singles.jsonl \
             /tmp/validate_singles.jsonl \
    --output-dir E:/RAW_TEXTS/cubemind_training \
    --dim 10240

  # Quick test with 1000 records:
  uv run python scripts/ingest_cubemind.py \
    --inputs E:/RAW_TEXTS/processed/realm_singles.jsonl \
    --output-dir /tmp/cubemind_test \
    --dim 256 --max-rows 1000
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ── Quality Filters ──────────────────────────────────────────────────────

MIN_TOKENS = 3            # Skip very short sentences
MAX_TOKENS = 200          # Skip runaway parses
MIN_NONPUNCT_TOKENS = 2   # Must have at least 2 content words
MAX_SINGLE_CHAR_RATIO = 0.5  # OCR garbage detection


def _is_quality(row: dict) -> bool:
    """Return True if the record passes quality filters."""
    lemmas = row.get("lemmas", [])
    n = len(lemmas)
    if n < MIN_TOKENS or n > MAX_TOKENS:
        return False

    # Must have a root verb
    root_verb = row.get("root_verb", "")
    if not root_verb or len(root_verb) < 2:
        return False

    # OCR garbage: too many single-character lemmas
    single_char = sum(1 for l in lemmas if len(l) <= 1)
    if n > 0 and single_char / n > MAX_SINGLE_CHAR_RATIO:
        return False

    # Must have non-punctuation tokens
    pos = row.get("pos", [])
    nonpunct = sum(1 for p in pos if p not in ("PUNCT", "SPACE", "SYM", "X"))
    if nonpunct < MIN_NONPUNCT_TOKENS:
        return False

    return True


# ── JSONL Streaming ──────────────────────────────────────────────────────

def stream_jsonl(paths, max_rows=None):
    """Stream filtered records from one or more JSONL files.

    Yields (row_dict, global_index) tuples.
    """
    total = 0
    for path in paths:
        path = Path(path)
        if not path.exists():
            log.warning("File not found, skipping: %s", path)
            continue
        log.info("Reading %s ...", path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if max_rows and total >= max_rows:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if not _is_quality(row):
                    continue
                yield row, total
                total += 1


_CONV_ID_RE = re.compile(r'^\[([^\]]+)\]')  # Extract [ID_XXXX] from id field


def _extract_source_key(row):
    """Extract a document-level source key for grouping.

    For conversation data where ``source`` is a broad label (e.g. "conversation")
    but ``id`` encodes per-conversation identifiers like ``[ID_5287]_m0_s0``,
    we extract the conversation ID (``ID_5287``) so each conversation is
    accumulated separately during cumulative encoding.
    """
    source = row.get("source", "")
    row_id = row.get("id", "")

    # If the id has a bracketed prefix like [ID_XXXX], use that as the
    # document key — it's more granular than the source field.
    m = _CONV_ID_RE.match(row_id)
    if m:
        return m.group(1)

    # For book-style sources with an id like [hash]_sN, group by the hash
    # prefix (same file = same document).
    if row_id and ']_' in row_id:
        return row_id.split(']')[0] + ']'

    return source or row_id or "unknown"


def load_all_rows(paths, max_rows=None):
    """Load all quality-filtered rows into memory, grouped by source."""
    rows = []
    sources = defaultdict(list)  # source_key -> [index_in_rows]
    realm_counts = Counter()
    skipped = 0
    total_read = 0

    for path in paths:
        path = Path(path)
        if not path.exists():
            log.warning("File not found: %s", path)
            continue
        log.info("Loading %s ...", path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                if max_rows and len(rows) >= max_rows:
                    break
                total_read += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if not isinstance(row, dict):
                    skipped += 1
                    continue
                if not _is_quality(row):
                    skipped += 1
                    continue

                idx = len(rows)
                rows.append(row)
                source_key = _extract_source_key(row)
                sources[source_key].append(idx)
                realm_counts[row.get("realm", "unknown")] += 1

    log.info(
        "Loaded %d quality records (skipped %d / %d total), %d unique sources",
        len(rows), skipped, total_read, len(sources),
    )
    return rows, sources, realm_counts


# ── Vocabulary ───────────────────────────────────────────────────────────

def build_vocab(rows, max_vocab=100000):
    """Build vocabulary from lemma frequency, returning top-N tokens."""
    freq = Counter()
    for row in rows:
        for lemma in row.get("lemmas", []):
            if len(lemma) >= 2:  # Skip single chars
                freq[lemma] += 1
    # Sort by frequency, take top N
    vocab = [token for token, _ in freq.most_common(max_vocab)]
    log.info("Vocabulary: %d tokens (from %d unique)", len(vocab), len(freq))
    return vocab


# ── Cumulative Context Encoding ──────────────────────────────────────────

# Byte popcount lookup table (256 entries, one per byte value)
_POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint32)


def _unpack_to_bipolar(bitpacked, dim):
    """Unpack uint32 bitpacked array to bipolar {-1, +1} float32."""
    bits = np.unpackbits(bitpacked.view(np.uint8), bitorder='little')[:dim]
    return bits.astype(np.float32) * 2.0 - 1.0


def _bipolar_to_bitpack(values, words_per_vec):
    """Threshold float values to bits and pack into uint32."""
    dim = len(values)
    bits = (values > 0).astype(np.uint8)
    padded = np.zeros(words_per_vec * 32, dtype=np.uint8)
    padded[:dim] = bits
    packed = np.packbits(padded, bitorder='little')
    return packed.view(np.uint32)[:words_per_vec].copy()


def apply_cumulative_encoding(states, sources, dim, max_context=0):
    """Convert independent sentence states to cumulative context states.

    Within each document, state_i becomes the accumulated bundle of
    sentences [0..i]. This ensures s_t XOR s_{t+1} captures the marginal
    contribution of sentence t+1 rather than random noise.

    Parameters
    ----------
    states : ndarray (N, words_per_vec) uint32
        Independent sentence encodings (modified in-place).
    sources : dict
        source_key -> [indices] mapping (document grouping).
    dim : int
        VSA dimension.
    max_context : int
        If > 0, apply exponential decay with effective window = max_context
        sentences. Default 0 = no decay (pure summation).
    """
    words_per_vec = states.shape[1]
    decay = (1.0 - 1.0 / max_context) if max_context > 0 else 1.0

    n_docs = 0
    n_states_modified = 0

    for source_key, indices in sources.items():
        if len(indices) < 2:
            continue

        n_docs += 1
        accum = np.zeros(dim, dtype=np.float32)

        for idx in indices:
            bipolar = _unpack_to_bipolar(states[idx], dim)

            if decay < 1.0:
                accum *= decay
            accum += bipolar

            states[idx] = _bipolar_to_bitpack(accum, words_per_vec)
            n_states_modified += 1

    log.info(
        "Cumulative encoding: %d documents, %d states modified (decay=%.3f)",
        n_docs, n_states_modified, decay,
    )
    return states


def compute_pair_hamming(states, pairs_index, dim):
    """Compute actual Hamming distance fraction for each pair (vectorized)."""
    N = len(pairs_index)
    if N == 0:
        return np.array([], dtype=np.float32)

    a = states[pairs_index[:, 0]]
    b = states[pairs_index[:, 1]]
    xor = a ^ b

    byte_counts = _POPCOUNT_LUT[xor.view(np.uint8)]
    popcount = byte_counts.reshape(N, -1).sum(axis=1)

    return (popcount / dim).astype(np.float32)


def filter_pairs_by_hamming(states, pairs_index, dim, lo=0.05, hi=0.40):
    """Filter pairs to those within a Hamming distance band."""
    hamming = compute_pair_hamming(states, pairs_index, dim)
    mask = (hamming >= lo) & (hamming <= hi)
    filtered = pairs_index[mask]

    log.info(
        "Hamming filter [%.2f, %.2f]: %d -> %d pairs (%.1f%% kept)",
        lo, hi, len(pairs_index), len(filtered),
        100 * len(filtered) / max(len(pairs_index), 1),
    )
    if len(hamming) > 0:
        log.info(
            "  Hamming stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            hamming.mean(), hamming.std(), hamming.min(), hamming.max(),
        )
        too_low = (hamming < lo).sum()
        too_high = (hamming > hi).sum()
        if too_low > 0 or too_high > 0:
            log.info(
                "  Dropped: %d too low (< %.2f), %d too high (> %.2f)",
                too_low, lo, too_high, hi,
            )

    return filtered


# ── Transition Pairs ─────────────────────────────────────────────────────

def build_transition_pairs(sources):
    """Build transition pair indices from source-grouped sentences.

    Consecutive sentences within the same source document form natural
    transition pairs (state_t, state_t+1). Cross-document pairs would
    produce meaningless XOR deltas.

    Returns (M, 2) array of (state_t_idx, state_t1_idx).
    """
    pairs = []
    for source_key, indices in sources.items():
        if len(indices) < 2:
            continue
        # Indices are already in document order (insertion order from JSONL)
        for i in range(len(indices) - 1):
            pairs.append((indices[i], indices[i + 1]))

    pairs_arr = np.array(pairs, dtype=np.uint32) if pairs else np.empty((0, 2), dtype=np.uint32)
    log.info(
        "Built %d transition pairs from %d multi-sentence sources",
        len(pairs_arr), sum(1 for v in sources.values() if len(v) >= 2),
    )
    return pairs_arr


# ── Encoding (via C++ TrainingPipeline) ──────────────────────────────────

def encode_via_pipeline(rows, dim=10240, queue_depth=2048):
    """Encode all rows through the C++ TrainingPipeline.

    Returns (N, words_per_vec) uint32 array of bitpacked VSA states.
    """
    import grilly_core

    words_per_vec = (dim + 31) // 32
    N = len(rows)

    # 1. Build vocabulary and register BLAKE3 fillers
    log.info("Initializing TrainingPipeline (dim=%d) ...", dim)
    pipeline = grilly_core.TrainingPipeline(dim=dim, ft_dim=300, queue_depth=queue_depth)
    encoder = pipeline.encoder()

    vocab = set()
    for row in rows:
        for lemma in row.get("lemmas", []):
            vocab.add(lemma)

    t0 = time.perf_counter()
    registered = 0
    for token in sorted(vocab):
        if not encoder.has_filler(token):
            bipolar = grilly_core.blake3_role(f"filler_{token}", dim)
            encoder.add_filler(token, bipolar)
            registered += 1
    log.info(
        "Registered %d BLAKE3 fillers (%.2fs), encoder vocab: %d",
        registered, time.perf_counter() - t0, encoder.vocab_size(),
    )

    # 2. Convert to ParsedDocuments
    log.info("Converting %d rows to ParsedDocuments ...", N)
    documents = []
    for row in rows:
        doc = grilly_core.ParsedDocument()
        doc.tokens = row.get("lemmas", [])
        doc.dependency_roles = row.get("deps", [])
        doc.positions = list(range(len(doc.tokens)))
        doc.llm_token_ids = []
        # Note: ParsedDocument SVC fields are not exposed in pybind11 bindings.
        # SVC triples are handled separately via WorldModel.add_fact().
        documents.append(doc)

    # 3. Start pipeline and consume
    log.info("Starting producer-consumer pipeline ...")
    pipeline.start(documents)

    states = np.zeros((N, words_per_vec), dtype=np.uint32)
    consumed = 0
    t_start = time.perf_counter()
    t_last = t_start

    while True:
        payload = pipeline.pop()
        if payload is None:
            break
        states[consumed] = payload.vsa_data
        consumed += 1

        now = time.perf_counter()
        if consumed % 50000 == 0 or (now - t_last) > 15.0:
            elapsed = now - t_start
            rate = consumed / elapsed if elapsed > 0 else 0
            stats = pipeline.stats()
            log.info(
                "  [%d/%d] consumer=%.0f docs/s  producer=%.0f docs/s  queue=%d",
                consumed, N, rate,
                stats.get("encoding_docs_per_sec", 0),
                stats.get("queue_current_size", 0),
            )
            t_last = now

    pipeline.join()
    elapsed = time.perf_counter() - t_start
    log.info(
        "Encoding complete: %d states in %.1fs (%.0f docs/s)",
        consumed, elapsed, consumed / elapsed if elapsed > 0 else 0,
    )

    if consumed < N:
        log.warning("Only consumed %d / %d — truncating states array", consumed, N)
        states = states[:consumed]

    return states


# ── Codebook Building ────────────────────────────────────────────────────

def build_codebook(vocab, dim=10240):
    """Build a ResonatorNetwork codebook from vocabulary tokens.

    Each token gets a BLAKE3-derived bipolar vector, bitpacked to uint32.
    """
    import grilly_core

    words_per_vec = (dim + 31) // 32
    codebook = np.zeros((len(vocab), words_per_vec), dtype=np.uint32)

    for i, token in enumerate(vocab):
        bipolar = grilly_core.blake3_role(f"filler_{token}", dim)
        codebook[i] = grilly_core.vsa_bitpack(bipolar)

    log.info("Built codebook: %d tokens x %d uint32 words", len(vocab), words_per_vec)
    return codebook


# ── WorldModel Facts ─────────────────────────────────────────────────────

def register_world_model_facts(rows, device, dim=10240, max_facts=500000):
    """Register SVC triples as WorldModel coherence constraints."""
    import grilly_core

    world_model = grilly_core.WorldModel(device, dim=dim)
    registered = 0

    for row in rows:
        if registered >= max_facts:
            break
        svc = row.get("svc", {})
        v = svc.get("v", "").strip()
        if not v:
            continue
        s = svc.get("s", "").strip() or "_"
        c = svc.get("c", "").strip() or "_"
        world_model.add_fact(s, v, c)
        registered += 1

    log.info(
        "WorldModel: %d facts, %d constraints",
        world_model.fact_count, world_model.constraint_count,
    )
    return world_model


def register_world_model_facts_from_svc(svc_data, device, dim=10240, max_facts=500000):
    """Register SVC triples from pre-extracted (svc_dict, realm) tuples.

    Uses add_fact_unchecked() to bypass the O(n^2) surprise check during
    bulk ingestion. Pre-deduplicated data doesn't need per-entry novelty
    filtering — random BLAKE3 vectors have Hamming distance ~dim/2, well
    above the surprise threshold.
    """
    import grilly_core

    world_model = grilly_core.WorldModel(
        device, dim=dim,
        fact_capacity=max_facts,
        constraint_capacity=max_facts,
    )
    registered = 0

    for svc, _realm in svc_data:
        if registered >= max_facts:
            break
        v = svc.get("v", "").strip()
        if not v:
            continue
        s = svc.get("s", "").strip() or "_"
        c = svc.get("c", "").strip() or "_"
        try:
            world_model.add_fact_unchecked(s, v, c)
            registered += 1
        except Exception as e:
            log.warning("WorldModel.add_fact_unchecked failed at %d: %s", registered, e)
            break

        if registered % 10000 == 0:
            log.info("  WorldModel: %d facts registered ...", registered)

    log.info(
        "WorldModel: %d facts, %d constraints",
        world_model.fact_count, world_model.constraint_count,
    )
    return world_model


# ── Main Pipeline ────────────────────────────────────────────────────────

def run_ingestion(
    input_paths,
    output_dir,
    dim=10240,
    max_rows=None,
    max_vocab=50000,
    max_facts=500000,
    queue_depth=2048,
    save_codebook=True,
    max_context=0,
    hamming_lo=0.05,
    hamming_hi=0.40,
):
    """Run the full CubeMind ingestion pipeline."""
    import grilly_core

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shader_dir = str(Path(__file__).parent.parent / "shaders" / "spv")
    words_per_vec = (dim + 31) // 32

    print("=" * 70)
    print("  CubeMind VSA Ingestion Pipeline")
    print("=" * 70)
    print(f"  Inputs:     {[str(p) for p in input_paths]}")
    print(f"  Output dir: {output_dir}")
    print(f"  VSA dim:    {dim} ({words_per_vec} uint32 words)")
    if max_rows:
        print(f"  Max rows:   {max_rows}")
    print()

    t_total = time.perf_counter()

    # ── 1. Load & filter ─────────────────────────────────────────────
    log.info("Phase 1: Loading and filtering records ...")
    rows, sources, realm_counts = load_all_rows(input_paths, max_rows=max_rows)

    if not rows:
        log.error("No records passed quality filter. Aborting.")
        return

    # Stats
    lengths = [len(r.get("lemmas", [])) for r in rows]
    print(f"  Records:      {len(rows):,d}")
    print(f"  Sources:      {len(sources):,d}")
    print(f"  Realms:       {len(realm_counts)}")
    for realm, count in realm_counts.most_common(10):
        print(f"    {realm:30s} {count:>8,d}")
    if len(realm_counts) > 10:
        print(f"    ... and {len(realm_counts) - 10} more")
    print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={int(np.median(lengths))}")
    print()

    # ── 2. Build vocabulary ──────────────────────────────────────────
    log.info("Phase 2: Building vocabulary ...")
    vocab = build_vocab(rows, max_vocab=max_vocab)

    # ── 4. Encode via TrainingPipeline ───────────────────────────────
    log.info("Phase 4: VSA encoding via TrainingPipeline ...")
    states = encode_via_pipeline(rows, dim=dim, queue_depth=queue_depth)

    # ── 4b. Cumulative context encoding ──────────────────────────────
    #  Before: each state = independent sentence encoding (~49% hamming)
    #  After:  each state = accumulated context up to that sentence
    #          XOR deltas capture marginal contribution (~15-20% hamming)
    log.info("Phase 4b: Applying cumulative context encoding ...")

    # Diagnostic: sample hamming before cumulative encoding
    pre_pairs = build_transition_pairs(sources)
    if len(pre_pairs) > 0:
        sample_n = min(5000, len(pre_pairs))
        sample_idx = np.random.default_rng(42).choice(len(pre_pairs), sample_n, replace=False)
        pre_hamming = compute_pair_hamming(states, pre_pairs[sample_idx], dim)
        log.info(
            "  BEFORE cumulative: hamming mean=%.3f std=%.3f (sampled %d pairs)",
            pre_hamming.mean(), pre_hamming.std(), sample_n,
        )

    states = apply_cumulative_encoding(states, sources, dim, max_context=max_context)

    if len(pre_pairs) > 0:
        post_hamming = compute_pair_hamming(states, pre_pairs[sample_idx], dim)
        log.info(
            "  AFTER  cumulative: hamming mean=%.3f std=%.3f",
            post_hamming.mean(), post_hamming.std(),
        )
    del pre_pairs

    # ── 4c. Rebuild pairs with hamming filtering ─────────────────────
    log.info("Phase 4c: Building and filtering transition pairs ...")
    pairs_index = build_transition_pairs(sources)
    pairs_index = filter_pairs_by_hamming(
        states, pairs_index, dim, lo=hamming_lo, hi=hamming_hi,
    )

    # ── 5. Save critical artifacts IMMEDIATELY (before optional steps) ──
    #    States + pairs are the expensive output; save before anything can OOM.
    log.info("Phase 5: Saving states, pairs, vocab, metadata ...")

    states_path = output_dir / "states.npy"
    np.save(str(states_path), states)
    log.info("Saved states: %s  shape=%s  (%.1f MB)",
             states_path, states.shape, states.nbytes / 1e6)

    pairs_path = output_dir / "pairs_index.npy"
    np.save(str(pairs_path), pairs_index)
    log.info("Saved pairs: %s  shape=%s", pairs_path, pairs_index.shape)

    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({token: i for i, token in enumerate(vocab)}, f)
    log.info("Saved vocab: %s (%d tokens)", vocab_path, len(vocab))

    # Realm metadata as JSONL (streams, doesn't build giant list in memory)
    realm_meta_path = output_dir / "realm_meta.jsonl"
    with open(realm_meta_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if i >= len(states):
                break
            svc = row.get("svc", {})
            meta = {
                "realm": row.get("realm", "unknown"),
                "source": row.get("source", ""),
                "complexity": row.get("complexity", 0),
                "n_tokens": row.get("n_tokens", len(row.get("lemmas", []))),
                "svc": {"s": svc.get("s", ""), "v": svc.get("v", ""), "c": svc.get("c", "")},
            }
            f.write(json.dumps(meta) + "\n")
    log.info("Saved realm metadata: %s", realm_meta_path)

    # Free row dicts and states array to reclaim memory before WorldModel.
    # States are already saved to disk — training loads from .npy.
    n_rows = len(rows)
    svc_data = [(r.get("svc", {}), r.get("realm", "unknown")) for r in rows]
    n_states = states.shape[0]
    states_nbytes = states.nbytes
    del rows
    del sources
    del states
    import gc; gc.collect()
    log.info("Freed rows + states array (~%.0f MB reclaimed)", states_nbytes / 1e6)

    # ── 6. Build codebook ────────────────────────────────────────────
    codebook_path = None
    if save_codebook:
        log.info("Phase 6: Building ResonatorNetwork codebook ...")
        codebook = build_codebook(vocab, dim=dim)

        device = grilly_core.Device()
        device.load_shaders(shader_dir)
        resonator = grilly_core.ResonatorNetwork(device, dim=dim)
        resonator.load_codebook(vocab, codebook.ravel())

        codebook_path = str(output_dir / "codebook.grly")
        resonator.save_codebook(codebook_path)
        log.info("Saved codebook: %s (%d tokens)", codebook_path, len(vocab))
        del codebook, resonator
        gc.collect()

    # ── 7. Register WorldModel facts (optional) ────────────────────
    wm_facts = 0
    wm_constraints = 0
    if max_facts > 0:
        log.info("Phase 7: Registering WorldModel facts (max %d) ...", max_facts)
        try:
            device = grilly_core.Device()
            device.load_shaders(shader_dir)
            world_model = register_world_model_facts_from_svc(
                svc_data, device, dim=dim, max_facts=max_facts
            )
            wm_facts = world_model.fact_count
            wm_constraints = world_model.constraint_count
            del world_model
        except Exception as e:
            log.warning("WorldModel registration failed (non-fatal): %s", e)
    else:
        log.info("Phase 7: Skipping WorldModel (--max-facts 0)")
    del svc_data
    gc.collect()

    # ── 8. Save manifest ─────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_total
    manifest = {
        "dim": dim,
        "words_per_vec": words_per_vec,
        "n_states": n_states,
        "n_pairs": int(pairs_index.shape[0]),
        "n_sources": len(realm_counts),
        "vocab_size": len(vocab),
        "max_vocab": max_vocab,
        "world_model_facts": wm_facts,
        "world_model_constraints": wm_constraints,
        "realm_distribution": dict(realm_counts.most_common()),
        "token_length_stats": {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": int(np.median(lengths)),
        },
        "cumulative_encoding": True,
        "cumulative_max_context": max_context,
        "hamming_filter_lo": hamming_lo,
        "hamming_filter_hi": hamming_hi,
        "input_files": [str(p) for p in input_paths],
        "elapsed_seconds": elapsed_total,
        "encoding_mb": states_nbytes / 1e6,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  CubeMind Ingestion Complete")
    print("=" * 70)
    print(f"  States:       {n_states:,d} x {words_per_vec} uint32  ({states_nbytes / 1e6:.1f} MB)")
    print(f"  Pairs:        {pairs_index.shape[0]:,d} transition pairs")
    print(f"  Vocabulary:   {len(vocab):,d} tokens")
    print(f"  WorldModel:   {wm_facts:,d} facts / {wm_constraints:,d} constraints")
    print(f"  Codebook:     {codebook_path if save_codebook else 'skipped'}")
    print(f"  Elapsed:      {elapsed_total:.1f}s")
    print(f"  Output dir:   {output_dir}")
    print("=" * 70)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CubeMind VSA Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full ingestion (all realms + validate):
  uv run python scripts/ingest_cubemind.py \\
    --inputs E:/RAW_TEXTS/processed/realm_singles.jsonl \\
             /tmp/validate_singles.jsonl \\
    --output-dir E:/RAW_TEXTS/cubemind_training

  # Quick test (256D, 1000 rows):
  uv run python scripts/ingest_cubemind.py \\
    --inputs E:/RAW_TEXTS/processed/realm_singles.jsonl \\
    --output-dir /tmp/cubemind_test \\
    --dim 256 --max-rows 1000
""",
    )
    parser.add_argument(
        "--inputs", "-i", nargs="+", required=True,
        help="Input JSONL files (singles from grilly_preprocess_v2.py)",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory for training artifacts",
    )
    parser.add_argument("--dim", type=int, default=10240, help="VSA dimension (default 10240)")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows to process")
    parser.add_argument("--max-vocab", type=int, default=50000, help="Max vocabulary size")
    parser.add_argument("--max-facts", type=int, default=500000, help="Max WorldModel facts")
    parser.add_argument("--queue-depth", type=int, default=2048, help="Pipeline queue depth")
    parser.add_argument("--no-codebook", action="store_true", help="Skip codebook building")
    parser.add_argument(
        "--max-context", type=int, default=0,
        help="Cumulative encoding decay window (0 = no decay, pure sum). "
             "E.g., 10 = effective window of 10 sentences",
    )
    parser.add_argument(
        "--hamming-lo", type=float, default=0.05,
        help="Min Hamming distance for pair filtering (default 0.05)",
    )
    parser.add_argument(
        "--hamming-hi", type=float, default=0.40,
        help="Max Hamming distance for pair filtering (default 0.40)",
    )
    args = parser.parse_args()

    run_ingestion(
        input_paths=[Path(p) for p in args.inputs],
        output_dir=args.output_dir,
        dim=args.dim,
        max_rows=args.max_rows,
        max_vocab=args.max_vocab,
        max_facts=args.max_facts,
        queue_depth=args.queue_depth,
        save_codebook=not args.no_codebook,
        max_context=args.max_context,
        hamming_lo=args.hamming_lo,
        hamming_hi=args.hamming_hi,
    )
