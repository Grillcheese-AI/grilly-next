"""
Export SimHash codebook and Qwen2.5 vocabulary for VSA bare-metal engine.

Pipeline:
  1. Load Qwen2.5-0.5B float embeddings from model/codebooks/qwen_embeddings.npy
  2. SimHash project to bipolar {-1,+1} via random hyperplane hashing
  3. Bitpack to uint32 format
  4. Save as model/codebooks/vsa_codebook.bin
  5. Export Qwen2.5-0.5B tokenizer vocabulary to model/codebooks/vocab.txt

The SimHash projection uses the same seed, chunking, and RNG method as
poc_simhash_vsa.py so that codebook entries are bit-identical to training.

Bitpacking convention (matches export_vsa_weights.py):
  +1 -> bit 1, -1 -> bit 0
  Bit b of word w = element w*32 + b

Binary layout: flattened uint32 array of vocab_size * words_per_vec words.
Token i's bitpacked vector occupies words [i * words_per_vec, (i+1) * words_per_vec).

Usage:
    python scripts/export_vsa_codebook.py
    python scripts/export_vsa_codebook.py --dim 10240 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
EMBEDDINGS_PATH = ROOT / "model" / "codebooks" / "qwen_embeddings.npy"
CODEBOOK_OUT = ROOT / "model" / "codebooks" / "vsa_codebook.bin"
VOCAB_OUT = ROOT / "model" / "codebooks" / "vocab.txt"


# ── SimHash (identical to poc_simhash_vsa.py) ─────────────────────────

def simhash_codebook(embeddings: np.ndarray, vsa_dim: int, seed: int = 42) -> np.ndarray:
    """Project float embeddings to bipolar {-1,+1} via random hyperplane hashing.

    CRITICAL: This must stay in sync with poc_simhash_vsa.simhash_codebook().
    Same RNG, same chunk size, same normalization — so the codebook matches
    the training data exactly.

    Args:
        embeddings: (vocab_size, d_model) float32
        vsa_dim: target bipolar dimension
        seed: random seed for reproducibility

    Returns:
        codebook: (vocab_size, vsa_dim) float32 with values in {-1, +1}
    """
    rng = np.random.default_rng(seed)
    d_model = embeddings.shape[1]

    chunk_size = 1024
    vocab_size = embeddings.shape[0]
    codebook = np.empty((vocab_size, vsa_dim), dtype=np.float32)

    for start in range(0, vsa_dim, chunk_size):
        end = min(start + chunk_size, vsa_dim)
        R = rng.standard_normal((d_model, end - start)).astype(np.float32)
        # Normalize columns for numerical stability
        R /= np.linalg.norm(R, axis=0, keepdims=True)
        proj = embeddings @ R  # (vocab, chunk)
        codebook[:, start:end] = np.sign(proj)

    # Fix zeros (sign(0) = 0) -> snap to +1
    codebook[codebook == 0] = 1.0

    return codebook


# ── Bitpacking ────────────────────────────────────────────────────────

def bitpack_codebook(codebook: np.ndarray) -> np.ndarray:
    """Pack a bipolar {-1,+1} codebook into uint32 bitpacked format.

    Convention: +1 -> bit 1, -1 -> bit 0.
    Bit b of word w corresponds to element w*32 + b.

    Args:
        codebook: (vocab_size, dim) float32, values in {-1, +1}.
                  dim must be a multiple of 32.

    Returns:
        packed: (vocab_size, dim // 32) uint32
    """
    vocab_size, dim = codebook.shape
    if dim % 32 != 0:
        raise ValueError(f"Dimension {dim} is not a multiple of 32")

    # Convert bipolar to bits: +1 -> 1, -1 -> 0
    bits = (codebook > 0).astype(np.uint32)

    words = dim // 32
    packed = np.zeros((vocab_size, words), dtype=np.uint32)

    for b in range(32):
        # bits[:, b::32] selects every 32nd element starting at b,
        # giving the b-th bit of each word across all words for each row.
        packed |= bits[:, b::32] << b

    return packed


# ── Vocabulary Export ─────────────────────────────────────────────────

def export_vocabulary(output_path: Path):
    """Export Qwen2.5-0.5B tokenizer vocabulary to a text file.

    One token per line. Newlines within tokens are replaced with \\n.
    """
    from transformers import AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"  Loading tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    vocab_size = tokenizer.vocab_size
    print(f"  Tokenizer vocab size: {vocab_size}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for token_id in range(vocab_size):
            try:
                token_str = tokenizer.decode([token_id])
            except Exception:
                token_str = f"<UNK_{token_id}>"

            # Replace literal newlines so they don't break the line-per-token format
            token_str = token_str.replace("\n", "\\n")

            f.write(token_str + "\n")

    print(f"  Vocabulary written to {output_path} ({vocab_size} tokens)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export SimHash codebook and Qwen2.5 vocabulary"
    )
    parser.add_argument(
        "--dim", type=int, default=2048,
        help="VSA bipolar dimension (default: 2048, must be multiple of 32)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for SimHash projection (default: 42)",
    )
    args = parser.parse_args()

    if args.dim % 32 != 0:
        print(f"ERROR: --dim must be a multiple of 32, got {args.dim}")
        sys.exit(1)

    print("=" * 60)
    print("  VSA Codebook & Vocabulary Export")
    print("=" * 60)
    print()

    # ── 1. Load embeddings ────────────────────────────────────────────
    print("[1/4] Loading Qwen2.5 embeddings ...")
    if not EMBEDDINGS_PATH.exists():
        print(f"  ERROR: {EMBEDDINGS_PATH} not found.")
        print(f"  Run: python scripts/extract_qwen_embeddings.py")
        sys.exit(1)

    embeddings = np.load(str(EMBEDDINGS_PATH))
    vocab_size, d_model = embeddings.shape
    print(f"  Shape: {embeddings.shape}  ({embeddings.nbytes / 1e6:.0f} MB)")

    # ── 2. SimHash projection ─────────────────────────────────────────
    print(f"\n[2/4] SimHash projection ({d_model} -> {args.dim}, seed={args.seed}) ...")
    codebook = simhash_codebook(embeddings, args.dim, seed=args.seed)
    print(f"  Codebook: {codebook.shape}")

    # Verify all values are bipolar
    n_nonbipolar = np.sum((codebook != 1.0) & (codebook != -1.0))
    if n_nonbipolar > 0:
        print(f"  WARNING: {n_nonbipolar} non-bipolar values found (should be 0)")
    else:
        print(f"  All values are bipolar {{-1, +1}}")

    del embeddings  # Free memory

    # ── 3. Bitpack and save codebook ──────────────────────────────────
    print(f"\n[3/4] Bitpacking codebook ...")
    words_per_vec = args.dim // 32
    packed = bitpack_codebook(codebook)
    print(f"  Packed shape: {packed.shape} = {packed.nbytes:,d} bytes "
          f"({vocab_size} tokens x {words_per_vec} words)")

    CODEBOOK_OUT.parent.mkdir(parents=True, exist_ok=True)
    packed.tofile(str(CODEBOOK_OUT))
    print(f"  Written to {CODEBOOK_OUT} ({CODEBOOK_OUT.stat().st_size:,d} bytes)")

    # ── 4. Export vocabulary ──────────────────────────────────────────
    print(f"\n[4/4] Exporting vocabulary ...")
    export_vocabulary(VOCAB_OUT)

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  DONE")
    print(f"  Codebook: {CODEBOOK_OUT}")
    print(f"    {vocab_size} tokens x {args.dim}d -> {vocab_size} x {words_per_vec} uint32")
    print(f"    {CODEBOOK_OUT.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"  Vocabulary: {VOCAB_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
