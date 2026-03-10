#!/usr/bin/env python
"""
VSA End-to-End Inference Demo
------------------------------
Full Vulkan inference pipeline:
  token IDs -> context accumulation (CPU) -> MLP (GPU) -> decode (GPU) -> next token

Demonstrates autoregressive generation using the trained student MLP
with SimHash-based context accumulation and the VSABaremetalEngine.

Usage:
    python scripts/vsa_inference_demo.py --prompt "The quick brown fox"
    python scripts/vsa_inference_demo.py --prompt "Hello world" --max-tokens 20
"""

import argparse
import pathlib
import sys
import time

import numpy as np

try:
    import grilly_core
except ImportError:
    print(
        "ERROR: grilly_core not found. Run: uv pip install -e .",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print(
        "ERROR: transformers not found. Run: pip install transformers",
        file=sys.stderr,
    )
    sys.exit(1)

# -- Paths -----------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHADER_DIR = str(ROOT / "shaders" / "spv")
WEIGHTS_PATH = str(ROOT / "model" / "student" / "cubemind_student.bin")
CODEBOOK_PATH = str(ROOT / "model" / "codebooks" / "vsa_codebook.bin")
VOCAB_PATH = ROOT / "model" / "codebooks" / "vocab.txt"
EMB_PATH = str(ROOT / "model" / "codebooks" / "qwen_embeddings.npy")

# -- Constants --------------------------------------------------------------

STATE_DIM = 2048
HIDDEN_DIM = 1024
WINDOW = 4
SEED = 42
TOKENIZER_MODEL = "Qwen/Qwen2.5-0.5B"

# -- Helpers ----------------------------------------------------------------


def load_vocabulary(path: pathlib.Path) -> list[str]:
    """Load vocabulary from vocab.txt (one word per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_bitpacked_codebook(path: str, words_per_vec: int) -> np.ndarray:
    """Load the bitpacked codebook as a 2D uint32 array.

    Returns (vocab_size, words_per_vec) uint32 array. Only 38MB for 152K tokens
    instead of 1.2GB float codebook + 545MB embeddings.
    """
    raw = np.fromfile(path, dtype=np.uint32)
    return raw.reshape(-1, words_per_vec)


def unpack_bipolar(packed: np.ndarray) -> np.ndarray:
    """Unpack a single uint32 vector to bipolar {-1, +1} float32.

    Inverse of bipolar_to_bitpacked_vec: bit b of word w -> element w*32 + b.
    """
    words = len(packed)
    dim = words * 32
    result = np.empty(dim, dtype=np.float32)
    for b in range(32):
        bits = (packed >> b) & 1
        result[b::32] = bits * 2.0 - 1.0  # 0 -> -1, 1 -> +1
    return result


def context_accumulate(
    codebook_packed: np.ndarray, token_ids: list[int], window: int
) -> np.ndarray:
    """Roll + majority-vote bundle over a sliding window of recent tokens.

    Uses bitpacked codebook — unpacks only the needed tokens (window=4).
    Each token vector is circular-shifted by its recency position (most recent
    token has shift=0), then all are summed and binarized via sign.
    """
    recent = token_ids[-window:]
    dim = codebook_packed.shape[1] * 32
    acc = np.zeros(dim, dtype=np.float32)
    for i, tok_id in enumerate(recent):
        shift = len(recent) - 1 - i  # most recent = shift 0
        vec = unpack_bipolar(codebook_packed[tok_id])
        vec = np.roll(vec, shift)
        acc += vec
    result = np.sign(acc)
    result[result == 0] = 1.0
    return result


def bipolar_to_bitpacked_vec(vec: np.ndarray) -> np.ndarray:
    """Pack a bipolar {-1, +1} float vector into uint32 words.

    Bit layout matches the Vulkan shader expectation: bit b of word w
    corresponds to element w + b * (dim // 32).
    """
    dim = len(vec)
    assert dim % 32 == 0, f"dim must be a multiple of 32, got {dim}"
    bits = (vec > 0).astype(np.uint32)
    words = dim // 32
    packed = np.zeros(words, dtype=np.uint32)
    for b in range(32):
        packed |= bits[b::32] << b
    return packed


# -- Main -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="VSA End-to-End Inference Demo (Vulkan GPU)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help='Input prompt text (default: "The quick brown fox")',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)",
    )
    args = parser.parse_args()

    # -- Validate model files -----------------------------------------------

    missing = []
    for path, name in [
        (WEIGHTS_PATH, "student weights"),
        (CODEBOOK_PATH, "codebook"),
        (str(VOCAB_PATH), "vocabulary"),
    ]:
        if not pathlib.Path(path).exists():
            missing.append((path, name))

    if missing:
        print("ERROR: Required model files are missing:", file=sys.stderr)
        for path, name in missing:
            print(f"  - {name}: {path}", file=sys.stderr)
        print(
            "\nTo generate these files, run the export scripts first:",
            file=sys.stderr,
        )
        print(
            "  python scripts/export_vsa_weights.py   (student weights)",
            file=sys.stderr,
        )
        print(
            "  python scripts/export_vsa_codebook.py  (codebook + vocab + embeddings)",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Tokenize prompt ----------------------------------------------------

    print(f"[TOKENIZE] Loading Qwen2.5 tokenizer ({TOKENIZER_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL, trust_remote_code=True
    )
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    print(f"[TOKENIZE] Prompt: {repr(args.prompt)}")
    print(f"[TOKENIZE] Token IDs ({len(token_ids)}): {token_ids}")

    # -- Load bitpacked codebook for context accumulation --------------------

    words_per_vec = STATE_DIM // 32
    print(f"[CODEBOOK] Loading bitpacked codebook ({words_per_vec} words/vec)...")
    t0 = time.perf_counter()
    codebook_packed = load_bitpacked_codebook(CODEBOOK_PATH, words_per_vec)
    elapsed = time.perf_counter() - t0
    print(f"[CODEBOOK] Loaded {codebook_packed.shape[0]:,} x {codebook_packed.shape[1]} "
          f"uint32 ({codebook_packed.nbytes / 1e6:.1f} MB) in {elapsed:.2f}s")

    # -- Initialize Vulkan engine -------------------------------------------

    print("[INIT] Creating Vulkan device...")
    dev = grilly_core.Device()
    dev.load_shaders(SHADER_DIR)

    print("[INIT] Loading vocabulary...")
    vocabulary = load_vocabulary(VOCAB_PATH)
    print(f"[INIT] {len(vocabulary):,} tokens loaded")

    print(f"[INIT] Creating VSABaremetalEngine (state={STATE_DIM}, hidden={HIDDEN_DIM})...")
    engine = grilly_core.VSABaremetalEngine(
        dev, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM
    )

    print(f"[INIT] Loading logic weights from {pathlib.Path(WEIGHTS_PATH).name}...")
    engine.load_logic_weights(WEIGHTS_PATH)

    print(f"[INIT] Loading codebook from {pathlib.Path(CODEBOOK_PATH).name}...")
    engine.load_codebook(CODEBOOK_PATH, vocabulary)

    assert engine.ready, "Engine failed to initialize — check model files"
    print(f"[READY] state_dim={STATE_DIM}, hidden_dim={HIDDEN_DIM}, "
          f"vocab={engine.vocab_size:,}, window={WINDOW}")

    # -- Autoregressive generation ------------------------------------------

    print()
    print(f"Prompt: {args.prompt}")
    print(f"Generating up to {args.max_tokens} tokens...")
    print("-" * 60)

    generated_words = []
    t_start = time.perf_counter()

    for step in range(args.max_tokens):
        # 1. Context accumulation (CPU): roll + majority-vote bundle
        context_float = context_accumulate(codebook_packed, token_ids, WINDOW)

        # 2. Bitpack the context state to uint32
        context_packed = bipolar_to_bitpacked_vec(context_float)

        # 3. GPU inference: MLP transform + codebook decode
        result = engine.step(dev, context_packed)
        word = result["word"]
        dist = result["distance"]

        # 4. Print token as it is generated
        print(f"  [{step:3d}] d={dist:5d}  {repr(word)}")

        generated_words.append(word)

        if word == "<EOS>":
            break

        # 5. Look up the token ID for the predicted word so we can feed it back
        #    into context accumulation. Search vocab for the word.
        try:
            next_token_id = vocabulary.index(word)
        except ValueError:
            # If the decoded word is not in vocabulary, use token 0 as fallback
            next_token_id = 0

        token_ids.append(next_token_id)

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    n_tokens = len(generated_words)

    # -- Summary ------------------------------------------------------------

    print("-" * 60)
    output_text = "".join(generated_words)
    toks_per_sec = n_tokens / elapsed if elapsed > 0 else float("inf")

    print(f"\nOutput: {output_text}")
    print(f"\n[STATS] {n_tokens} tokens in {elapsed:.3f}s ({toks_per_sec:.1f} tok/s)")
    print(f"[STATS] Context window: {WINDOW}, State dim: {STATE_DIM}, "
          f"Hidden dim: {HIDDEN_DIM}")


if __name__ == "__main__":
    main()
