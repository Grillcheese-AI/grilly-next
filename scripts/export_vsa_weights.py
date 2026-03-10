"""
Export VSA weights from float {-1,+1} .npz to bitpacked uint32 binary format.

Reads a .npz file containing w1 and w2 bipolar float arrays and packs them
into the binary format expected by VSABaremetalEngine.load_logic_weights().

Weight layout in output binary file:
  - w1: hidden_dim x state_words uint32  (e.g., 1024 x 64 for dim=2048)
  - w2: state_dim  x hidden_words uint32 (e.g., 2048 x 32 for dim=2048)
  - Both matrices are concatenated contiguously.

Bitpacking convention:
  +1 -> bit 1, -1 -> bit 0
  Bit b of word w corresponds to element w*32 + b.

Usage:
    python scripts/export_vsa_weights.py --weights model/student/poc_weights.npz --output cubemind_student.bin
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def bitpack_matrix(mat: np.ndarray) -> np.ndarray:
    """Pack a bipolar {-1,+1} float matrix into uint32 bitpacked format.

    Args:
        mat: (rows, cols) float32 with values in {-1, +1}.
             cols must be a multiple of 32.

    Returns:
        packed: (rows, cols // 32) uint32
    """
    rows, cols = mat.shape
    if cols % 32 != 0:
        raise ValueError(f"Column count {cols} is not a multiple of 32")

    # Convert bipolar to bits: +1 -> 1, -1 -> 0
    bits = (mat > 0).astype(np.uint32)

    words = cols // 32
    packed = np.zeros((rows, words), dtype=np.uint32)

    for b in range(32):
        # bits[:, b::32] selects every 32nd element starting at b,
        # giving the b-th bit of each word across all words for each row.
        packed |= bits[:, b::32] << b

    return packed


def main():
    parser = argparse.ArgumentParser(
        description="Export VSA weights from .npz (bipolar float) to bitpacked uint32 binary"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to .npz file with w1 and w2 keys (bipolar float arrays)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output binary file (e.g., cubemind_student.bin)",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not weights_path.exists():
        print(f"ERROR: weights file not found: {weights_path}")
        sys.exit(1)

    # ── Load weights ──────────────────────────────────────────────────
    print(f"Loading weights from {weights_path} ...")
    data = np.load(str(weights_path))

    if "w1" not in data or "w2" not in data:
        print(f"ERROR: .npz must contain 'w1' and 'w2' keys. Found: {list(data.keys())}")
        sys.exit(1)

    w1 = data["w1"]  # (hidden_dim, state_dim)
    w2 = data["w2"]  # (state_dim, hidden_dim)

    print(f"  w1 shape: {w1.shape}  (hidden_dim x state_dim)")
    print(f"  w2 shape: {w2.shape}  (state_dim x hidden_dim)")

    hidden_dim, state_dim = w1.shape
    state_dim_2, hidden_dim_2 = w2.shape

    if state_dim != state_dim_2 or hidden_dim != hidden_dim_2:
        print(f"ERROR: dimension mismatch — "
              f"w1 is ({hidden_dim}, {state_dim}), w2 is ({state_dim_2}, {hidden_dim_2})")
        sys.exit(1)

    # ── Verify bipolarity ─────────────────────────────────────────────
    def check_bipolar(name: str, arr: np.ndarray):
        unique = np.unique(arr)
        non_bipolar = np.sum((arr != 1.0) & (arr != -1.0))
        if non_bipolar > 0:
            print(f"  WARNING: {name} has {non_bipolar} non-bipolar values "
                  f"(unique sample: {unique[:10]})")
            print(f"  Applying sign() to binarize ...")
            arr = np.sign(arr).astype(np.float32)
            arr[arr == 0] = 1.0
        else:
            print(f"  {name}: all values are bipolar {{-1, +1}}")
        return arr

    w1 = check_bipolar("w1", w1)
    w2 = check_bipolar("w2", w2)

    # ── Bitpack ───────────────────────────────────────────────────────
    print(f"\nBitpacking ...")

    state_words = state_dim // 32
    hidden_words = hidden_dim // 32

    w1_packed = bitpack_matrix(w1)  # (hidden_dim, state_words)
    w2_packed = bitpack_matrix(w2)  # (state_dim, hidden_words)

    print(f"  w1 packed: {w1_packed.shape} = {w1_packed.nbytes:,d} bytes "
          f"({hidden_dim} neurons x {state_words} state_words)")
    print(f"  w2 packed: {w2_packed.shape} = {w2_packed.nbytes:,d} bytes "
          f"({state_dim} neurons x {hidden_words} hidden_words)")

    total_bytes = w1_packed.nbytes + w2_packed.nbytes
    print(f"  Total: {total_bytes:,d} bytes ({total_bytes / 1024:.1f} KB)")

    # ── Write binary ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(w1_packed.tobytes())
        f.write(w2_packed.tobytes())

    print(f"\nWritten to {output_path} ({output_path.stat().st_size:,d} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
