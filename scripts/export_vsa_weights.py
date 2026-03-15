"""
Export VSA weights from float {-1,+1} .npz to bitpacked uint32 binary format.

Reads a .npz file containing w1-w4 bipolar float arrays and packs them
into the binary format expected by VSABaremetalEngine.load_logic_weights().

Weight layout in output binary file (4-layer residual MLP):
  - w1: hidden_dim x state_words uint32  (input -> hidden, 256 KB)
  - w2: hidden_dim x hidden_words uint32 (hidden -> hidden, 128 KB)
  - w3: hidden_dim x hidden_words uint32 (hidden -> hidden, 128 KB)
  - w4: state_dim  x hidden_words uint32 (hidden -> output, 256 KB)
  - All matrices concatenated contiguously (768 KB total at dim=2048).

Bitpacking convention:
  +1 -> bit 1, -1 -> bit 0
  Bit b of word w corresponds to element w*32 + b.

Usage:
    python scripts/export_vsa_weights.py --weights model/student/poc_weights.npz
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
        help="Path to .npz file with w1-w4 keys (bipolar float arrays)",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).parent.parent / "model" / "student" / "cubemind_student.bin"),
        help="Output binary file path",
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

    keys = ["w1", "w2", "w3", "w4"]
    for k in keys:
        if k not in data:
            print(f"ERROR: .npz missing key '{k}'. Found: {list(data.keys())}")
            sys.exit(1)

    weights = [data[k] for k in keys]

    for k, w in zip(keys, weights):
        print(f"  {k} shape: {w.shape}")

    # Infer dimensions from w1 (hidden_dim, state_dim)
    hidden_dim, state_dim = weights[0].shape

    # ── Verify bipolarity ─────────────────────────────────────────────
    def check_bipolar(name: str, arr: np.ndarray):
        non_bipolar = np.sum((arr != 1.0) & (arr != -1.0))
        if non_bipolar > 0:
            print(f"  WARNING: {name} has {non_bipolar} non-bipolar values")
            print(f"  Applying sign() to binarize ...")
            arr = np.sign(arr).astype(np.float32)
            arr[arr == 0] = 1.0
        else:
            print(f"  {name}: all values are bipolar {{-1, +1}}")
        return arr

    weights = [check_bipolar(k, w) for k, w in zip(keys, weights)]

    # ── Bitpack ───────────────────────────────────────────────────────
    print(f"\nBitpacking ...")

    packed = [bitpack_matrix(w) for w in weights]

    for k, p in zip(keys, packed):
        print(f"  {k} packed: {p.shape} = {p.nbytes:,d} bytes")

    # Concatenate: w1, w2, w3, w4 contiguous
    combined = np.concatenate([p.ravel() for p in packed])
    total_bytes = combined.nbytes
    print(f"  Total: {total_bytes:,d} bytes ({total_bytes / 1024:.1f} KB)")

    # ── Write binary ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        combined.tofile(f)

    print(f"\nWritten to {output_path} ({output_path.stat().st_size:,d} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
