"""
SimHash VSA Next-Token Prediction POC
──────────────────────────────────────
Proves that SimHash-projected Qwen2.5 embeddings + VSA context
accumulation make next-token prediction learnable.

Pipeline:
  1. Load Qwen2.5 float embeddings -> SimHash to bipolar codebook
  2. Build context VSA states via permutation + bundling
  3. Train 2-layer MLP (grilly autograd) to predict binding transform
  4. Eval: Hamming similarity of predicted vs true next-token code

Usage:
    python scripts/poc_simhash_vsa.py --steps 2000
    python scripts/poc_simhash_vsa.py --steps 0          # embedding sanity check only
    python scripts/poc_simhash_vsa.py --context-window 4  # tune context size
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numba as nb
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
EMBEDDINGS_PATH = ROOT / "model" / "codebooks" / "qwen_embeddings.npy"
TRAJECTORIES_PATH = ROOT / "model" / "teacher" / "teacher_golden_trajectories.jsonl"

# ── Configuration ─────────────────────────────────────────────────────

DIM = 2048           # VSA dimension (reduced for POC; scale to 10240 after validation)
HIDDEN = 1024        # MLP hidden dimension
BATCH_SIZE = 512     # Training batch size
LR = 1e-3            # Learning rate
GRAD_CLIP = 5.0      # Gradient clipping norm
EVAL_EVERY = 100     # Print metrics every N steps
EVAL_SAMPLES = 512   # Number of samples for evaluation


# ── SimHash ───────────────────────────────────────────────────────────

def simhash_codebook(embeddings: np.ndarray, vsa_dim: int, seed: int = 42) -> np.ndarray:
    """Project float embeddings to bipolar {-1,+1} via random hyperplane hashing.

    SimHash preserves cosine similarity as Hamming similarity:
        P(h(a) == h(b)) = 1 - arccos(cos(a,b)) / pi

    Args:
        embeddings: (vocab_size, d_model) float32
        vsa_dim: target bipolar dimension
        seed: random seed for reproducibility

    Returns:
        codebook: (vocab_size, vsa_dim) float32 with values in {-1, +1}
    """
    rng = np.random.default_rng(seed)
    d_model = embeddings.shape[1]

    # Random projection matrix — each column is a random hyperplane normal
    # Process in chunks to avoid OOM on large projections
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

    # Fix zeros (sign(0) = 0) — snap to +1
    codebook[codebook == 0] = 1.0

    return codebook


def validate_simhash(codebook: np.ndarray, embeddings: np.ndarray):
    """Sanity check: verify SimHash preserves semantic neighborhoods."""
    # Use token IDs for common English words (Qwen2.5 BPE)
    # These are approximate — we check relative ordering, not exact values
    test_groups = [
        # Semantically related pairs (should have higher similarity)
        ("related", [
            (279, 323),     # common function words
            (498, 499),     # consecutive IDs (often morphological variants)
            (1075, 1076),   # consecutive IDs
        ]),
        # Random pairs (should be ~0.50)
        ("random", [
            (100, 50000),
            (200, 100000),
            (300, 150000),
        ]),
    ]

    print("\n  SimHash Validation:")
    print("  " + "-" * 50)

    for group_name, pairs in test_groups:
        sims = []
        for id_a, id_b in pairs:
            if id_a >= len(codebook) or id_b >= len(codebook):
                continue
            # Hamming similarity in bipolar: (D + dot(a,b)) / (2*D)
            dot = np.sum(codebook[id_a] * codebook[id_b])
            ham_sim = (DIM + dot) / (2 * DIM)
            sims.append(ham_sim)

            # Also compute original cosine similarity for comparison
            if embeddings is not None:
                a_f, b_f = embeddings[id_a], embeddings[id_b]
                cos_sim = np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-8)

        if sims:
            print(f"    {group_name:>10s}: mean_hamming={np.mean(sims):.4f}  "
                  f"(expected: {'> 0.52' if group_name == 'related' else '~ 0.50'})")

    # Global stats
    rng = np.random.default_rng(99)
    sample_ids = rng.choice(len(codebook), size=1000, replace=False)
    sample = codebook[sample_ids]
    # Pairwise similarities for a random sample
    dots = sample @ sample.T
    ham_sims = (DIM + dots) / (2 * DIM)
    np.fill_diagonal(ham_sims, np.nan)
    print(f"\n    Global pairwise (1000 random tokens):")
    print(f"      mean={np.nanmean(ham_sims):.4f}  std={np.nanstd(ham_sims):.4f}")
    print(f"      min={np.nanmin(ham_sims):.4f}   max={np.nanmax(ham_sims):.4f}")
    print()


# ── VSA Context Accumulator (numba-accelerated) ─────────────────────

@nb.njit(cache=True, parallel=True)
def _build_all_pairs(codebook, flat_ids, pair_info, window, dim):
    """Build all (context_state, transform) pairs in parallel with numba.

    Args:
        codebook: (vocab, dim) bipolar float32
        flat_ids: flat int32 array of all token IDs across all trajectories
        pair_info: (N, 3) int32 — [traj_flat_offset, pos_in_traj, next_token_id]
        window: context window size
        dim: VSA dimension

    Returns:
        states: (N, dim) float32 bipolar
        transforms: (N, dim) float32 bipolar
    """
    n = pair_info.shape[0]
    states = np.empty((n, dim), dtype=np.float32)
    transforms = np.empty((n, dim), dtype=np.float32)

    for i in nb.prange(n):
        traj_off = pair_info[i, 0]
        pos = pair_info[i, 1]
        next_tok = pair_info[i, 2]

        start = pos - window + 1
        if start < 0:
            start = 0

        # Permute + bundle inline
        for d in range(dim):
            accum = np.float32(0.0)
            for idx in range(start, pos + 1):
                shift = pos - idx
                src = (d - shift) % dim
                accum += codebook[flat_ids[traj_off + idx], src]
            if accum > 0.0:
                states[i, d] = 1.0
            elif accum < 0.0:
                states[i, d] = -1.0
            else:
                states[i, d] = 1.0

        # Transform = bind(context, next_token)
        for d in range(dim):
            transforms[i, d] = states[i, d] * codebook[next_tok, d]

    return states, transforms


# ── Data Loading ──────────────────────────────────────────────────────

def load_trajectories(path: Path, max_lines: int = 50000) -> list[list[int]]:
    """Load token trajectories from JSONL file."""
    trajectories = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                data = json.loads(line)
                tokens = data["tokens"]
                if len(tokens) >= 3:  # need at least context + target
                    trajectories.append(tokens)
            except (json.JSONDecodeError, KeyError):
                continue
    return trajectories


def make_training_pairs(trajectories: list[list[int]], codebook: np.ndarray,
                        window: int, max_pairs: int = 200000,
                        seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Build (context_state, target_transform) pairs for training.

    Uses numba-parallelized _build_all_pairs for speed.

    Returns:
        states: (N, DIM) float32 bipolar context states
        transforms: (N, DIM) float32 bipolar target transforms
    """
    rng = np.random.default_rng(seed)
    vocab_size = codebook.shape[0]
    dim = codebook.shape[1]

    # Flatten all trajectories into one contiguous array + build pair metadata
    flat_ids_list = []
    pair_info_list = []  # (traj_flat_offset, pos_in_traj, next_token_id)
    offset = 0

    for traj in trajectories:
        traj_len = len(traj)
        flat_ids_list.extend(traj)
        for pos in range(window - 1, traj_len - 1):
            next_tok = traj[pos + 1]
            if next_tok < vocab_size:
                pair_info_list.append((offset, pos, next_tok))
        offset += traj_len

    flat_ids = np.array(flat_ids_list, dtype=np.int32)
    pair_info = np.array(pair_info_list, dtype=np.int32)

    # Sample if too many
    if len(pair_info) > max_pairs:
        indices = rng.choice(len(pair_info), size=max_pairs, replace=False)
        pair_info = pair_info[indices]

    print(f"  Building {len(pair_info):,d} training pairs (window={window}) ...")
    print(f"  JIT compiling numba kernel (first run only) ...")

    t0 = time.perf_counter()
    states, transforms = _build_all_pairs(codebook, flat_ids, pair_info, window, dim)
    elapsed = time.perf_counter() - t0

    print(f"  Done: {len(pair_info):,d} pairs in {elapsed:.1f}s")
    return states, transforms


# ── Model (grilly autograd) ──────────────────────────────────────────

class BinaryMLP:
    """2-layer MLP with sign activation (Straight-Through Estimator for backward).

    Forward: x -> W1*x -> sign(h) -> W2*sign(h)
    Backward: STE passes gradients through sign as identity.
    """

    def __init__(self, dim: int, hidden: int, seed: int = 42):
        from grilly.nn.autograd import Variable

        rng = np.random.default_rng(seed)
        # Xavier-like init scaled for bipolar domain
        scale1 = np.sqrt(2.0 / (dim + hidden))
        scale2 = np.sqrt(2.0 / (hidden + dim))

        self.w1 = Variable(
            rng.standard_normal((hidden, dim)).astype(np.float32) * scale1,
            requires_grad=True,
        )
        self.w2 = Variable(
            rng.standard_normal((dim, hidden)).astype(np.float32) * scale2,
            requires_grad=True,
        )

    def forward(self, x):
        """Forward pass with STE sign activation."""
        from grilly.nn.autograd import Variable

        # Layer 1: linear
        h = x @ self.w1.data.T  # (batch, hidden) — raw numpy matmul for speed
        h = Variable(h, requires_grad=True)

        # Sign activation with STE: forward uses sign, backward passes through
        h_sign = Variable(np.sign(h.data).astype(np.float32), requires_grad=True)
        # Link gradient: h_sign.grad will be copied to h.grad (STE)
        h_sign._ste_source = h

        # Layer 2: linear
        out_data = h_sign.data @ self.w2.data.T  # (batch, dim)
        out = Variable(out_data, requires_grad=True)

        # Store intermediates for manual backward
        self._cache = (x, h, h_sign, out)
        return out

    def backward(self, grad_output: np.ndarray):
        """Manual backward pass (grilly autograd handles the rest)."""
        x_data, h, h_sign, out = self._cache

        # Grad for W2: grad_output.T @ h_sign
        grad_w2 = grad_output.T @ h_sign.data  # (dim, hidden)

        # Grad through layer 2 -> h_sign
        grad_h_sign = grad_output @ self.w2.data  # (batch, hidden)

        # STE: pass gradient through sign unchanged
        grad_h = grad_h_sign

        # Grad for W1: grad_h.T @ x
        grad_w1 = grad_h.T @ x_data  # (hidden, dim)

        # Accumulate
        if self.w1.grad is None:
            self.w1.grad = grad_w1
        else:
            self.w1.grad += grad_w1
        if self.w2.grad is None:
            self.w2.grad = grad_w2
        else:
            self.w2.grad += grad_w2

    def parameters(self):
        return [self.w1, self.w2]

    def zero_grad(self):
        self.w1.grad = None
        self.w2.grad = None


def clip_grad_norm(params, max_norm: float):
    """Clip gradient norm across all parameters."""
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += np.sum(p.grad ** 2)
    total_norm = np.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return total_norm


# ── Simple AdamW (numpy, no GPU needed for POC) ─────────────────────

class SimpleAdamW:
    """Minimal AdamW for the POC. No GPU overhead."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = wd
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad / BATCH_SIZE  # mean gradient

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p.data)


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate(model: BinaryMLP, codebook: np.ndarray,
             states: np.ndarray, transforms: np.ndarray,
             n_samples: int = EVAL_SAMPLES) -> dict:
    """Evaluate prediction quality via Hamming similarity.

    For each sample:
      1. Forward context state through model -> predicted transform
      2. predicted_next = context * sign(predicted_transform)  (unbind)
      3. true_next = context * true_transform  (unbind)
      4. Hamming similarity between predicted_next and true_next
    """
    rng = np.random.default_rng(77)
    idx = rng.choice(len(states), size=min(n_samples, len(states)), replace=False)

    batch_states = states[idx]
    batch_transforms = transforms[idx]

    # Forward
    out = model.forward(batch_states)
    pred_transforms = np.sign(out.data)
    pred_transforms[pred_transforms == 0] = 1.0

    # Unbind to get predicted next-token codes
    pred_next = batch_states * pred_transforms
    true_next = batch_states * batch_transforms

    # Hamming similarity: (D + dot) / (2D) for bipolar
    dots = np.sum(pred_next * true_next, axis=1)
    ham_sims = (DIM + dots) / (2 * DIM)

    # Also compute cosine similarity of raw outputs vs targets
    raw_dots = np.sum(out.data * batch_transforms, axis=1)
    raw_norms = np.linalg.norm(out.data, axis=1) * np.linalg.norm(batch_transforms, axis=1)
    cos_sims = raw_dots / (raw_norms + 1e-8)

    return {
        "hamming_sim": float(np.mean(ham_sims)),
        "hamming_std": float(np.std(ham_sims)),
        "cosine_sim": float(np.mean(cos_sims)),
        "n_samples": len(idx),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    global DIM, HIDDEN

    parser = argparse.ArgumentParser(description="SimHash VSA Next-Token POC")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps (0 = sanity check only)")
    parser.add_argument("--context-window", type=int, default=8, help="Context window size")
    parser.add_argument("--max-trajectories", type=int, default=10000, help="Max trajectories to load")
    parser.add_argument("--max-pairs", type=int, default=100000, help="Max training pairs")
    parser.add_argument("--dim", type=int, default=DIM, help="VSA dimension")
    parser.add_argument("--hidden", type=int, default=HIDDEN, help="MLP hidden dim")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    DIM = args.dim
    HIDDEN = args.hidden

    print("=" * 70)
    print("  SimHash VSA Next-Token Prediction POC")
    print("=" * 70)
    print()

    # ── 1. Load Qwen embeddings ──────────────────────────────────────
    print("[1/5] Loading Qwen2.5 embeddings ...")
    if not EMBEDDINGS_PATH.exists():
        print(f"  ERROR: {EMBEDDINGS_PATH} not found.")
        print(f"  Run: python scripts/extract_qwen_embeddings.py")
        sys.exit(1)

    embeddings = np.load(str(EMBEDDINGS_PATH))
    print(f"  Shape: {embeddings.shape}  ({embeddings.nbytes / 1e6:.0f} MB)")

    # ── 2. SimHash to bipolar codebook ───────────────────────────────
    print(f"\n[2/5] SimHash projection ({embeddings.shape[1]} -> {DIM}) ...")
    t0 = time.perf_counter()
    codebook = simhash_codebook(embeddings, DIM, seed=args.seed)
    print(f"  Codebook: {codebook.shape}  ({codebook.nbytes / 1e6:.0f} MB)")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")

    # Free embeddings — no longer needed
    del embeddings

    validate_simhash(codebook, None)

    if args.steps == 0:
        print("Steps=0, sanity check complete.")
        return

    # ── 3. Build training data ───────────────────────────────────────
    print(f"[3/5] Loading trajectories from {TRAJECTORIES_PATH.name} ...")
    trajectories = load_trajectories(TRAJECTORIES_PATH, max_lines=args.max_trajectories)
    print(f"  Loaded {len(trajectories):,d} trajectories")
    avg_len = np.mean([len(t) for t in trajectories])
    print(f"  Average length: {avg_len:.1f} tokens")

    # Filter tokens to valid vocab range
    vocab_size = codebook.shape[0]
    n_oov = sum(1 for t in trajectories for tok in t if tok >= vocab_size)
    if n_oov > 0:
        print(f"  Warning: {n_oov:,d} OOV tokens (id >= {vocab_size}) will be skipped")

    print()
    states, transforms = make_training_pairs(
        trajectories, codebook, window=args.context_window,
        max_pairs=args.max_pairs, seed=args.seed,
    )
    # Free trajectories
    del trajectories

    # Train/eval split (90/10)
    n = len(states)
    split = int(n * 0.9)
    train_states, eval_states = states[:split], states[split:]
    train_transforms, eval_transforms = transforms[:split], transforms[split:]
    print(f"\n  Train: {len(train_states):,d}  Eval: {len(eval_states):,d}")

    # ── 4. Train ─────────────────────────────────────────────────────
    print(f"\n[4/5] Training ({args.steps} steps, batch={BATCH_SIZE}, "
          f"lr={args.lr}, dim={DIM}, hidden={HIDDEN}, window={args.context_window}) ...")
    print()

    model = BinaryMLP(DIM, HIDDEN, seed=args.seed)
    optimizer = SimpleAdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 1)

    best_sim = 0.0
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        # Sample batch
        idx = rng.choice(len(train_states), size=BATCH_SIZE, replace=False)
        batch_x = train_states[idx]
        batch_y = train_transforms[idx]

        # Forward
        model.zero_grad()
        out = model.forward(batch_x)

        # MSE loss (works well for bipolar targets in [-1, +1])
        diff = out.data - batch_y
        loss_val = float(np.mean(diff ** 2))

        # Backward (manual, since we're doing STE)
        grad_output = 2.0 * diff / diff.size  # d(MSE)/d(out)
        # Scale up for batch accumulation
        grad_output *= BATCH_SIZE
        model.backward(grad_output)

        # Clip and step
        grad_norm = clip_grad_norm(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Eval
        if step % EVAL_EVERY == 0 or step == 1:
            metrics = evaluate(model, codebook, eval_states, eval_transforms)
            elapsed = time.perf_counter() - t_start
            best_sim = max(best_sim, metrics["hamming_sim"])

            print(f"  step {step:>5d} | loss={loss_val:.5f} | "
                  f"ham_sim={metrics['hamming_sim']:.4f} | "
                  f"cos_sim={metrics['cosine_sim']:.4f} | "
                  f"gnorm={grad_norm:.2f} | "
                  f"{elapsed:.0f}s")

    # ── 5. Final evaluation ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print()

    final = evaluate(model, codebook, eval_states, eval_transforms, n_samples=len(eval_states))
    print(f"  Context window:     {args.context_window}")
    print(f"  Training steps:     {args.steps}")
    print(f"  Training pairs:     {len(train_states):,d}")
    print(f"  Eval pairs:         {len(eval_states):,d}")
    print()
    print(f"  Hamming similarity: {final['hamming_sim']:.4f} +/- {final['hamming_std']:.4f}")
    print(f"  Cosine similarity:  {final['cosine_sim']:.4f}")
    print(f"  Best during train:  {best_sim:.4f}")
    print(f"  Random baseline:    0.5000")
    print()

    if final["hamming_sim"] > 0.55:
        print("  ** PASS ** Similarity significantly above random.")
        print("  Approach validated — proceed to Vulkan port.")
    elif final["hamming_sim"] > 0.52:
        print("  ** MARGINAL ** Slight signal detected.")
        print("  Consider: larger context window, more training data, or Approach B.")
    else:
        print("  ** FAIL ** No meaningful signal above random.")
        print("  Investigate: SimHash quality, context encoding, or try Approach B/C.")

    # Save weights for Vulkan export
    weights_path = ROOT / "model" / "student" / "poc_weights.npz"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    w1_bipolar = np.sign(model.w1.data).astype(np.float32)
    w2_bipolar = np.sign(model.w2.data).astype(np.float32)
    w1_bipolar[w1_bipolar == 0] = 1.0
    w2_bipolar[w2_bipolar == 0] = 1.0
    np.savez(str(weights_path), w1=w1_bipolar, w2=w2_bipolar)
    print(f"\n  Weights saved to {weights_path}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
