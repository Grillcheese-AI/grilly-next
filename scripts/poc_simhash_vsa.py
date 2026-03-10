"""
SimHash VSA Next-Token Prediction POC
──────────────────────────────────────
Proves that SimHash-projected Qwen2.5 embeddings + VSA context
accumulation make next-token prediction learnable.

Pipeline:
  1. Load Qwen2.5 float embeddings -> SimHash to bipolar codebook
  2. Build context VSA states via permute-bind chain (streaming, unlimited context)
  3. Train 4-layer self-binarizing MLP (tanh annealing, grilly autograd) to predict binding transform
  4. Eval: Hamming similarity of predicted vs true next-token code

Usage:
    python scripts/poc_simhash_vsa.py --steps 2000
    python scripts/poc_simhash_vsa.py --steps 0          # embedding sanity check only
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
GRAD_CLIP = 5.0      # Gradient clipping norm (tanh gradients well-behaved, loose safety net)
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

@nb.njit(cache=True)
def _build_permute_bind_pairs(codebook, flat_ids, traj_starts, traj_lengths,
                               max_pairs, dim):
    """Build (context_state, transform) pairs using permute-bind chain.

    For each trajectory, walks tokens sequentially:
        S_0 = codebook[tok_0]
        S_t = roll(S_{t-1}, 1) * codebook[tok_t]   (element-wise multiply in bipolar)

    At each position t (starting from t=1), emits:
        state = S_t
        transform = S_t * codebook[next_token]   (binding transform)
    """
    n_traj = len(traj_starts)

    # First pass: count total pairs
    total = 0
    for t in range(n_traj):
        length = traj_lengths[t]
        if length >= 2:
            pairs_from_traj = length - 1
            total += pairs_from_traj
    if total > max_pairs:
        total = max_pairs

    states = np.empty((total, dim), dtype=np.float32)
    transforms = np.empty((total, dim), dtype=np.float32)

    pair_idx = 0
    for t in range(n_traj):
        if pair_idx >= total:
            break
        start = traj_starts[t]
        length = traj_lengths[t]
        if length < 2:
            continue

        # Initialize state with first token
        state = np.empty(dim, dtype=np.float32)
        for d in range(dim):
            state[d] = codebook[flat_ids[start], d]

        for pos in range(1, length):
            if pair_idx >= total:
                break

            # Permute: circular shift by 1
            last = state[dim - 1]
            for d in range(dim - 1, 0, -1):
                state[d] = state[d - 1]
            state[0] = last

            # Bind with current token (element-wise multiply in bipolar)
            tok_id = flat_ids[start + pos]
            for d in range(dim):
                state[d] = state[d] * codebook[tok_id, d]

            # Emit pair: (state, transform to next token)
            if pos < length - 1:
                next_tok = flat_ids[start + pos + 1]
                for d in range(dim):
                    states[pair_idx, d] = state[d]
                    transforms[pair_idx, d] = state[d] * codebook[next_tok, d]
                pair_idx += 1

    # Trim to actual count
    return states[:pair_idx], transforms[:pair_idx]


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
                        max_pairs: int = 200000,
                        seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Build (context_state, target_transform) pairs for training.

    Uses permute-bind chain: streaming, unlimited context, O(1) per token.

    Returns:
        states: (N, DIM) float32 bipolar context states
        transforms: (N, DIM) float32 bipolar target transforms
    """
    vocab_size = codebook.shape[0]
    dim = codebook.shape[1]

    # Filter OOV tokens from trajectories
    filtered = []
    for traj in trajectories:
        clean = [t for t in traj if t < vocab_size]
        if len(clean) >= 3:
            filtered.append(clean)

    # Flatten trajectories + compute starts/lengths
    flat_ids = np.concatenate([np.array(t, dtype=np.int32) for t in filtered])
    traj_lengths = np.array([len(t) for t in filtered], dtype=np.int32)
    traj_starts = np.zeros(len(filtered), dtype=np.int32)
    traj_starts[1:] = np.cumsum(traj_lengths[:-1])

    print(f"  Building {max_pairs:,d} training pairs (permute-bind chain) ...")
    print(f"  JIT compiling numba kernel (first run only) ...")

    t0 = time.perf_counter()
    states, transforms = _build_permute_bind_pairs(
        codebook, flat_ids, traj_starts, traj_lengths, max_pairs, dim)
    elapsed = time.perf_counter() - t0

    print(f"  Done: {len(states):,d} pairs in {elapsed:.1f}s")
    return states, transforms


# ── Model (grilly autograd) ──────────────────────────────────────────

class SelfBinarizingMLP:
    """4-layer residual MLP with tanh(v*x) self-binarizing annealing.

    Replaces STE sign() with smooth tanh(v*x) where v anneals from 1 to v_max.
    At v=1: smooth gradients, clean training signal.
    At v=10: tanh ≈ sign, weights are effectively binary.
    Export: sign(W_float) produces identical binary weights for GPU shader.

    Architecture (bipolar float domain):
        W_bin = tanh(v * W)                              (smooth weight binarization)
        h1 = tanh(v * (W1_bin @ x))                      (input -> hidden)
        h2 = tanh(v * (W2_bin @ h1)) * h1                (hidden -> hidden, residual)
        h3 = tanh(v * (W3_bin @ h2)) * h2                (hidden -> hidden, residual)
        out = tanh(v * (W4_bin @ h3))                     (hidden -> output)
    """

    def __init__(self, dim: int, hidden: int, seed: int = 42):
        from grilly.nn.autograd import Variable

        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / (dim + hidden))
        s2 = np.sqrt(2.0 / (hidden + hidden))
        s4 = np.sqrt(2.0 / (hidden + dim))

        self.w1 = Variable(rng.standard_normal((hidden, dim)).astype(np.float32) * s1,
                           requires_grad=True)
        self.w2 = Variable(rng.standard_normal((hidden, hidden)).astype(np.float32) * s2,
                           requires_grad=True)
        self.w3 = Variable(rng.standard_normal((hidden, hidden)).astype(np.float32) * s2,
                           requires_grad=True)
        self.w4 = Variable(rng.standard_normal((dim, hidden)).astype(np.float32) * s4,
                           requires_grad=True)
        self.sharpness = 1.0  # v parameter — annealed externally

    def forward(self, x_np: np.ndarray):
        from grilly.nn.autograd import Variable, tanh, matmul, transpose

        v = self.sharpness
        x = Variable(x_np, requires_grad=False)

        # Smooth weight binarization: tanh(v * W) ≈ sign(W) as v → ∞
        w1_bin = tanh(self.w1 * v)
        w2_bin = tanh(self.w2 * v)
        w3_bin = tanh(self.w3 * v)
        w4_bin = tanh(self.w4 * v)

        # Layer 1: input -> hidden
        h1 = tanh(matmul(x, transpose(w1_bin)) * v)

        # Layer 2: hidden -> hidden + residual (multiply = XOR in bipolar)
        h2 = tanh(matmul(h1, transpose(w2_bin)) * v) * h1

        # Layer 3: hidden -> hidden + residual
        h3 = tanh(matmul(h2, transpose(w3_bin)) * v) * h2

        # Layer 4: hidden -> output
        out = tanh(matmul(h3, transpose(w4_bin)) * v)

        return out

    def parameters(self):
        return [self.w1, self.w2, self.w3, self.w4]

    def zero_grad(self):
        for w in self.parameters():
            w.grad = None



# ── Evaluation ────────────────────────────────────────────────────────

def evaluate(model: SelfBinarizingMLP, codebook: np.ndarray,
             states: np.ndarray, transforms: np.ndarray,
             n_samples: int = EVAL_SAMPLES) -> dict:
    """Evaluate prediction quality via Hamming similarity."""
    from grilly.nn.autograd import no_grad

    rng = np.random.default_rng(77)
    idx = rng.choice(len(states), size=min(n_samples, len(states)), replace=False)

    batch_states = states[idx]
    batch_transforms = transforms[idx]

    # Forward (no gradient tracking needed for eval)
    with no_grad():
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

    print()
    states, transforms = make_training_pairs(
        trajectories, codebook,
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
          f"lr={args.lr}, dim={DIM}, hidden={HIDDEN}, self-binarizing tanh) ...")
    print()

    model = SelfBinarizingMLP(DIM, HIDDEN, seed=args.seed)

    from grilly.optim import AutoHypergradientAdamW
    optimizer = AutoHypergradientAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        hyper_lr=0.01,
        lr_min=1e-5,
        lr_max=0.05,
        warmup_steps=50,
        track_surprise=True,
        use_gpu=False,  # CPU training (numpy autograd)
    )

    # Sharpness annealing: v = 1 → v_max linearly over training
    v_min = 1.0
    v_max = 10.0

    rng = np.random.default_rng(args.seed + 1)
    best_sim = 0.0
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        # Anneal sharpness
        model.sharpness = v_min + (v_max - v_min) * (step - 1) / max(args.steps - 1, 1)

        # Sample batch
        idx = rng.choice(len(train_states), size=BATCH_SIZE, replace=False)
        batch_x = train_states[idx]
        batch_y = train_transforms[idx]

        # Forward (autograd builds computation graph)
        model.zero_grad()
        out = model.forward(batch_x)

        # Logistic loss: mean(log(1 + exp(-output * target)))
        margin = out.data * batch_y
        loss_val = float(np.mean(np.log1p(np.exp(-margin))))

        # Backward: d/d(out) = -target * sigmoid(-margin) / N
        sigmoid_neg = 1.0 / (1.0 + np.exp(margin))
        grad_output = -batch_y * sigmoid_neg / margin.size
        out.backward(grad_output)

        # Gradient norm (for monitoring)
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += np.sum(p.grad ** 2)
        grad_norm = np.sqrt(grad_norm)

        # Clip gradients
        if grad_norm > GRAD_CLIP:
            scale = GRAD_CLIP / (grad_norm + 1e-8)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad *= scale

        # Optimizer step
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
                  f"lr={optimizer.current_lr:.6f} | "
                  f"v={model.sharpness:.1f} | "
                  f"{elapsed:.0f}s")

    # ── 5. Final evaluation ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print()

    final = evaluate(model, codebook, eval_states, eval_transforms, n_samples=len(eval_states))
    print(f"  Context encoder:    permute-bind chain (unlimited)")
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

    # Save weights for Vulkan export (all 4 layers, binarized)
    weights_path = ROOT / "model" / "student" / "poc_weights.npz"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, param in [("w1", model.w1), ("w2", model.w2),
                         ("w3", model.w3), ("w4", model.w4)]:
        w = np.sign(param.data).astype(np.float32)
        w[w == 0] = 1.0
        saved[name] = w
    np.savez(str(weights_path), **saved)
    print(f"\n  Weights saved to {weights_path} (4 layers, sign-binarized from tanh)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
