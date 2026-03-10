"""
VSA Student Training
────────────────────
Trains a 4-layer self-binarizing residual MLP to predict next-token
binding transforms from VSA context states.

Pipeline:
  1. Load Qwen2.5 float embeddings -> SimHash to bipolar codebook
  2. Build context VSA states via permute-bind chain (streaming, unlimited context)
  3. Train 4-layer MLP (tanh(1*W) smooth weights, grilly autograd) with logistic loss
  4. Export: sign(W_float) -> bipolar weights for Vulkan shader

Architecture:
  W_bin = tanh(W)              (smooth weight approximation during training)
  h1 = tanh(W1_bin @ x)       (input -> hidden)
  h2 = tanh(W2_bin @ h1) * h1 (hidden -> hidden, residual)
  h3 = tanh(W3_bin @ h2) * h2 (hidden -> hidden, residual)
  out = tanh(W4_bin @ h3)     (hidden -> output)

Export: sign(W) produces identical binary weights for XNOR+POPCNT on GPU.

Usage:
    python scripts/train_student.py
    python scripts/train_student.py --dim 2048 --hidden 1024 --steps 3000
    python scripts/train_student.py --steps 0  # embedding sanity check only
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numba as nb
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
EMBEDDINGS_PATH = ROOT / "model" / "codebooks" / "qwen_embeddings.npy"
TRAJECTORIES_PATH = ROOT / "model" / "teacher" / "teacher_golden_trajectories.jsonl"
WEIGHTS_PATH = ROOT / "model" / "student" / "student_weights.npz"

# ── Defaults ──────────────────────────────────────────────────────────

DIM = 2048
HIDDEN = 1024
BATCH_SIZE = 512
LR = 1e-3
GRAD_CLIP = 5.0
EVAL_EVERY = 100
EVAL_SAMPLES = 512


# ── SimHash ───────────────────────────────────────────────────────────

def simhash_codebook(embeddings: np.ndarray, vsa_dim: int, seed: int = 42) -> np.ndarray:
    """Project float embeddings to bipolar {-1,+1} via random hyperplane hashing."""
    rng = np.random.default_rng(seed)
    d_model = embeddings.shape[1]
    vocab_size = embeddings.shape[0]
    codebook = np.empty((vocab_size, vsa_dim), dtype=np.float32)

    chunk_size = 1024
    for start in range(0, vsa_dim, chunk_size):
        end = min(start + chunk_size, vsa_dim)
        R = rng.standard_normal((d_model, end - start)).astype(np.float32)
        R /= np.linalg.norm(R, axis=0, keepdims=True)
        codebook[:, start:end] = np.sign(embeddings @ R)

    codebook[codebook == 0] = 1.0
    return codebook


def validate_simhash(codebook: np.ndarray, dim: int):
    """Sanity check: verify SimHash preserves semantic neighborhoods."""
    test_groups = [
        ("related", [(279, 323), (498, 499), (1075, 1076)]),
        ("random", [(100, 50000), (200, 100000), (300, 150000)]),
    ]

    print("\n  SimHash Validation:")
    print("  " + "-" * 50)

    for group_name, pairs in test_groups:
        sims = []
        for id_a, id_b in pairs:
            if id_a >= len(codebook) or id_b >= len(codebook):
                continue
            dot = np.sum(codebook[id_a] * codebook[id_b])
            sims.append((dim + dot) / (2 * dim))

        if sims:
            print(f"    {group_name:>10s}: mean_hamming={np.mean(sims):.4f}  "
                  f"(expected: {'> 0.52' if group_name == 'related' else '~ 0.50'})")

    rng = np.random.default_rng(99)
    sample = codebook[rng.choice(len(codebook), size=1000, replace=False)]
    ham_sims = (dim + sample @ sample.T) / (2 * dim)
    np.fill_diagonal(ham_sims, np.nan)
    print(f"\n    Global pairwise (1000 random tokens):")
    print(f"      mean={np.nanmean(ham_sims):.4f}  std={np.nanstd(ham_sims):.4f}")
    print(f"      min={np.nanmin(ham_sims):.4f}   max={np.nanmax(ham_sims):.4f}")
    print()


# ── VSA Context Accumulator (numba-accelerated) ─────────────────────

@nb.njit(cache=True)
def _build_permute_bind_pairs(codebook, flat_ids, traj_starts, traj_lengths,
                               max_pairs, dim):
    """Build (context_state, transform) pairs using permute-bind chain."""
    n_traj = len(traj_starts)

    total = 0
    for t in range(n_traj):
        length = traj_lengths[t]
        if length >= 2:
            total += length - 1
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

        state = np.empty(dim, dtype=np.float32)
        for d in range(dim):
            state[d] = codebook[flat_ids[start], d]

        for pos in range(1, length):
            if pair_idx >= total:
                break

            last = state[dim - 1]
            for d in range(dim - 1, 0, -1):
                state[d] = state[d - 1]
            state[0] = last

            tok_id = flat_ids[start + pos]
            for d in range(dim):
                state[d] = state[d] * codebook[tok_id, d]

            if pos < length - 1:
                next_tok = flat_ids[start + pos + 1]
                for d in range(dim):
                    states[pair_idx, d] = state[d]
                    transforms[pair_idx, d] = state[d] * codebook[next_tok, d]
                pair_idx += 1

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
                if len(tokens) >= 3:
                    trajectories.append(tokens)
            except (json.JSONDecodeError, KeyError):
                continue
    return trajectories


def make_training_pairs(trajectories: list[list[int]], codebook: np.ndarray,
                        max_pairs: int = 200000,
                        seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Build (context_state, target_transform) pairs via permute-bind chain."""
    vocab_size = codebook.shape[0]
    dim = codebook.shape[1]

    filtered = []
    for traj in trajectories:
        clean = [t for t in traj if t < vocab_size]
        if len(clean) >= 3:
            filtered.append(clean)

    flat_ids = np.concatenate([np.array(t, dtype=np.int32) for t in filtered])
    traj_lengths = np.array([len(t) for t in filtered], dtype=np.int32)
    traj_starts = np.zeros(len(filtered), dtype=np.int32)
    traj_starts[1:] = np.cumsum(traj_lengths[:-1])

    print(f"  Building {max_pairs:,d} training pairs (permute-bind chain) ...")
    print(f"  JIT compiling numba kernel (first run only) ...")

    t0 = time.perf_counter()
    states, transforms = _build_permute_bind_pairs(
        codebook, flat_ids, traj_starts, traj_lengths, max_pairs, dim)
    print(f"  Done: {len(states):,d} pairs in {time.perf_counter() - t0:.1f}s")
    return states, transforms


# ── Model ─────────────────────────────────────────────────────────────

class SelfBinarizingMLP:
    """4-layer residual MLP with smooth tanh weight approximation.

    Weights trained as tanh(W) -- smooth bipolar approximation.
    Export: sign(W) produces exact binary weights for XNOR+POPCNT.
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

    def forward(self, x_np: np.ndarray):
        from grilly.nn.autograd import Variable, tanh, matmul, transpose

        x = Variable(x_np, requires_grad=False)

        w1_bin = tanh(self.w1)
        w2_bin = tanh(self.w2)
        w3_bin = tanh(self.w3)
        w4_bin = tanh(self.w4)

        h1 = tanh(matmul(x, transpose(w1_bin)))
        h2 = tanh(matmul(h1, transpose(w2_bin))) * h1
        h3 = tanh(matmul(h2, transpose(w3_bin))) * h2
        out = tanh(matmul(h3, transpose(w4_bin)))

        return out

    def parameters(self):
        return [self.w1, self.w2, self.w3, self.w4]

    def zero_grad(self):
        for w in self.parameters():
            w.grad = None


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate(model, codebook, states, transforms, dim,
             n_samples=EVAL_SAMPLES):
    """Evaluate prediction quality via Hamming similarity."""
    from grilly.nn.autograd import no_grad

    rng = np.random.default_rng(77)
    idx = rng.choice(len(states), size=min(n_samples, len(states)), replace=False)

    batch_states = states[idx]
    batch_transforms = transforms[idx]

    with no_grad():
        out = model.forward(batch_states)
    pred_transforms = np.sign(out.data)
    pred_transforms[pred_transforms == 0] = 1.0

    pred_next = batch_states * pred_transforms
    true_next = batch_states * batch_transforms

    dots = np.sum(pred_next * true_next, axis=1)
    ham_sims = (dim + dots) / (2 * dim)

    raw_dots = np.sum(out.data * batch_transforms, axis=1)
    raw_norms = np.linalg.norm(out.data, axis=1) * np.linalg.norm(batch_transforms, axis=1)
    cos_sims = raw_dots / (raw_norms + 1e-8)

    return {
        "hamming_sim": float(np.mean(ham_sims)),
        "hamming_std": float(np.std(ham_sims)),
        "cosine_sim": float(np.mean(cos_sims)),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VSA Student Training")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--max-trajectories", type=int, default=10000)
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(WEIGHTS_PATH))
    args = parser.parse_args()

    dim = args.dim
    hidden = args.hidden

    print("=" * 70)
    print("  VSA Student Training")
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
    print(f"\n[2/5] SimHash projection ({embeddings.shape[1]} -> {dim}) ...")
    t0 = time.perf_counter()
    codebook = simhash_codebook(embeddings, dim, seed=args.seed)
    print(f"  Codebook: {codebook.shape}  ({codebook.nbytes / 1e6:.0f} MB)")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")
    del embeddings

    validate_simhash(codebook, dim)

    if args.steps == 0:
        print("Steps=0, sanity check complete.")
        return

    # ── 3. Build training data ───────────────────────────────────────
    print(f"[3/5] Loading trajectories from {TRAJECTORIES_PATH.name} ...")
    trajectories = load_trajectories(TRAJECTORIES_PATH, max_lines=args.max_trajectories)
    print(f"  Loaded {len(trajectories):,d} trajectories")
    print(f"  Average length: {np.mean([len(t) for t in trajectories]):.1f} tokens")

    print()
    states, transforms = make_training_pairs(
        trajectories, codebook, max_pairs=args.max_pairs, seed=args.seed)
    del trajectories

    n = len(states)
    split = int(n * 0.9)
    train_states, eval_states = states[:split], states[split:]
    train_transforms, eval_transforms = transforms[:split], transforms[split:]
    print(f"\n  Train: {len(train_states):,d}  Eval: {len(eval_states):,d}")

    # ── 4. Train ─────────────────────────────────────────────────────
    from grilly.optim import AutoHypergradientAdamW

    print(f"\n[4/5] Training ({args.steps} steps, batch={BATCH_SIZE}, "
          f"lr={args.lr}, dim={dim}, hidden={hidden}) ...")
    print()

    model = SelfBinarizingMLP(dim, hidden, seed=args.seed)
    optimizer = AutoHypergradientAdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
        hyper_lr=0.01, lr_min=1e-4, lr_max=0.05,
        warmup_steps=50, track_surprise=True, use_gpu=False,
    )

    rng = np.random.default_rng(args.seed + 1)
    best_sim = 0.0
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        idx = rng.choice(len(train_states), size=BATCH_SIZE, replace=False)
        batch_x = train_states[idx]
        batch_y = train_transforms[idx]

        model.zero_grad()
        out = model.forward(batch_x)

        margin = out.data * batch_y
        loss_val = float(np.mean(np.log1p(np.exp(-margin))))
        sigmoid_neg = 1.0 / (1.0 + np.exp(margin))
        grad_output = -batch_y * sigmoid_neg / batch_x.shape[1]

        out.backward(grad_output)

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += np.sum(p.grad ** 2)
        grad_norm = np.sqrt(grad_norm)

        if grad_norm > GRAD_CLIP:
            scale = GRAD_CLIP / (grad_norm + 1e-8)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad *= scale

        optimizer.step()

        if step % EVAL_EVERY == 0 or step == 1:
            metrics = evaluate(model, codebook, eval_states, eval_transforms, dim)
            elapsed = time.perf_counter() - t_start
            best_sim = max(best_sim, metrics["hamming_sim"])

            print(f"  step {step:>5d} | loss={loss_val:.5f} | "
                  f"ham_sim={metrics['hamming_sim']:.4f} | "
                  f"cos_sim={metrics['cosine_sim']:.4f} | "
                  f"gnorm={grad_norm:.2f} | "
                  f"lr={optimizer.current_lr:.6f} | "
                  f"{elapsed:.0f}s")

    # ── 5. Final evaluation ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print()

    final = evaluate(model, codebook, eval_states, eval_transforms, dim,
                     n_samples=len(eval_states))
    print(f"  Architecture:       4-layer residual MLP (tanh, v=1 snap)")
    print(f"  Training steps:     {args.steps}")
    print(f"  Training pairs:     {len(train_states):,d}")
    print(f"  Eval pairs:         {len(eval_states):,d}")
    print()
    print(f"  Hamming similarity: {final['hamming_sim']:.4f} +/- {final['hamming_std']:.4f}")
    print(f"  Cosine similarity:  {final['cosine_sim']:.4f}")
    print(f"  Best during train:  {best_sim:.4f}")
    print(f"  Random baseline:    0.5000")
    print()

    # Save weights (bipolar float .npz for export_vsa_weights.py)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, param in [("w1", model.w1), ("w2", model.w2),
                         ("w3", model.w3), ("w4", model.w4)]:
        w = np.sign(param.data).astype(np.float32)
        w[w == 0] = 1.0
        saved[name] = w
    np.savez(str(output_path), **saved)
    print(f"  Weights saved to {output_path}")
    print(f"  Export: python scripts/export_vsa_weights.py --weights {output_path}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
