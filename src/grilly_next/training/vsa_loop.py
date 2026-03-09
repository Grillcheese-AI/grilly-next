"""VSA Training Loop with dual-path learning: Hebbian + Hippocampal Consolidation.

Hypergradient LR adaptation (OSGM-style) wraps the GPU Surprise-Momentum
optimizer, adjusting eta_base each step based on loss curvature signals.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import grilly_core


@dataclass
class TrainingResult:
    loss: float
    hebbian_weight_norm: float
    dream_report: Optional[object] = None  # DreamReport from C++
    world_model_fact_count: int = 0
    effective_lr: float = 0.0


class VSATrainingLoop:
    """Integrated VSA training with Hebbian fast path and hippocampal slow path.

    Hypergradient adaptation (OSGM from arXiv:2502.11229) adjusts the
    Surprise-Momentum optimizer's base learning rate each step using
    loss-gradient alignment signals with AdaGrad-style stabilization.
    """

    def __init__(
        self,
        device,
        d_model: int = 768,
        vsa_dim: int = 10240,
        K: int = 16,
        router_hidden: int = 32,
        dream_interval: int = 1000,
        dream_cycles: int = 128,
        hebbian_lr: float = 0.0001,
        hebbian_decay: float = 0.999,
        hebbian_clamp: float = 1.0,
        hebbian_interval: int = 50,
        enable_hebbian: bool = True,
        optim_lr: float = 0.01,
        optim_clip: float = 1.0,
        # Hypergradient parameters
        hyper_lr: float = 0.0,
        hyper_warmup: int = 10,
        hyper_meta_decay: float = 0.99,
        lr_min: float = 1e-5,
        lr_max: float = 0.01,
        gpu_yield_ms: float = 0.5,
        seed: int = 42,
    ):
        self.device = device
        self.vsa_dim = vsa_dim
        self.K = K
        self.dream_interval = dream_interval
        self.dream_cycles = dream_cycles
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.hebbian_clamp = hebbian_clamp
        self.hebbian_interval = hebbian_interval
        self.enable_hebbian = enable_hebbian
        self.step_count = 0
        self._gpu_yield_s = gpu_yield_ms / 1000.0  # Convert ms to seconds

        # C++ components
        self.model = grilly_core.VSAHypernetwork(
            device, d_model=d_model, vsa_dim=vsa_dim, K=K,
            router_hidden=router_hidden, seed=seed,
        )
        self.tape = grilly_core.TapeContext(device)
        self.consolidator = grilly_core.HippocampalConsolidator(max_capacity=10000)
        self.world_model = grilly_core.WorldModel(device)

        # Surprise-Momentum optimizer (applies gradients to hypernetwork weights)
        self.optimizer = grilly_core.SurpriseMomentumOptimizer(
            device, self.model, lr=optim_lr, clip=optim_clip
        )
        self.optimizer.weight_decay = 0.01  # Prevent weight explosion in full-rank W2

        # GPU Hebbian weight matrix (persistent, dim x dim)
        self.stdp_weights = None
        self._pre_accum = None
        self._post_accum = None
        self._accum_count = 0
        if enable_hebbian:
            self.stdp_weights = grilly_core.StdpWeights(device, dim=vsa_dim)
            self._pre_accum = np.zeros(vsa_dim, dtype=np.float32)
            self._post_accum = np.zeros(vsa_dim, dtype=np.float32)

        # ── Hypergradient state ─────────────────────────────────────────
        # OSGM-style: adapt eta_base using loss curvature with AdaGrad
        self._hyper_lr = hyper_lr
        self._hyper_warmup = hyper_warmup
        self._meta_decay = hyper_meta_decay
        self._lr_min = lr_min
        self._lr_max = lr_max
        self._G_lr = 1.0       # AdaGrad accumulator (seeded at 1.0, not 0.0)
        self._prev_loss = None  # loss at step k-1
        self._prev_dloss = 0.0  # smoothed loss derivative (EMA)
        self._loss_ema = None   # exponential moving average of loss
        self._loss_var = 1e-4   # variance estimate for normalization

    def _hypergradient_step(self, loss: float):
        """Adapt optimizer.eta_base using loss-based hypergradient signal.

        Uses OSGM-style (arXiv:2502.11229) adaptation:
          1. Compute normalized loss change: h = (loss - loss_ema) / sqrt(var)
          2. AdaGrad accumulate: G = decay*G + (1-decay)*h²
          3. Update: eta_base -= hyper_lr * h / sqrt(G)
          4. Clamp to [lr_min, lr_max] with 3% rate limit
        """
        # NaN protection: if loss is NaN/inf, reset LR to initial and skip
        if not np.isfinite(loss):
            self.optimizer.eta_base = float(np.clip(
                self._lr_min * 10, self._lr_min, self._lr_max))
            self._loss_ema = None
            self._loss_var = 1e-4
            self._G_lr = 1.0
            return

        if self.step_count <= self._hyper_warmup:
            # Warmup: just accumulate statistics
            if self._loss_ema is None:
                self._loss_ema = loss
            else:
                self._loss_ema = 0.99 * self._loss_ema + 0.01 * loss
            self._prev_loss = loss
            return

        # Update loss EMA and variance (alpha=0.01 → ~100-sample window,
        # smooths out single-sample noise in stochastic training)
        if self._loss_ema is None:
            self._loss_ema = loss
        alpha = 0.01
        self._loss_ema = (1.0 - alpha) * self._loss_ema + alpha * loss
        diff = loss - self._loss_ema
        self._loss_var = (1.0 - alpha) * self._loss_var + alpha * diff * diff

        # Normalized loss change (hypergradient proxy)
        # Positive h = loss went up relative to EMA → decrease LR
        # Negative h = loss went down → increase LR
        h = diff / (np.sqrt(self._loss_var) + 1e-8)
        h = float(np.clip(h, -1.0, 1.0))

        # RMSProp-style accumulator (not pure AdaGrad — decays old info)
        self._G_lr = self._meta_decay * self._G_lr + (1.0 - self._meta_decay) * h * h

        # Learning rate update
        lr_delta = self._hyper_lr * h / (np.sqrt(self._G_lr) + 1e-8)

        old_lr = self.optimizer.eta_base
        target_lr = old_lr - lr_delta

        # Rate limit: max 3% relative change per step
        max_change = 0.03 * old_lr
        target_lr = np.clip(target_lr, old_lr - max_change, old_lr + max_change)

        # Global bounds
        new_lr = float(np.clip(target_lr, self._lr_min, self._lr_max))
        self.optimizer.eta_base = new_lr

        self._prev_loss = loss

    def _bitunpack_to_bipolar(self, bitpacked: np.ndarray) -> np.ndarray:
        """Unpack uint32 bitpacked array to bipolar {-1, +1} float array."""
        bits = np.unpackbits(
            bitpacked.view(np.uint8), bitorder="little"
        )[: self.vsa_dim].astype(np.float32)
        return bits * 2.0 - 1.0

    def step(self, state_t: np.ndarray, state_t1: np.ndarray) -> TrainingResult:
        """Run one training step.

        Parameters
        ----------
        state_t : np.ndarray
            Current VSA state, bitpacked uint32 array.
        state_t1 : np.ndarray
            Next VSA state, bitpacked uint32 array.

        Returns
        -------
        TrainingResult
        """
        self.step_count += 1

        # 1. Compute true delta (XOR)
        true_delta = state_t ^ state_t1

        # 2. C++ forward + loss + backward + optimizer step
        loss = grilly_core.vsa_training_step(
            self.device, self.tape, self.model, state_t, true_delta,
            optimizer=self.optimizer,
        )

        # 3. Hypergradient: adapt optimizer LR based on loss curvature
        self._hypergradient_step(loss)

        # 4. Fast path: Hebbian weight update (batched every N steps)
        if self.enable_hebbian:
            pre_bipolar = self._bitunpack_to_bipolar(state_t)
            post_bipolar = self._bitunpack_to_bipolar(true_delta)
            self._pre_accum += pre_bipolar
            self._post_accum += post_bipolar
            self._accum_count += 1

            if self._accum_count >= self.hebbian_interval:
                inv_n = 1.0 / self._accum_count
                grilly_core.stdp_update_gpu(
                    self.device, self.stdp_weights,
                    self._pre_accum * inv_n,
                    self._post_accum * inv_n,
                    lr=self.hebbian_lr,
                    weight_min=-self.hebbian_clamp,
                    weight_max=self.hebbian_clamp,
                    decay=self.hebbian_decay,
                )
                self._pre_accum[:] = 0.0
                self._post_accum[:] = 0.0
                self._accum_count = 0

        # 5. Slow path: hippocampal recording + dream consolidation
        dream_report = None
        if self.dream_interval > 0:
            self.consolidator.record_episode(state_t, state_t1)
            if self.step_count % self.dream_interval == 0:
                dream_report = self.consolidator.dream(
                    self.world_model, cycles=self.dream_cycles
                )

        # 7. GPU yield — prevent 100% GPU saturation on Windows
        if self._gpu_yield_s > 0:
            time.sleep(self._gpu_yield_s)

        # Sample Hebbian norm periodically (every 100 steps) to avoid readback cost
        hebbian_norm = 0.0
        if self.enable_hebbian and self.step_count % 100 == 0:
            hebbian_norm = self.stdp_weights.norm()

        return TrainingResult(
            loss=loss,
            hebbian_weight_norm=hebbian_norm,
            dream_report=dream_report,
            world_model_fact_count=self.world_model.fact_count,
            effective_lr=self.optimizer.eta_base,
        )
