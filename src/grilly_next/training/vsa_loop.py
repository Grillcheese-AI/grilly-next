"""VSA Training Loop with dual-path learning: STDP + Hippocampal Consolidation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

import grilly_core
from grilly.functional.snn import reset_net
from grilly.nn.snn_neurons import IFNode
from grilly.nn.snn_synapses import DualTimescaleSynapse
from grilly.nn import STDPLayer


@dataclass
class TrainingResult:
    loss: float
    stdp_weight_norm: float
    dream_report: Optional[object] = None  # DreamReport from C++
    world_model_fact_count: int = 0


class VSATrainingLoop:
    """Integrated VSA training with STDP fast path and hippocampal slow path.

    Parameters
    ----------
    device : grilly_core.Device
        GPU device context.
    d_model : int
        Transformer hidden dimension (default 768).
    vsa_dim : int
        VSA vector dimension (default 10240).
    K : int
        Number of future trajectory branches (default 4, adjustable).
    tau_fast : float
        Fast synapse time constant for STDP path (default 2.0).
    tau_slow : float
        Slow synapse time constant for hippocampal path (default 20.0).
    dream_interval : int
        Steps between hippocampal dream cycles (default 1000).
    dream_cycles : int
        Synthetic mutations per dream (default 128).
    seed : int
        Random seed for hypernetwork init (default 42).
    """

    def __init__(
        self,
        device,
        d_model: int = 768,
        vsa_dim: int = 10240,
        K: int = 4,
        tau_fast: float = 2.0,
        tau_slow: float = 20.0,
        dream_interval: int = 1000,
        dream_cycles: int = 128,
        seed: int = 42,
    ):
        self.device = device
        self.vsa_dim = vsa_dim
        self.K = K
        self.dream_interval = dream_interval
        self.dream_cycles = dream_cycles
        self.step_count = 0

        # C++ components
        self.model = grilly_core.VSAHypernetwork(
            device, d_model=d_model, vsa_dim=vsa_dim, K=K, seed=seed
        )
        self.tape = grilly_core.TapeContext(device)
        self.consolidator = grilly_core.HippocampalConsolidator(max_capacity=10000)
        self.world_model = grilly_core.WorldModel(device)

        # SNN components (from grilly framework)
        self.spike_layer = IFNode(v_threshold=0.0, v_reset=0.0, step_mode="s")
        self.dual_synapse_pre = DualTimescaleSynapse(
            tau_fast=tau_fast, tau_slow=tau_slow, use_gpu=False
        )
        self.dual_synapse_post = DualTimescaleSynapse(
            tau_fast=tau_fast, tau_slow=tau_slow, use_gpu=False
        )
        self.stdp = STDPLayer(in_features=vsa_dim, out_features=vsa_dim)

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

        # 2. C++ forward + loss + backward
        loss = grilly_core.vsa_training_step(
            self.device, self.tape, self.model, state_t, true_delta
        )

        # 3. Fast path: STDP weight update
        pre_bipolar = self._bitunpack_to_bipolar(state_t)
        pre_spikes = (pre_bipolar > 0).astype(np.float32)  # {0, 1}

        post_bipolar = self._bitunpack_to_bipolar(true_delta)
        post_spikes = (post_bipolar > 0).astype(np.float32)

        # Filter through dual-timescale synapses
        pre_filtered = self.dual_synapse_pre(
            pre_spikes.reshape(1, -1)
        )
        post_filtered = self.dual_synapse_post(
            post_spikes.reshape(1, -1)
        )

        # STDP update: Hebbian outer product dW = a_plus * (post * pre^T) - a_minus * (pre * post^T)
        # This bypasses STDPLayer.update_weights() which has a kwarg mismatch
        # with the underlying VulkanSNN.stdp_learning() API.
        pre_flat = pre_filtered.reshape(-1)
        post_flat = post_filtered.reshape(-1)
        a_plus, a_minus = 0.01, 0.01
        dw = a_plus * np.outer(post_flat, pre_flat) - a_minus * np.outer(pre_flat, post_flat)
        self.stdp.weight += dw

        # 4. Slow path: hippocampal recording
        self.consolidator.record_episode(state_t, state_t1)

        # 5. Periodic dream consolidation
        dream_report = None
        if self.step_count % self.dream_interval == 0:
            dream_report = self.consolidator.dream(
                self.world_model, cycles=self.dream_cycles
            )

        # 6. Reset SNN membrane states
        reset_net(self.spike_layer)

        return TrainingResult(
            loss=loss,
            stdp_weight_norm=float(np.linalg.norm(self.stdp.weight)),
            dream_report=dream_report,
            world_model_fact_count=self.world_model.fact_count,
        )
