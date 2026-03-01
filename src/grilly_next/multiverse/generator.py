"""
Hilbert Multiverse Generator: Real-time Cognitive Tone Warping
Implements Annex H of the Grilly-Next Whitepaper.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import grilly_core

class CognitiveUniverse(IntEnum):
    """
    Phase offsets representing distinct semantic domains/tones.
    Translates continuous Hilbert metrics into VSA circular permutations.
    """
    ANALYTICAL = 0       # Base logical reality
    EMPATHETIC = 512     # Modulated for emotional alignment
    SKEPTICAL = 1024     # Modulated for critical/adversarial tone
    POETIC = 2048        # Modulated for rhythmic/aesthetic vocabulary
    ASSERTIVE = 4096     # Modulated for direct, authoritative phrasing

@dataclass
class MultiverseStep:
    state: np.ndarray
    decoded_text: str
    universe: CognitiveUniverse
    coherence_score: float

class MultiverseGenerator:
    """
    Partitions the K parallel futures into distinct Cognitive Universes.
    Allows real-time style-morphing without model retraining.
    """
    def __init__(
        self,
        device: grilly_core.Device,
        model: grilly_core.VSAHypernetwork,
        world_model: grilly_core.WorldModel,
        vsa_cache: grilly_core.VSACache,
        vsa_dim: int = 10240,
        K: int = 128
    ):
        self.device = device
        self.model = model
        self.world_model = world_model
        self.vsa_cache = vsa_cache
        self.vsa_dim = vsa_dim
        self.K = K

    def encode_prompt(self, text: str) -> np.ndarray:
        """
        Converts a human text prompt into a discrete geometric VSA state.
        Calls the C++ BLAKE3/Binding encoding pipeline.
        """
        # Assuming grilly_core exposes the C++ text-to-vsa encoding function
        # This returns a bitpacked uint32 array of shape (320,)
        return grilly_core.encode_text(text, self.vsa_dim)

    def insert_with_text(self, text: str, state: np.ndarray) -> bool:
        """
        Populate the VSA Cache with a known geometric state and its human-readable translation.
        This enables the Reverse-LSH lookup during Many-Worlds generation.
        """
        try:
            # Assumes the C++ VSACache binding accepts the string label
            self.vsa_cache.insert_packed(state, text)
            return True
        except Exception:
            # Fallback if the binding takes specific kwargs
            try:
                self.vsa_cache.insert_packed(state, surprise=1.0, label=text)
                return True
            except Exception as e:
                print(f"Cache insert failed: {e}")
                return False

    def generate_from_prompt(
        self,
        prompt: str,
        context_state: Optional[np.ndarray] = None,
        target_universe: CognitiveUniverse = CognitiveUniverse.ANALYTICAL,
        max_steps: int = 50
    ) -> List[MultiverseStep]:
        """
        The conversational entry point. 
        Binds the user prompt to the ongoing conversational state using hardware XOR, 
        then executes the Many-Worlds generation.
        """
        # 1. Encode the raw text into a discrete geometric hypervector
        prompt_state = self.encode_prompt(prompt)

        # 2. State Binding: XOR the prompt with the ongoing conversational context.
        # In VSA, XOR combines two concepts perfectly without losing information.
        if context_state is not None:
            initial_state = context_state ^ prompt_state
        else:
            initial_state = prompt_state

        # 3. Collapse the wave-function into the requested cognitive universe
        return self.generate_warped(initial_state, target_universe, max_steps)

    def generate_warped(
        self, 
        initial_state: np.ndarray, 
        target_universe: CognitiveUniverse = CognitiveUniverse.ANALYTICAL,
        max_steps: int = 50
    ) -> List[MultiverseStep]:
        """
        Executes Many-Worlds generation, collapsing the superposition
        into the requested cognitive universe.
        """
        current_state = initial_state.copy()
        trajectory = []
        tape = grilly_core.TapeContext(self.device)

        # We partition the K=128 futures into 4 distinct universes (32 branches each)
        # This allows the model to explore multiple tones simultaneously
        partitions = [
            CognitiveUniverse.ANALYTICAL,
            CognitiveUniverse.EMPATHETIC,
            CognitiveUniverse.SKEPTICAL,
            target_universe # Ensure the requested universe is fully represented
        ]
        
        # Create the phase_shifts array to pass to C++
        phase_shifts = np.zeros(self.K, dtype=np.uint32)
        branches_per_partition = self.K // len(partitions)
        
        for i, universe in enumerate(partitions):
            start_idx = i * branches_per_partition
            end_idx = start_idx + branches_per_partition
            phase_shifts[start_idx:end_idx] = int(universe.value)

        for step in range(max_steps):
            tape.reset()

            # 1. Hypernetwork Proposes K Deltas [1, K, 10240]
            continuous_deltas = self.model.forward_inference(tape, current_state)
            
            # 2. Vulkan ARV & Cognitive Warp Dispatch (~1.5ms)
            evaluation = grilly_core.evaluate_many_worlds(
                self.device, 
                current_state, 
                continuous_deltas, 
                phase_shifts, # Inject the Hilbert Phase Shifts
                self.world_model
            )
            
            scores = evaluation["coherence_scores"]
            future_states = evaluation["future_states"]
            
            # 3. Collapse the wave-function into the target universe
            # We isolate only the scores from the partition matching target_universe
            target_partition_idx = partitions.index(target_universe)
            start_k = target_partition_idx * branches_per_partition
            end_k = start_k + branches_per_partition
            
            target_scores = scores[start_k:end_k]
            
            # Find the most coherent logical path *within* the requested stylistic universe
            local_best_k = int(np.argmin(target_scores))
            global_best_k = start_k + local_best_k
            best_score = target_scores[local_best_k]

            # Hallucination Kill-Switch
            if best_score > self.world_model.hallucination_threshold():
                trajectory.append(MultiverseStep(
                    state=current_state,
                    decoded_text="[Generation Aborted: No Coherent Trajectory Available]",
                    universe=target_universe,
                    coherence_score=best_score
                ))
                break

            # 4. Commit to the Warped Future
            next_state = future_states[global_best_k]
            
            # 5. Decode from the VSA Cache
            # The cache natively resolves the shifted vector to the stylistically aligned text!
            lookup = self.vsa_cache.lookup_packed(self.device, next_state, top_k=1)
            decoded_text = lookup["strings"][0] if lookup["distances"][0] < (self.vsa_dim * 0.45) else "[Uncharted]"
            
            trajectory.append(MultiverseStep(
                state=next_state,
                decoded_text=decoded_text,
                universe=target_universe,
                coherence_score=best_score
            ))

            current_state = next_state

        return trajectory