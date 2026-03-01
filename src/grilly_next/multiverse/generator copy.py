"""
Hilbert Multiverse Generator: Real-time Cognitive Tone Warping.
Implements Annex H of the Grilly-Next Whitepaper.

Phase offsets translate continuous Hilbert metrics into VSA circular
permutations (Section H.2): S_phi[i] = S[(i - phi) mod D].
Because circular shifting strictly preserves Hamming weight and expected
orthogonality (Theorem 4), style transfer is lossless.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

import grilly_core

# Simple positional dependency roles used when no parser is available.
_DEFAULT_ROLES = ["ROOT", "nsubj", "dobj", "iobj", "advmod", "amod", "prep", "pobj"]


class CognitiveUniverse(IntEnum):
    """Phase offsets representing distinct semantic domains/tones.

    Each value is a circular permutation offset applied to VSA vectors,
    placing the state into an orthogonal, non-interfering semantic space.
    """
    ANALYTICAL = 0       # Base logical reality
    EMPATHETIC = 512     # Modulated for emotional alignment
    SKEPTICAL = 1024     # Modulated for critical/adversarial tone
    POETIC = 2048        # Modulated for rhythmic/aesthetic vocabulary
    ASSERTIVE = 4096     # Modulated for direct, authoritative phrasing


@dataclass
class MultiverseStep:
    """One step of multiverse generation output."""
    state: np.ndarray
    decoded_text: str
    universe: CognitiveUniverse
    coherence_score: float


class MultiverseGenerator:
    """Partitions K parallel futures into distinct Cognitive Universes.

    Allows real-time style-morphing without model retraining by applying
    circular permutations (Annex H, Section H.2) to the hypernetwork's
    continuous deltas before Many-Worlds evaluation.

    Parameters
    ----------
    device : grilly_core.Device
        GPU device context.
    model : grilly_core.VSAHypernetwork
        Trained hypernetwork that predicts K future deltas.
    world_model : grilly_core.WorldModel
        WorldModel for coherence checking against known facts/constraints.
    vsa_cache : grilly_core.VSACache
        Cache for decoding bitpacked states back to text.
    vsa_dim : int
        VSA vector dimension (default 10240).
    K : int
        Number of parallel trajectory branches (default 128).
    hallucination_threshold : float
        Max violation count before aborting generation (default D * 0.45).
    """

    def __init__(
        self,
        device: grilly_core.Device,
        model: grilly_core.VSAHypernetwork,
        world_model: grilly_core.WorldModel,
        vsa_cache: grilly_core.VSACache,
        vsa_dim: int = 10240,
        K: int = 128,
        hallucination_threshold: Optional[float] = None,
    ):
        self.device = device
        self.model = model
        self.world_model = world_model
        self.vsa_cache = vsa_cache
        self.vsa_dim = vsa_dim
        self.K = K
        self.hallucination_threshold = (
            hallucination_threshold
            if hallucination_threshold is not None
            else vsa_dim * 0.45
        )
        # Text index: maps VSACache insertion order → text string.
        # Populated by insert_with_text().
        self._text_index: List[str] = []
        # TextEncoder for converting raw text → bitpacked VSA states.
        self._text_encoder = grilly_core.TextEncoder(dim=vsa_dim)

    def insert_with_text(
        self, text: str, packed_vec: np.ndarray,
        surprise: float = 1.0, stress: float = 0.0,
    ) -> bool:
        """Insert a bitpacked vector into VSACache and record its text.

        Returns True if successfully inserted (not evicted immediately).
        """
        ok = self.vsa_cache.insert_packed(packed_vec, surprise, stress)
        if ok:
            self._text_index.append(text)
        return ok

    def _decode_from_cache(self, packed_state: np.ndarray) -> str:
        """Look up a bitpacked state in the VSACache and return decoded text."""
        result = self.vsa_cache.lookup_packed(self.device, packed_state, top_k=1)
        distances = np.asarray(result["distances"])
        indices = np.asarray(result["indices"])

        if len(distances) == 0 or distances[0] > self.vsa_dim * 0.45:
            return "[Uncharted]"

        idx = int(indices[0])
        if idx < len(self._text_index):
            return self._text_index[idx]
        return f"[Index {idx}]"

    @staticmethod
    def _apply_phase_shift(
        deltas: np.ndarray, phase_shifts: np.ndarray
    ) -> np.ndarray:
        """Apply circular permutation to continuous deltas.

        Per Annex H Section H.2: S_phi[i] = S[(i - phi) mod D].
        This is implemented as np.roll on each branch's delta vector.

        Parameters
        ----------
        deltas : np.ndarray
            Float32 array of shape [K, D] — continuous predictions.
        phase_shifts : np.ndarray
            Uint32 array of shape [K] — phase offset per branch.

        Returns
        -------
        np.ndarray
            Phase-shifted deltas, same shape [K, D].
        """
        shifted = deltas.copy()
        unique_shifts = np.unique(phase_shifts)
        for shift in unique_shifts:
            if shift == 0:
                continue
            mask = phase_shifts == shift
            # np.roll with negative shift = shift left = S[(i - phi) mod D]
            shifted[mask] = np.roll(shifted[mask], -int(shift), axis=-1)
        return shifted

    def generate_warped(
        self,
        initial_state: np.ndarray,
        target_universe: CognitiveUniverse = CognitiveUniverse.ANALYTICAL,
        max_steps: int = 50,
    ) -> List[MultiverseStep]:
        """Execute Many-Worlds generation, collapsing into the target universe.

        Partitions K branches into 4 cognitive universes. The target universe
        always gets the last partition to ensure full representation.

        Parameters
        ----------
        initial_state : np.ndarray
            Bitpacked uint32 VSA state [words_per_vec].
        target_universe : CognitiveUniverse
            Which cognitive tone to collapse into.
        max_steps : int
            Maximum generation steps.

        Returns
        -------
        List[MultiverseStep]
            Trajectory of decoded steps in the target universe.
        """
        current_state = initial_state.copy()
        trajectory: List[MultiverseStep] = []

        # Partition K branches into 4 cognitive universes
        partitions = [
            CognitiveUniverse.ANALYTICAL,
            CognitiveUniverse.EMPATHETIC,
            CognitiveUniverse.SKEPTICAL,
            target_universe,  # Ensure target is fully represented
        ]

        # Build phase_shifts array [K]
        phase_shifts = np.zeros(self.K, dtype=np.uint32)
        branches_per_partition = self.K // len(partitions)
        for i, universe in enumerate(partitions):
            start_idx = i * branches_per_partition
            end_idx = start_idx + branches_per_partition
            phase_shifts[start_idx:end_idx] = int(universe.value)

        for step in range(max_steps):
            # Fresh tape per step (forward_inference calls tape.begin())
            tape = grilly_core.TapeContext(self.device)

            # 1. Hypernetwork forward: get K continuous deltas [K, D]
            continuous_deltas = self.model.forward_inference(
                self.device, tape, current_state
            )

            # 2. Apply Hilbert phase shifts (circular permutation, Annex H.2)
            warped_deltas = self._apply_phase_shift(continuous_deltas, phase_shifts)

            # 3. Evaluate Many-Worlds against WorldModel constraints
            evaluation = grilly_core.evaluate_many_worlds(
                self.device, current_state, warped_deltas, self.world_model
            )

            scores = evaluation.coherence_scores()
            future_states = evaluation.future_states_array()

            # 4. Collapse into the target universe's partition
            target_partition_idx = partitions.index(target_universe)
            start_k = target_partition_idx * branches_per_partition
            end_k = start_k + branches_per_partition

            target_scores = scores[start_k:end_k]

            # Find the most coherent path within the target universe
            # (fewest violations = lowest score)
            local_best_k = int(np.argmin(target_scores))
            global_best_k = start_k + local_best_k
            best_score = float(target_scores[local_best_k])

            # Hallucination kill-switch
            if best_score > self.hallucination_threshold:
                trajectory.append(MultiverseStep(
                    state=current_state,
                    decoded_text="[Generation Aborted: No Coherent Trajectory]",
                    universe=target_universe,
                    coherence_score=best_score,
                ))
                break

            # 5. Commit to the best trajectory in the target universe
            next_state = future_states[global_best_k].copy()

            # 6. Decode from VSA cache
            decoded_text = self._decode_from_cache(next_state)

            trajectory.append(MultiverseStep(
                state=next_state,
                decoded_text=decoded_text,
                universe=target_universe,
                coherence_score=best_score,
            ))

            current_state = next_state

        return trajectory

    def query(
        self,
        text: str,
        universe: CognitiveUniverse = CognitiveUniverse.ANALYTICAL,
        max_steps: int = 10,
    ) -> List[MultiverseStep]:
        """Query the multiverse with a natural-language input.

        Encodes the text as an SVC triple via the WorldModel, then runs
        Many-Worlds generation in the requested cognitive universe.

        Parameters
        ----------
        text : str
            Input text (treated as a verb-centric SVC triple).
        universe : CognitiveUniverse
            Target cognitive tone.
        max_steps : int
            Maximum generation steps.

        Returns
        -------
        List[MultiverseStep]
            Decoded trajectory in the target universe.
        """
        # Encode text as a minimal SVC triple: ("_", text, "_")
        # For richer encoding, the caller should use encode_triple directly.
        state = self.world_model.encode_triple("_", text, "_")
        return self.generate_warped(state, universe, max_steps)

    def query_all_universes(
        self,
        text: str,
        max_steps: int = 5,
    ) -> dict:
        """Query the multiverse across all cognitive universes at once.

        Returns a dict mapping CognitiveUniverse -> List[MultiverseStep].
        """
        state = self.world_model.encode_triple("_", text, "_")
        results = {}
        for universe in CognitiveUniverse:
            results[universe] = self.generate_warped(
                state, universe, max_steps
            )
        return results

    # ── Prompt-based API (Annex H.3) ─────────────────────────────────

    def encode_prompt(self, text: str) -> np.ndarray:
        """Convert raw text into a bitpacked VSA state via TextEncoder.

        Tokenizes on whitespace, assigns simple positional dependency
        roles, and encodes via the C++ TextEncoder pipeline:
        token filler x role x position -> bundle -> bitpack.

        Without FastText fillers loaded, uses BLAKE3 fallback fillers
        (random-but-deterministic). With FastText loaded via
        load_fasttext_from_hf(), tokens get semantically meaningful
        bipolar vectors.
        """
        tokens = text.lower().split()
        if not tokens:
            tokens = ["_"]

        # Assign positional dependency roles (cycling through defaults)
        roles = [_DEFAULT_ROLES[i % len(_DEFAULT_ROLES)] for i in range(len(tokens))]
        positions = list(range(len(tokens)))

        result = self._text_encoder.encode_sentence(tokens, roles, positions)
        return np.asarray(result["data"], dtype=np.uint32)

    def generate_from_prompt(
        self,
        prompt: str,
        context_state: Optional[np.ndarray] = None,
        target_universe: CognitiveUniverse = CognitiveUniverse.ANALYTICAL,
        max_steps: int = 50,
    ) -> List[MultiverseStep]:
        """Conversational entry point: encode prompt and generate.

        Encodes the prompt text into a VSA state via TextEncoder, then
        optionally XOR-binds with a context state (e.g., the last step's
        state from a prior turn) to create a composite query that carries
        forward conversational context.

        Parameters
        ----------
        prompt : str
            Natural-language input text.
        context_state : np.ndarray, optional
            Bitpacked uint32 state from a previous generation step.
            XOR-binding composes prompt and context into a single
            quasi-orthogonal representation (VSA binding property).
        target_universe : CognitiveUniverse
            Which cognitive tone to collapse into.
        max_steps : int
            Maximum generation steps.

        Returns
        -------
        List[MultiverseStep]
            Trajectory of decoded steps in the target universe.
        """
        prompt_state = self.encode_prompt(prompt)

        if context_state is not None:
            # XOR binding: combines concepts without information loss.
            # The result is quasi-orthogonal to both inputs but
            # recoverable via unbinding (XOR is its own inverse).
            initial_state = context_state ^ prompt_state
        else:
            initial_state = prompt_state

        return self.generate_warped(initial_state, target_universe, max_steps)

    def load_fasttext_from_hf(
        self,
        repo_id: str = "facebook/fasttext-en-vectors",
        top_k: int = 50000,
    ) -> int:
        """Download FastText vectors from HuggingFace and register as fillers.

        Downloads the model.bin from the specified HuggingFace repo,
        loads the top_k most frequent word vectors, and projects each
        through the TextEncoder's LSH random projection (300D float ->
        D-dimensional bipolar {-1,+1}).

        Requires: pip install fasttext huggingface_hub

        Parameters
        ----------
        repo_id : str
            HuggingFace model repo (default: facebook/fasttext-en-vectors).
        top_k : int
            Number of words to load (most frequent first). Default 50000.

        Returns
        -------
        int
            Number of fillers successfully registered.
        """
        try:
            from huggingface_hub import hf_hub_download
            import fasttext
        except ImportError:
            raise ImportError(
                "FastText streaming requires: uv pip install fasttext-wheel huggingface-hub"
            )

        model_path = hf_hub_download(repo_id=repo_id, filename="model.bin")
        ft_model = fasttext.load_model(model_path)

        words = ft_model.get_words()
        count = 0
        for word in words[:top_k]:
            if self._text_encoder.has_filler(word):
                continue
            vec = ft_model.get_word_vector(word)
            bipolar = self._text_encoder.project_to_bipolar(
                vec.astype(np.float32)
            )
            self._text_encoder.add_filler(word, bipolar)
            count += 1

        return count
