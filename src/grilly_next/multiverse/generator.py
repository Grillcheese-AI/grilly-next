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
    """Phase offsets for cognitive tone warping (Annex H, Table H.1).

    Each value is a circular permutation offset applied to the K continuous
    deltas before Many-Worlds evaluation. The shift preserves Hamming weight
    and expected orthogonality (Theorem 4), so style transfer is lossless.
    """
    ANALYTICAL = 0       # Base logical reality
    EMPATHETIC = 512     # Modulated for emotional alignment
    SKEPTICAL  = 1024    # Modulated for critical/adversarial tone
    POETIC     = 2048    # Modulated for rhythmic/aesthetic vocabulary
    ASSERTIVE  = 4096    # Modulated for direct, authoritative phrasing


@dataclass
class MultiverseStep:
    """A single step in a multiverse generation trajectory."""
    state: np.ndarray          # Bitpacked uint32 future state
    decoded_text: str          # Human-readable decoded text
    universe: CognitiveUniverse
    coherence_score: float     # Violation count (0 = coherent)


class MultiverseGenerator:
    """Partitions K parallel futures into distinct Cognitive Universes.

    Combines VSAHypernetwork (forward inference), Many-Worlds evaluation,
    ResonatorNetwork (decoding), and VSACache (associative memory) into
    a unified generation pipeline with real-time cognitive tone warping.
    """

    def __init__(
        self,
        device,
        model,
        world_model,
        vsa_cache,
        vsa_dim: int = 10240,
        K: int = 128,
        hallucination_threshold: Optional[float] = None,
        resonator: Optional[object] = None,
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
        # Text index: maps VSACache insertion order -> text string.
        # Populated by insert_with_text().
        self._text_index: List[str] = []
        # TextEncoder for converting raw text -> bitpacked VSA states.
        self._text_encoder = grilly_core.TextEncoder(dim=vsa_dim)
        # ResonatorNetwork for decoding future states back to words.
        # If not provided, falls back to VSACache lookup only.
        self.resonator = resonator

    # ── Phase Shifting (Annex H, Section H.2) ────────────────────────

    @staticmethod
    def _apply_phase_shift(
        deltas: np.ndarray,
        phase_shifts: np.ndarray,
    ) -> np.ndarray:
        """Apply circular permutation to each row of the delta matrix.

        S_phi[i] = S[(i - phi) mod D]

        This maps continuous Hilbert phase offsets to O(1) hardware
        operations on the bitpacked representation.
        """
        result = deltas.copy()
        for k in range(deltas.shape[0]):
            shift = int(phase_shifts[k])
            if shift != 0:
                result[k] = np.roll(deltas[k], -shift)
        return result

    # ── Cache & Decode ───────────────────────────────────────────────

    def insert_with_text(self, text: str, state: np.ndarray) -> bool:
        """Insert a state into VSACache and record the text mapping."""
        try:
            self.vsa_cache.insert_packed(state, surprise=1.0, stress=0.0)
            self._text_index.append(text)
            return True
        except Exception:
            return False

    def _decode_from_cache(self, state: np.ndarray) -> str:
        """Decode a bitpacked state via VSACache nearest-neighbor lookup."""
        result = self.vsa_cache.lookup_packed(self.device, state, top_k=1)
        indices = result["indices"]
        distances = result["distances"]

        if len(indices) == 0:
            return "[Uncharted]"

        idx = int(indices[0])
        dist = float(distances[0])

        if dist > self.hallucination_threshold:
            return "[Uncharted]"

        if idx < len(self._text_index):
            return self._text_index[idx]
        return f"[Index {idx}]"

    def _decode_with_resonator(
        self, state: np.ndarray, max_tokens: int = 5
    ) -> str:
        """Decode a future state using the ResonatorNetwork.

        Uses batch_unbind with default positional roles to extract
        up to max_tokens words from the state bundle.
        """
        if self.resonator is None or self.resonator.codebook_size == 0:
            return self._decode_from_cache(state)

        roles = [_DEFAULT_ROLES[i % len(_DEFAULT_ROLES)]
                 for i in range(max_tokens)]
        positions = list(range(max_tokens))

        try:
            results = self.resonator.batch_unbind(state, roles, positions)
            words = []
            for r in results:
                if r["similarity"] > 0.1:  # Filter noise
                    words.append(r["word"])
            if words:
                return " ".join(words)
        except Exception:
            pass

        # Fall back to cache lookup
        return self._decode_from_cache(state)

    # ── Generation ───────────────────────────────────────────────────

    def generate_warped(
        self,
        initial_state: np.ndarray,
        target_universe: CognitiveUniverse = CognitiveUniverse.ANALYTICAL,
        max_steps: int = 50,
    ) -> List[MultiverseStep]:
        """Execute Many-Worlds generation with cognitive tone warping.

        Partitions K futures across cognitive universes, applies phase
        shifts, evaluates coherence, and collapses into the target
        universe at each step.
        """
        current_state = initial_state.copy()
        trajectory: List[MultiverseStep] = []

        # Partition K branches across universes
        universes = list(CognitiveUniverse)
        branches_per_universe = max(1, self.K // len(universes))

        # Build phase shift array
        phase_shifts = np.zeros(self.K, dtype=np.uint32)
        for i, universe in enumerate(universes):
            start = i * branches_per_universe
            end = min(start + branches_per_universe, self.K)
            phase_shifts[start:end] = universe.value

        for step in range(max_steps):
            # 1. Forward inference: current_state -> K continuous deltas
            tape = grilly_core.TapeContext(self.device)
            continuous_deltas = self.model.forward_inference(
                self.device, tape, current_state
            )

            # 2. Apply phase shifts (circular permutation in Python)
            warped_deltas = self._apply_phase_shift(
                continuous_deltas, phase_shifts
            )

            # 3. Many-Worlds coherence evaluation
            evaluation = grilly_core.evaluate_many_worlds(
                self.device, current_state, warped_deltas, self.world_model
            )

            # 4. Collapse: select best branch in the target universe partition
            target_idx = universes.index(target_universe)
            start_k = target_idx * branches_per_universe
            end_k = min(start_k + branches_per_universe, self.K)

            scores = evaluation.coherence_scores()
            target_scores = scores[start_k:end_k]
            local_best = int(np.argmin(target_scores))
            global_best = start_k + local_best
            best_score = float(target_scores[local_best])

            # 5. Hallucination kill-switch
            if best_score > self.hallucination_threshold:
                trajectory.append(MultiverseStep(
                    state=current_state,
                    decoded_text="[Halted: no coherent trajectory]",
                    universe=target_universe,
                    coherence_score=best_score,
                ))
                break

            # 6. Extract the best future state
            future_states = evaluation.future_states_array()
            words_per_vec = (self.vsa_dim + 31) // 32
            next_state = future_states[global_best, :words_per_vec].copy()

            # 7. Decode: resonator first, cache fallback
            decoded = self._decode_with_resonator(next_state)

            trajectory.append(MultiverseStep(
                state=next_state,
                decoded_text=decoded,
                universe=target_universe,
                coherence_score=best_score,
            ))

            current_state = next_state

        return trajectory

    # ── Query API ────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        universe: CognitiveUniverse = CognitiveUniverse.ANALYTICAL,
        max_steps: int = 5,
    ) -> List[MultiverseStep]:
        """Query the multiverse with a text string.

        Encodes the text as a WorldModel triple and generates
        a trajectory in the specified cognitive universe.
        """
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

        Requires: pip install fasttext-wheel huggingface-hub

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
