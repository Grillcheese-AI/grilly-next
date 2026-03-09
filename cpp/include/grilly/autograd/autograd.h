#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "grilly/autograd/tape_arena.h"
#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/cubemind/types.h"
#include "grilly/pipeline_cache.h"

namespace grilly {
namespace autograd {

// ── C++ Autograd Engine for Grilly ──────────────────────────────────────
//
// Reverse-mode automatic differentiation via a Wengert list recorded on
// a bump-allocated TapeArena. The core insight:
//
//   Because operations are recorded sequentially during the forward pass,
//   the allocation order IS a valid topological ordering. Backward is just
//   a linear pointer walk: tail → prev → prev → ... → nullptr.
//
// Design constraints (The Golden Rule):
//   Nodes live in the arena and are freed by reset() WITHOUT destructors.
//   Therefore nodes must NEVER own heap memory (no std::vector, no
//   std::shared_ptr, no std::string). They hold:
//     - Scalars (uint32_t, float, bool)
//     - Buffer IDs referencing BufferPool-managed GPU memory
//     - Raw pointers to other arena-allocated objects
//     - Fixed-size arrays
//
// GPU data lifecycle:
//   Tensor weights, activations, and gradients live in BufferPool (VRAM).
//   Nodes hold uint64_t buffer_ids — lightweight handles. When the arena
//   resets, the graph metadata vanishes but VRAM data is untouched.
//   BufferPool manages VRAM allocation/deallocation independently.

// ── TensorRef ───────────────────────────────────────────────────────────
//
// Lightweight tensor descriptor for the autograd graph. Fixed-size
// (no heap), safe for bump allocation. Max 8 dimensions covers all
// practical neural network shapes.

struct TensorRef {
    uint64_t buffer_id;      // VkBuffer handle (0 = invalid/none)
    uint32_t ndim;           // Number of dimensions (max 8)
    uint32_t shape[8];       // Shape (e.g., {batch, seq_len, hidden_dim})
    uint32_t dtype;          // 0=f32, 1=f16, 2=u32, 3=i32
    bool requires_grad;

    size_t numel() const {
        size_t n = 1;
        for (uint32_t i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }

    size_t size_bytes() const {
        size_t elem_size = (dtype == 1) ? 2 : 4;
        return numel() * elem_size;
    }

    static TensorRef none() {
        TensorRef ref{};
        ref.buffer_id = 0ULL;
        ref.ndim = 0;
        ref.requires_grad = false;
        return ref;
    }

    bool valid() const { return buffer_id != 0ULL; }
};

// ── OpType ──────────────────────────────────────────────────────────────
//
// Each tag maps to a specific GLSL backward shader. The backward engine
// dispatches the right shader based on this tag — no virtual function
// overhead for the common path.

enum class OpType : uint8_t {
    // Arithmetic
    Add = 0,
    Sub,
    Mul,
    Div,
    Neg,
    Pow,

    // Linear algebra
    MatMul,
    Linear,

    // Activations
    ReLU,
    GELU,
    SiLU,
    Tanh,
    Sigmoid,
    Softmax,

    // Normalization
    LayerNorm,
    RMSNorm,

    // Attention
    FlashAttention2,

    // Convolution
    Conv2d,
    Conv1d,

    // Reductions
    Sum,
    Mean,
    Max,
    Min,

    // Shape ops (backward just reshapes gradients, no shader needed)
    Reshape,
    Transpose,
    Slice,

    // Loss functions
    CrossEntropy,
    MSELoss,

    // CubeMind — Surprise-Momentum optimizer hook
    CubeMindSurprise,

    // CubeMind — Temporal Foresight (counterfactual contradiction penalty)
    TemporalSurprise,

    // VSA Hypernetwork
    VSAUnpackProject,      // Fused bitunpack -> fp32 -> linear projection
    VSASurrogateLoss,      // Hinge + contrastive margin loss on K branches
    VSALoRAExpand,         // LoRA basis expansion: delta_k = B @ c_k
    VSACosineBlendLoss,    // Cosine blend loss with router + oracle distillation

    _Count  // sentinel
};

// ── TemporalSurpriseParams ────────────────────────────────────────────
//
// Packed into Node::params[64] for TemporalSurprise nodes.
// Stores counterfactual evaluation results computed during the forward pass.
// The backward pass reads these to compute the gradient multiplier.
//
// Memory: 24 bytes (fits comfortably in the 64-byte params buffer).
//
struct TemporalSurpriseParams {
    float avg_coherence;        // Mean coherence across N counterfactual branches
    float avg_contradiction;    // Mean surprise (contradiction) across branches
    float temporal_multiplier;  // Pre-computed: 1.0 - 2*avg_contradiction
    float alpha;                // Sensitivity scaling (default 1.0)
    uint32_t num_branches;      // Number of counterfactual branches evaluated
    uint32_t dt;                // Time steps projected forward
};
static_assert(sizeof(TemporalSurpriseParams) <= 64,
              "TemporalSurpriseParams must fit in Node::params[64]");

// ── VSASurrogateLossParams ──────────────────────────────────────────────
//
// Packed into Node::params[64] for VSASurrogateLoss nodes.
// Stores hinge + contrastive margin loss configuration for K future
// trajectory branches. Forward pass writes winning_k, runner_up_k, and
// loss_value; backward pass reads them to route gradients.
//
// Memory: 44 bytes (fits comfortably in the 64-byte params buffer).
//
struct VSASurrogateLossParams {
    float gamma;           // Hinge margin (default 1.0)
    float delta_margin;    // Contrastive margin (default 1.0)
    float lambda;          // Contrastive weight (default 0.3)
    uint32_t K;            // Number of future trajectories
    uint32_t D;            // VSA dimension (10240)
    uint32_t winning_k;    // Written by forward, read by backward
    uint32_t runner_up_k;  // Written by forward, read by backward
    float loss_value;      // Written by forward
    float temperature;     // Gumbel-softmax temperature (default 1.0)
    float diversity_lambda; // Diversity loss weight (default 0.01)
};
static_assert(sizeof(VSASurrogateLossParams) <= 64,
              "Must fit in Node::params[64]");

// ── VSALoRAExpandParams ──────────────────────────────────────────────
//
// Packed into Node::params[64] for VSALoRAExpand nodes.
// Stores dimensions for LoRA basis expansion: delta_k = B @ c_k.
//
// Memory: 12 bytes.
//
struct VSALoRAExpandParams {
    uint32_t K;         // Number of branches
    uint32_t D;         // VSA dimension (vsa_dim)
    uint32_t rank;      // LoRA basis rank
};
static_assert(sizeof(VSALoRAExpandParams) <= 64,
              "Must fit in Node::params[64]");

// ── VSACosineBlendLossParams ─────────────────────────────────────────
//
// Packed into Node::params[64] for VSACosineBlendLoss nodes.
// Cosine blend loss with router-weighted branch mixing, oracle
// distillation (KL divergence), and entropy regularization.
//
// Memory: 28 bytes.
//
struct VSACosineBlendLossParams {
    uint32_t K;              // Number of branches
    uint32_t D;              // VSA dimension (vsa_dim)
    float temperature;       // Router softmax temperature
    float lambda_distill;    // Oracle KL distillation weight
    float lambda_entropy;    // Entropy regularization weight
    float loss_value;        // Written by forward pass
    float tau_oracle;        // Oracle softmax temperature
};
static_assert(sizeof(VSACosineBlendLossParams) <= 64,
              "Must fit in Node::params[64]");

// ── Node ────────────────────────────────────────────────────────────────
//
// A single operation in the computation graph. Allocated on the TapeArena.
// Threaded into a Wengert list via prev_in_tape.
//
// The backward engine walks: tail → prev_in_tape → prev_in_tape → nullptr
// For each node with grad_output_buffer != 0, it dispatches the backward
// shader and accumulates gradients into input buffers with +=.

static constexpr uint32_t kMaxNodeIO = 8;

struct Node {
    // ── Wengert list link ──
    Node* prev_in_tape;              // Previous node in allocation order

    // ── Operation identity ──
    OpType op;                       // Which backward shader to dispatch
    uint32_t seq;                    // Allocation sequence number (debugging)

    // ── Input/Output tensor descriptors ──
    uint32_t num_inputs;
    uint32_t num_outputs;
    TensorRef inputs[kMaxNodeIO];
    TensorRef outputs[kMaxNodeIO];

    // ── Saved state for backward ──
    // Buffer IDs for tensors needed during backward but not graph edges.
    // E.g., the weight matrix in Linear, the pre-activation values in ReLU.
    uint32_t num_saved;
    uint64_t saved_buffer_ids[kMaxNodeIO];

    // ── Gradient buffer IDs ──
    // Filled during backward(). grad_output_buffer holds dL/d(output).
    // grad_input_buffers[] are allocated by the backward engine and hold
    // dL/d(input_i) after this node's backward shader runs.
    uint64_t grad_output_buffer;     // Gradient flowing in from downstream
    uint64_t grad_input_buffers[kMaxNodeIO];  // Gradients flowing to inputs

    // ── Per-op scalar parameters ──
    // Inline push-constant data for the backward shader.
    // 64 bytes covers all current shaders (largest: attention at ~48 bytes).
    uint8_t params[64];
    uint32_t params_size;

    // ── CubeMind emotion state ──
    // Only populated for CubeMindSurprise nodes. Stored inline to avoid
    // any heap allocation. The surprise value modulates the learning rate.
    cubemind::EmotionState emotion;
};

// ── allocate_node() implementation ──────────────────────────────────────
//
// Now that Node is defined, we can implement the Wengert list threading.
// This must be in the header because it's a template.

template <typename T, typename... Args>
T* TapeArena::allocate_node(Args&&... args) {
    T* node = allocate<T>(std::forward<Args>(args)...);
    node->prev_in_tape = tail_;
    tail_ = node;
    return node;
}

// ── BackwardEngine ──────────────────────────────────────────────────────
//
// The execution engine for reverse-mode AD. Given a loss node, it:
//   1. Seeds dL/d(loss) = 1.0 into the loss node's grad_output_buffer
//   2. Walks the Wengert list backward: tail → prev → prev → nullptr
//   3. For each node with a populated grad_output_buffer:
//      a. Dispatches the appropriate backward shader (fnn-linear-backward.glsl, etc.)
//      b. Allocates gradient buffers for inputs from BufferPool
//      c. Accumulates gradients into input nodes via += (handles fan-out)
//   4. After the walk, leaf tensors have accumulated gradients in BufferPool
//
// The engine holds no state between calls — it's a pure function of the
// arena's Wengert list. Thread safety comes from the thread-local arena.

class BackwardEngine {
public:
    BackwardEngine(BufferPool& pool, CommandBatch& batch, PipelineCache& cache);

    /// Run backward from `loss_node` through the tape.
    ///
    /// `tape`: the arena whose Wengert list we traverse
    /// `loss_node`: the node that produced the loss (must be in the tape)
    /// `grad_output_buffer`: buffer ID containing dL/d(loss), usually all 1.0
    ///
    /// After this call, each leaf input's gradient can be read from the
    /// buffer ID stored in the node's grad_input_buffers[].
    void backward(TapeArena& tape, Node* loss_node, uint64_t grad_output_buffer);

    /// Get the gradient buffer ID for a given input buffer.
    /// Searches the grad accumulation map populated during backward().
    /// Returns 0 if no gradient was computed for this buffer.
    uint64_t get_grad_buffer(uint64_t input_buffer_id) const;

    /// Clear gradient accumulation state (call between training steps).
    void clear_grads();

    struct Stats {
        uint32_t nodes_visited;      // Total nodes walked
        uint32_t nodes_with_grad;    // Nodes that had gradients to propagate
        uint32_t shaders_dispatched; // Vulkan backward shaders dispatched
        uint32_t cpu_fallbacks;      // Nodes that fell back to CPU
    };
    Stats last_stats() const { return stats_; }

private:
    /// Dispatch the backward shader for a single node.
    /// Reads node->grad_output_buffer, writes node->grad_input_buffers[].
    void dispatch_node_backward(Node* node);

    /// Accumulate gradient: if buffer already has a gradient, add to it.
    /// This handles fan-out: when tensor A feeds into ops B and C,
    /// both contribute gradients that get summed into A's grad buffer.
    void accumulate_grad(Node* target_node, uint32_t input_idx,
                         uint64_t grad_buffer);

    /// Shader dispatch helpers for specific op types
    void backward_linear(Node* node);
    void backward_linear_chunked(Node* node, uint32_t batch_seq,
                                  uint32_t output_dim, uint32_t input_dim);
    void backward_matmul(Node* node);
    void backward_relu(Node* node);
    void backward_gelu(Node* node);
    void backward_silu(Node* node);
    void backward_tanh(Node* node);
    void backward_sigmoid(Node* node);
    void backward_softmax(Node* node);
    void backward_layernorm(Node* node);
    void backward_attention(Node* node);
    void backward_conv2d(Node* node);
    void backward_conv1d(Node* node);
    void backward_add(Node* node);
    void backward_sub(Node* node);
    void backward_mul(Node* node);
    void backward_div(Node* node);
    void backward_cross_entropy(Node* node);
    void backward_mse(Node* node);
    void backward_cubemind_surprise(Node* node);
    void backward_temporal_surprise(Node* node);

    // Shape ops — no shader needed, just reshape the gradient buffer
    void backward_reshape(Node* node);
    void backward_transpose(Node* node);
    void backward_sum(Node* node);
    void backward_mean(Node* node);

    // VSA Hypernetwork backward handlers
    void backward_vsa_surrogate_loss(Node* node);
    void backward_vsa_unpack_project(Node* node);
    void backward_vsa_lora_expand(Node* node);
    void backward_vsa_cosine_blend_loss(Node* node);

    BufferPool& pool_;
    CommandBatch& batch_;
    PipelineCache& cache_;
    Stats stats_{};

    // Per-step backward temporary buffers (released by TapeContext::end())
    std::vector<GrillyBuffer> backward_bufs_;
    friend class TapeContext;  // TapeContext releases these in end()

    // Gradient accumulation: input_buffer_id → accumulated_grad_buffer_id
    // Using a flat array (arena-friendly) instead of std::unordered_map.
    // Max 4096 unique tensors per forward pass is generous.
    static constexpr size_t kMaxGradEntries = 4096;
    struct GradEntry {
        uint64_t buffer_id;       // The input tensor's buffer ID
        uint64_t grad_buffer_id;  // The accumulated gradient buffer ID
    };
    GradEntry grad_table_[kMaxGradEntries];
    uint32_t grad_count_ = 0;

    // Overflow slot — returned when grad_table_ is full (should never happen).
    // Member variable avoids returning address of local/temporary.
    uint64_t overflow_slot_ = 0;

    /// Find or insert a grad entry. Returns the grad_buffer_id slot.
    uint64_t& find_or_insert_grad(uint64_t buffer_id);
};

// ── TapeContext ─────────────────────────────────────────────────────────
//
// High-level API wrapping TapeArena + BackwardEngine for Python callers.
// Manages the record/backward/reset lifecycle.

class TapeContext {
public:
    TapeContext(BufferPool& pool, CommandBatch& batch, PipelineCache& cache,
                size_t arena_capacity = TapeArena::kDefaultCapacity);

    /// Begin recording a new forward pass.
    void begin();

    /// Record an operation. Returns the arena-allocated Node*.
    Node* record_op(OpType op,
                    const TensorRef* inputs, uint32_t num_inputs,
                    const TensorRef* outputs, uint32_t num_outputs,
                    const void* params = nullptr, uint32_t params_size = 0);

    /// Save buffer IDs needed for backward (weight, activation cache, etc.)
    void save_for_backward(Node* node, const uint64_t* buffer_ids, uint32_t count);

    /// Run backward from the loss node.
    void backward(Node* loss_node, uint64_t grad_output_buffer);

    /// Get gradient buffer for a specific input.
    uint64_t get_grad_buffer(uint64_t input_buffer_id) const;

    /// End the tape: reset arena, clear gradients, release step buffers.
    void end();

    /// Acquire a GPU buffer that will be auto-released when end() is called.
    /// Use for per-step temporaries (activations, intermediates, gradients).
    /// Do NOT use for model weights — those should use pool.acquire() directly.
    GrillyBuffer acquire_temp(size_t bytes);

    bool is_recording() const { return recording_; }
    TapeArena& arena() { return arena_; }
    BufferPool& pool() { return pool_; }
    BackwardEngine::Stats last_backward_stats() const { return engine_.last_stats(); }

    // Arena stats
    size_t arena_bytes_used() const { return arena_.bytes_used(); }
    float arena_utilization() const { return arena_.utilization(); }

private:
    BufferPool& pool_;
    TapeArena arena_;
    BackwardEngine engine_;
    bool recording_ = false;
    uint32_t seq_counter_ = 0;
    std::vector<GrillyBuffer> step_bufs_;
};

}  // namespace autograd
}  // namespace grilly
