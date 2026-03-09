#pragma once

#include "grilly/autograd/autograd.h"
#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly::autograd {

/// Upload a BitpackedVec to a step-scoped GPU buffer and wrap as TensorRef(dtype=u32).
/// Buffer is auto-released when tape.end() is called.
TensorRef upload_bitpacked(TapeContext& tape,
                           const uint32_t* data,
                           uint32_t num_words,
                           uint32_t dim);

/// Dispatch the VSA surrogate loss forward pass on GPU.
/// Two-pass: dot products (GPU) -> argmax (CPU) -> hinge+contrastive (GPU)
/// Intermediate buffers are step-scoped (released on tape.end()).
float dispatch_vsa_loss_forward(TapeContext& tape,
                                CommandBatch& batch,
                                PipelineCache& cache,
                                Node* node);

/// Dispatch the VSA surrogate loss backward pass with a pre-allocated gradient buffer.
/// Called from BackwardEngine which manages buffer lifetime separately.
void dispatch_vsa_loss_backward_with_buf(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node,
                                          float grad_scale,
                                          GrillyBuffer& grad_buf);

/// Dispatch cosine blend loss forward pass.
/// Three-part loss: L_cosine + lambda_distill * L_KL - lambda_entropy * L_entropy.
/// Router weights are computed on CPU and uploaded for GPU blending.
float dispatch_cosine_blend_loss_forward(TapeContext& tape,
                                         CommandBatch& batch,
                                         PipelineCache& cache,
                                         Node* node,
                                         const float* router_weights,
                                         uint32_t batch_size);

/// Dispatch cosine blend loss backward pass.
void dispatch_cosine_blend_loss_backward(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node,
                                          float grad_scale,
                                          GrillyBuffer& grad_buf);

/// Dispatch LoRA expand backward pass.
/// Computes grad_coefficients and accumulates grad_B.
void dispatch_lora_expand_backward(BufferPool& pool,
                                    CommandBatch& batch,
                                    PipelineCache& cache,
                                    Node* node,
                                    float grad_scale,
                                    GrillyBuffer& grad_coeffs_buf,
                                    GrillyBuffer& grad_basis_buf);

/// Dispatch the VSA unpack + project forward pass.
void dispatch_vsa_unpack_project_forward(BufferPool& pool,
                                         CommandBatch& batch,
                                         PipelineCache& cache,
                                         Node* node);

/// Dispatch the VSA unpack + project backward pass.
/// Computes grad_W_proj and grad_b_proj. No gradient for VSA state.
void dispatch_vsa_unpack_project_backward(BufferPool& pool,
                                          CommandBatch& batch,
                                          PipelineCache& cache,
                                          Node* node);

}  // namespace grilly::autograd
