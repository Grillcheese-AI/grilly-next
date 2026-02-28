#pragma once

#include "grilly/autograd/autograd.h"
#include "grilly/buffer_pool.h"
#include "grilly/command_batch.h"
#include "grilly/pipeline_cache.h"

namespace grilly::autograd {

/// Upload a BitpackedVec to a GPU buffer and wrap as TensorRef(dtype=u32).
TensorRef upload_bitpacked(BufferPool& pool,
                           const uint32_t* data,
                           uint32_t num_words,
                           uint32_t dim);

/// Dispatch the VSA surrogate loss forward pass on GPU.
/// Two-pass: dot products (GPU) -> argmax (CPU) -> hinge+contrastive (GPU)
float dispatch_vsa_loss_forward(BufferPool& pool,
                                CommandBatch& batch,
                                PipelineCache& cache,
                                Node* node);

/// Dispatch the VSA surrogate loss backward pass on GPU.
/// Sparse gradient routing: only winner + runner-up branches.
void dispatch_vsa_loss_backward(BufferPool& pool,
                                CommandBatch& batch,
                                PipelineCache& cache,
                                Node* node,
                                float grad_scale);

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
