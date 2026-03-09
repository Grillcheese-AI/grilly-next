#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "grilly/cubemind/types.h"
#include "grilly/cubemind/vsa.h"

namespace grilly {
namespace cubemind {

// ── MultimodalEncoder: Vision-Text VSA Fusion ────────────────────────
//
// Projects continuous image embeddings (e.g., 768D from ViT-Base or
// 1024D from ViT-Large) into the same bipolar VSA space as the text
// encoder, then fuses them via role-filler binding + bundling.
//
// The mathematical fusion is:
//   State_Fused = sgn((Role_Image ⊗ Filler_Image) + State_Text)
//
// Uses the Johnson-Lindenstrauss lemma: a random Gaussian projection
// from R^vit_dim to {-1,+1}^dim preserves cosine similarity as
// Hamming distance. The projection matrix is deterministic (seed 42)
// to ensure reproducibility.
//
// Memory footprint: a fused 10240D vector = 1.25 KB total.

class MultimodalEncoder {
public:
    /// @param dim      VSA hypervector dimension (e.g., 10240)
    /// @param vit_dim  Vision model output dimension (e.g., 768 for ViT-Base)
    MultimodalEncoder(uint32_t dim, uint32_t vit_dim);

    /// Project dense float image features into the VSA bipolar space.
    /// Uses LSH random projection: bipolar[j] = sgn(sum_i(vit[i] * M[i][j]))
    std::vector<int8_t> project_image_features(const float* vit_features) const;

    /// Bind the image to its structural role and fuse with a text bundle.
    /// Returns a single bitpacked vector containing both modalities.
    BitpackedVec fuse_with_text(const std::vector<int8_t>& image_bipolar,
                                 const BitpackedVec& text_bundle) const;

    uint32_t dim() const { return dim_; }
    uint32_t vit_dim() const { return vit_dim_; }

private:
    uint32_t dim_;
    uint32_t vit_dim_;
    std::vector<float> projection_matrix_;  // [vit_dim_ * dim_], row-major
    std::vector<int8_t> role_image_;         // BLAKE3 deterministic image role
};

}  // namespace cubemind
}  // namespace grilly
