#include "grilly/cubemind/multimodal_encoder.h"

#include <random>

namespace grilly {
namespace cubemind {

// ── MultimodalEncoder ────────────────────────────────────────────────

MultimodalEncoder::MultimodalEncoder(uint32_t dim, uint32_t vit_dim)
    : dim_(dim), vit_dim_(vit_dim) {

    // 1. Initialize the LSH Gaussian random projection matrix
    // Seed 42 ensures the projection is identical across all runs,
    // matching the TextEncoder/SemanticAssigner convention.
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    projection_matrix_.resize(static_cast<size_t>(vit_dim_) * dim_);
    for (size_t i = 0; i < projection_matrix_.size(); ++i) {
        projection_matrix_[i] = dist(gen);
    }

    // 2. Pre-generate the static structural Role for images using BLAKE3
    // Matches the deterministic generation from vsa.cpp
    role_image_ = blake3Role("role_image", dim_);
}

std::vector<int8_t> MultimodalEncoder::project_image_features(
    const float* vit_features) const {
    std::vector<int8_t> bipolar(dim_);

    // Execute the LSH Random Projection:
    //   bipolar[j] = sgn( sum_i( vit[i] * M[i][j] ) )
    //
    // By the Johnson-Lindenstrauss lemma, this preserves cosine
    // similarity as Hamming distance in the bipolar space.
    for (uint32_t j = 0; j < dim_; ++j) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < vit_dim_; ++i) {
            sum += vit_features[i] *
                   projection_matrix_[static_cast<size_t>(i) * dim_ + j];
        }
        bipolar[j] = (sum > 0.0f) ? 1 : -1;
    }

    return bipolar;
}

BitpackedVec MultimodalEncoder::fuse_with_text(
    const std::vector<int8_t>& image_bipolar,
    const BitpackedVec& text_bundle) const {

    // 1. Role-Filler Binding: Bind the projected image to its structural role
    //    bound = Role_Image ⊗ Filler_Image (element-wise multiply)
    std::vector<int8_t> bound_image =
        vsaBind(image_bipolar.data(), role_image_.data(), dim_);

    // 2. Unpack the text bundle to bipolar int8 for bundling
    std::vector<int8_t> unpacked_text = vsaUnpack(text_bundle);

    // 3. Bundle the image state and text state together
    //    Fused = sgn(bound_image + unpacked_text)
    std::vector<const int8_t*> bundle_ptrs = {
        bound_image.data(), unpacked_text.data()};
    std::vector<int8_t> fused_state = vsaBundle(bundle_ptrs, dim_);

    // 4. Bitpack for GPU Hamming distance search
    return vsaBitpack(fused_state.data(), dim_);
}

}  // namespace cubemind
}  // namespace grilly
