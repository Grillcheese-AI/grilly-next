import torch
import numpy as np
import os
from google.colab import files

# --- Configuration ---
VOCAB_SIZE = 152000
DIM = 10240
SEED = 42

print(f"[INFO] Generating VSA Codebook (Vocab: {VOCAB_SIZE}, Dim: {DIM})...")

# Enforce CUDA to match your H200 training environment
device = torch.device("cuda")
print(f"[INFO] Using device: {device} for PyTorch RNG...")

# 1. Lock the seed
torch.manual_seed(SEED)

# 2. Generate the Holographic Vectors
print("[INFO] Allocating Bipolar Tensor on A100...")
vsa_embeddings = torch.sign(torch.randn((VOCAB_SIZE, DIM), device=device, dtype=torch.float32))
vsa_embeddings[vsa_embeddings == 0] = 1 

# 3. Move to host memory for packing
bipolar_weights = vsa_embeddings.cpu()

# 4. Bit-pack into uint32
print("[INFO] Bit-packing floats into dense uint32 binary...")
binary_weights = (bipolar_weights > 0).to(torch.uint8).numpy()
packed_uint32 = np.packbits(binary_weights, bitorder='little').view(np.uint32)

# 5. Export to binary
filename = "vsa_codebook.bin"
packed_uint32.tofile(filename)

file_size_mb = os.path.getsize(filename) / (1024 * 1024)
print(f"[SUCCESS] Holographic Codebook exported to {filename}!")
print(f"[SUCCESS] Final File Size: {file_size_mb:.2f} MB")

# 6. Trigger automatic download to your local machine
print("[INFO] Downloading to your local machine...")
files.download(filename)