import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

# --- Configuration ---
DIM = 10240
BATCH_SIZE = 1024  # Bumped up massively since 1x H200 has 141GB VRAM
LEARNING_RATE = 1e-4
# Qwen2.5 Vocab size is 151,936. We pad to 152,000 for safety.
VOCAB_SIZE = 152000 

class IntegerTrajectoryDataset(IterableDataset):
    def __init__(self, jsonl_filepath):
        self.filepath = jsonl_filepath

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Cannot find {self.filepath}")
            
        with open(self.filepath, 'r') as f:
            for i, line in enumerate(f):
                # Shard the file reading across CPU workers safely
                if i % num_workers == worker_id:
                    try:
                        data = json.loads(line)
                        trajectory = data["tokens"]
                        
                        # Yield token transitions as integer pairs
                        for j in range(len(trajectory) - 1):
                            yield trajectory[j], trajectory[j+1]
                    except (json.JSONDecodeError, KeyError):
                        continue

class StabilizedHypernetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.network(x)

def export_to_vulkan_binary(model_weights, filename="cubemind_student.bin"):
    print("\n[INFO] Binarizing weights for Vulkan Engine...")
    weights = model_weights.cpu()
    bipolar_weights = torch.sign(weights)
    bipolar_weights[bipolar_weights == 0] = 1  # Prevent exact zeros
    
    # Pack booleans into uint32 for extreme C++ efficiency
    binary_weights = (bipolar_weights > 0).to(torch.uint8).numpy()
    packed_uint32 = np.packbits(binary_weights, bitorder='little').view(np.uint32)
    packed_uint32.tofile(filename)
    
    print(f"[SUCCESS] Exported bare-metal Vulkan payload to {filename}")

def main():
    print(f"Initializing Single H200 Distillation Pipeline...")
    device = torch.device("cuda:0")

    # 1. Initialize the Frozen VSA Table directly in VRAM (~6.2 GB)
    print("Generating deterministic VSA Hypervectors...")
    # Seeded to ensure the same token ID ALWAYS gets the exact same hypervector
    torch.manual_seed(42) 
    vsa_embeddings = torch.sign(torch.randn((VOCAB_SIZE, DIM), device=device, dtype=torch.float32))
    vsa_embeddings[vsa_embeddings == 0] = 1 # Ensure pure bipolar space (-1, 1)

    # 2. Initialize Model & Optimizer
    model = StabilizedHypernetwork(DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()
    
    # 3. Initialize High-Speed DataLoader
    dataset = IntegerTrajectoryDataset("teacher_golden_trajectories.jsonl")
    # Using 8 CPU workers to saturate the GPU with data
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True)
    
    model.train()
    print("Commencing Hypernetwork Training...")
        
    for step, (curr_ids, targ_ids) in enumerate(dataloader):
        # Move integer IDs to GPU
        curr_ids = curr_ids.to(device, non_blocking=True)
        targ_ids = targ_ids.to(device, non_blocking=True)
        
        # Hardware-accelerated dictionary lookup
        curr_state = vsa_embeddings[curr_ids]
        targ_state = vsa_embeddings[targ_ids]
        
        optimizer.zero_grad(set_to_none=True)
        
        # Ground truth transformation: Element-wise multiply for bipolar VSA binding
        ground_truth_transform = curr_state * targ_state
        predicted_transform = model(curr_state)
        
        loss = mse_loss(predicted_transform, ground_truth_transform)
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step:06d} | Geometric MSE Loss: {loss.item():.5f}")

    # Training complete, trigger the Vulkan payload export
    export_to_vulkan_binary(model.network[-2].weight.data)

if __name__ == "__main__":
    main()