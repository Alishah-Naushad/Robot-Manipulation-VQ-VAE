import torch
import torch.profiler as profiler
from hq_vae import HierarchicalLFQHVQVAE  # Replace with your actual import

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalLFQHVQVAE(
    feature_dim=256,
    z_dim=64,
    q_dim=32,
    num_z_codes=1024,
    num_q_codes=512,
    hidden_dim=128,
).to(device)

# Create dummy batch
batch_size = 1024
x = torch.randn(batch_size, 256, device=device)

# Warm up GPU
for _ in range(3):
    _ = model(x)

# Profile with detailed info
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        model(x)

# Print results sorted by CUDA time
print("=" * 80)
print("SORTED BY CUDA TIME (GPU bottlenecks)")
print("=" * 80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Print results sorted by CPU time
print("\n" + "=" * 80)
print("SORTED BY CPU TIME (CPU bottlenecks)")
print("=" * 80)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# Export trace for Chrome visualization
prof.export_chrome_trace("hvqvae_trace.json")
print("\nTrace exported to hvqvae_trace.json - open in Chrome at chrome://tracing")
