import time
import torch
from hq_vae import HierarchicalLFQHVQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalLFQHVQVAE(
    feature_dim=256,
    z_dim=64,
    q_dim=32,
    num_z_codes=1024,
    num_q_codes=512,
    hidden_dim=128,
).to(device)
model.train()

x = torch.randn(1024, 256, device="cuda")

# Warmup
for _ in range(3):
    _ = model(x)

# Test 1: Forward only (no backward)
torch.cuda.synchronize()
start = time.time()
for _ in range(20):
    with torch.no_grad():
        output = model(x)
torch.cuda.synchronize()
print(
    f"Forward only (no backward): {(time.time() - start) / 20 * 1000:.2f}ms per batch"
)

# Test 2: Forward + backward
torch.cuda.synchronize()
start = time.time()
for _ in range(20):
    output = model(x)
    output["loss"].backward()
torch.cuda.synchronize()
print(f"Forward + backward: {(time.time() - start) / 20 * 1000:.2f}ms per batch")

# Test 3: Forward + backward + optimizer step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
torch.cuda.synchronize()
start = time.time()
for _ in range(20):
    optimizer.zero_grad()
    output = model(x)
    output["loss"].backward()
    optimizer.step()
torch.cuda.synchronize()
print(
    f"Forward + backward + optimizer: {(time.time() - start) / 20 * 1000:.2f}ms per batch"
)
