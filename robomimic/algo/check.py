import torch

ckpt_path = "/home/retrocausal-train/Desktop/lipvq/LipVQ-VAE/expdata/robocasa/fawad_heirarchal_v1/512_128_lipvq_gmm/20251206021655/models/model_epoch_300.pth"

state = torch.load(ckpt_path, map_location="cpu")
stae = state["model"]

# list all keys
for k, v in state.items():
    print(k)
    print(v)
