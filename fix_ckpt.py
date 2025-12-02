import torch
import sys
import os


def fix_checkpoint(checkpoint_path):
    """
    Remove non-model keys from checkpoint
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Check what's in the model state dict
    if "model" in ckpt:
        model_dict = ckpt["model"]
        print(f"\nOriginal keys in model dict: {len(model_dict)}")

        # Keys to remove
        keys_to_remove = ["vq_vae_enabled", "vqvae_config"]
        removed_keys = []

        for key in keys_to_remove:
            if key in model_dict:
                removed_keys.append(key)
                model_dict.pop(key)
                print(f"  Removed: {key}")

        if removed_keys:
            # Save fixed checkpoint
            output_path = checkpoint_path.replace(".pth", "_fixed.pth")

            # Make sure we don't overwrite
            if os.path.exists(output_path):
                output_path = checkpoint_path.replace(".pth", "_fixed_new.pth")

            torch.save(ckpt, output_path)
            print(f"\n✅ Fixed checkpoint saved to: {output_path}")
            print(f"   Removed {len(removed_keys)} non-model keys")
            print(f"   Remaining keys: {len(model_dict)}")

            # Show some model keys to verify
            print(f"\nSample of remaining keys:")
            for i, key in enumerate(list(model_dict.keys())[:10]):
                print(f"  {i+1}. {key}")
            if len(model_dict) > 10:
                print(f"  ... and {len(model_dict) - 10} more")

            return output_path
        else:
            print("\n✅ No keys to remove - checkpoint is already clean")
            return checkpoint_path
    else:
        print("ERROR: Checkpoint doesn't have 'model' key")
        print(f"Available keys: {list(ckpt.keys())}")
        return None


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python fix_checkpoint.py /path/to/checkpoint.pth")
    #     sys.exit(1)

    # checkpoint_path = sys.argv[1]

    # if not os.path.exists(checkpoint_path):
    #     print(f"ERROR: Checkpoint not found: {checkpoint_path}")
    #     sys.exit(1)

    fixed_path = fix_checkpoint(
        "/home/retrocausal-train/Desktop/lipvq/LipVQ-VAE/expdata/robocasa/fawad_heirarchal_v1/seed_123_ds_human-50/20251129004921/models/model_epoch_1000.pth"
    )

    if fixed_path:
        print(f"\n📝 To use the fixed checkpoint, run:")
        print(f"   python scripts/train.py --config config.json --resume {fixed_path}")
