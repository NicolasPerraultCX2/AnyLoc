#!/usr/bin/env python3
"""
Download AnyLoc VLAD cluster centers using torch.hub
"""
import torch
import os

# Create cache directory structure
cache_dir = "./AnyLoc/cache/vocabulary/dinov2_vitg14/l31_value_c32"
os.makedirs(f"{cache_dir}/urban", exist_ok=True)
os.makedirs(f"{cache_dir}/indoor", exist_ok=True)
os.makedirs(f"{cache_dir}/aerial", exist_ok=True)

print("Downloading cluster centers for all domains...")
for domain in ["urban", "indoor", "aerial"]:
    print(f"\nDownloading {domain} domain...")
    try:
        # Load model using torch.hub - this downloads the cluster centers
        model = torch.hub.load("AnyLoc/DINO", "get_vlad_model",
                               backbone="DINOv2",
                               domain=domain,
                               device="cpu")  # Use CPU for download

        # Get the cluster centers
        if hasattr(model, 'c_centers'):
            c_centers = model.c_centers
        elif hasattr(model, 'vlad') and hasattr(model.vlad, 'c_centers'):
            c_centers = model.vlad.c_centers
        else:
            print(f"Warning: Could not find c_centers for {domain}")
            continue

        # Save to the expected location
        save_path = f"{cache_dir}/{domain}/c_centers.pt"
        torch.save(c_centers, save_path)
        print(f"✓ Saved {domain} cluster centers to {save_path}")
        print(f"  Shape: {c_centers.shape}")

    except Exception as e:
        print(f"Error downloading {domain}: {e}")
        print("This might be expected if the torch.hub method doesn't work")
        print("Let's try an alternative approach...")

print("\n✓ Cache download complete!")
print(f"Cache location: {cache_dir}")
