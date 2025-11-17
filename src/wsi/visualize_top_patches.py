# --------------------------------------------------------------
# TOP N ATTENTION PATCHES (5x3 grid)
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
import os
from pathlib import Path
import wsi.config_wsi as config


def visualize_top_patches(
    att_norm, prob, coords, slide, slide_id,
    top_k=25, grid_cols=5, patch_size=config.patch_size, level=config.level,
    save_dir=config.image_output_directory, prefix=""
):
    """
    Plots the top_k patches with highest attention.
    Saves as PNG.
    """
    if top_k > 25:
        print("Warning: top_k > 25 may lead to cluttered plots.")
        print("Reducing to top_k = 25.")
        top_k = 25


    # Top-k
    top_idx = np.argsort(att_norm)[-top_k:][::-1]
    top_att = att_norm[top_idx]
    top_coords = coords[top_idx]

    # get downsample factor
    ds_factor = int(slide.level_downsamples[level])

    # Extract patches
    patches = []
    for x_l0, y_l0 in top_coords:
        x_l2 = x_l0 // ds_factor
        y_l2 = y_l0 // ds_factor
        region = slide.read_region((x_l2 * ds_factor, y_l2 * ds_factor), level, (patch_size, patch_size))
        patch = np.array(region.convert("RGB"))
        patches.append(patch)

    # Plot
    grid_rows = (top_k + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3*grid_cols, 3*grid_rows))
    axes = axes.flatten() if grid_rows > 1 else [axes]

    for i in range(top_k):
        ax = axes[i]
        ax.imshow(patches[i])
        ax.set_title(f"Att: {top_att[i]:.3f}", fontsize=10)
        ax.axis("off")
    for i in range(top_k, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"{slide_id} | Pred: {prob:.3f} → {'CANCER' if prob > 0.5 else 'NORMAL'}", fontsize=14)
    plt.tight_layout()

    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{prefix}_{slide_id}_top{top_k}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()

    slide.close()

def visualize_top_patches_(
    model, batch, slide_path,
    top_k=25, grid_cols=5, patch_size=224, level=2,
    save_dir="../results/top_patches", prefix=""
):
    """
    Plots the top_k patches with highest attention.
    Saves as PNG.
    """
    if top_k > 25:
        print("Warning: top_k > 25 may lead to cluttered plots.")
        print("Reducing to top_k = 25.")
        top_k = 25
    model.eval()
    feats = batch["features"].to(config.DEVICE)
    mask = batch["mask"].to(config.DEVICE)
    logits, att_raw = model(feats, return_att=True)
    prob = torch.sigmoid(logits).item()

    # Normalize attention
    att_raw = att_raw.squeeze()
    att_masked = att_raw.masked_fill(mask.squeeze() == 0, float('-inf'))
    att_norm = torch.softmax(att_masked, dim=0).cpu().numpy()

    # Valid patches
    coords = batch["coordinates"][0].numpy()
    valid = coords[:, 0] != -1
    coords = coords[valid]
    att_norm = att_norm[valid]

    # Top-k
    top_idx = np.argsort(att_norm)[-top_k:][::-1]
    top_att = att_norm[top_idx]
    top_coords = coords[top_idx]

    # Open slide
    slide = openslide.OpenSlide(slide_path)
    ds_factor = int(slide.level_downsamples[level])

    # Extract patches
    patches = []
    for x_l0, y_l0 in top_coords:
        x_l2 = x_l0 // ds_factor
        y_l2 = y_l0 // ds_factor
        region = slide.read_region((x_l2 * ds_factor, y_l2 * ds_factor), level, (patch_size, patch_size))
        patch = np.array(region.convert("RGB"))
        patches.append(patch)

    # Plot
    grid_rows = (top_k + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3*grid_cols, 3*grid_rows))
    axes = axes.flatten() if grid_rows > 1 else [axes]

    for i in range(top_k):
        ax = axes[i]
        ax.imshow(patches[i])
        ax.set_title(f"Att: {top_att[i]:.3f}", fontsize=10)
        ax.axis("off")
    for i in range(top_k, len(axes)):
        axes[i].axis("off")

    slide_id = batch["slide_ids"][0]
    plt.suptitle(f"{slide_id} | Pred: {prob:.3f} → {'CANCER' if prob > 0.5 else 'NORMAL'}", fontsize=14)
    plt.tight_layout()

    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{prefix}_{slide_id}_top{top_k}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()

    slide.close()