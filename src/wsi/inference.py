# --------------------------------------------------------------
# HEATMAPS: 3 TEST SLIDES (POS + NEG)
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
import os
from matplotlib.colors import LinearSegmentedColormap
#from torchmil.models.abmil import ABMIL
from wsi.abmil import ABMIL, PR_ABMIL
import wsi.config_wsi as config
from wsi.make_bags_dataset import make_bags_dataset
from wsi.visualize_top_patches import visualize_top_patches


@torch.no_grad()
def predict_and_heatmap(model, batch, slide_path):
    model.eval()
    feats = batch["features"].to(config.DEVICE)
    mask = batch["mask"].to(config.DEVICE)
    # logits, att_raw = model(feats, return_att=True)
    logits, att_raw = model(feats, mask=mask, return_attention=True)

    prob = torch.sigmoid(logits).item()

    # # Normalize attention: only for stock torchmil ABMIL
    # att_raw = att_raw.squeeze()
    # att_masked = att_raw.masked_fill(mask.squeeze() == 0, float('-inf'))
    # att_norm = torch.softmax(att_masked, dim=0).cpu().numpy()
    att_norm = att_raw.squeeze().cpu().numpy()

    # Valid patches
    coords = batch["coordinates"][0].numpy()
    valid = coords[:, 0] != -1
    coords = coords[valid]
    att_norm = att_norm[valid]

    # === NORM ATTENTION TO [0,1] for visualization purposes===
    att_min = att_norm.min()
    att_max = att_norm.max()
    att_norm = (att_norm - att_min) / (att_max - att_min + 1e-8)

    # === THUMBNAIL + SCALING ===
    slide = openslide.OpenSlide(slide_path)

    # Level 0 dimensions
    level0_w, level0_h = slide.dimensions

    # Thumbnail
    thumb_level = int(slide.level_count - 1)
    thumb = slide.read_region((0, 0), thumb_level, slide.level_dimensions[thumb_level])
    thumb = np.array(thumb.convert("RGB"))
    thumb_w, thumb_h = thumb.shape[1], thumb.shape[0]

    # save the thumbnail
    output_dir = config.image_output_directory
    os.makedirs(output_dir, exist_ok=True)
    slide_id = batch["slide_ids"][0]
    thumb_save_path = os.path.join(output_dir, f"{slide_id}_thumbnail.png")
    plt.imsave(thumb_save_path, thumb)
    print(f"Saved thumbnail: {thumb_save_path}")

    # Scale: level 0 → thumbnail
    scale_x = thumb_w / level0_w
    scale_y = thumb_h / level0_h

    # === PLOT ===
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(thumb)
    sc = ax.scatter(
        coords[:, 0] * scale_x, coords[:, 1] * scale_y,
        c=att_norm, cmap="Reds", s=20, alpha=0.8, vmin=0, vmax=1
    )
    plt.colorbar(sc, label="Attention (rescaled softmax: 1->0)", shrink=0.6)
    ax.set_title(f"{batch['slide_ids'][0]}\nPred: {prob:.3f} → {'CANCER' if prob > 0.5 else 'NORMAL'}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Save figure
    output_dir = config.image_output_directory
    os.makedirs(output_dir, exist_ok=True)
    slide_id = batch["slide_ids"][0]
    save_path = os.path.join(output_dir, f"{slide_id}_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap: {save_path}")

    # visualize top patches
    visualize_top_patches(att_norm=att_norm,
                          prob=prob,
                          coords=coords,
                          slide=slide,
                          slide_id=slide_id,
                          patch_size=config.patch_size,
                          level=config.level,
                          save_dir=config.image_output_directory,
                          prefix="TEST")



if __name__ == "__main__":
    # === LOAD MODEL & DATA ===

    train_loader, val_loader, test_loader = make_bags_dataset()

    criterion = torch.nn.BCEWithLogitsLoss()
    # model = ABMIL(in_shape=(config.INPUT_DIM,), criterion=criterion).to(config.DEVICE)
    # model = ABMIL(input_size=config.INPUT_DIM,
    #               hidden_dim=512,
    #               dropout=config.DROPOUT_RATE,
    #               dropout_attn=config.DROPOUT_RATE_ATTN).to(config.DEVICE)

    model = PR_ABMIL(
        input_size=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT_RATE,
        dropout_attn=config.DROPOUT_RATE_ATTN,
        patch_dropout=0.,  # 10% dropout
        permute=False
    ).to(config.DEVICE)

    model.load_state_dict(torch.load(config.trained_model_file, map_location=config.DEVICE))

    # === PICK 3 TEST SLIDES ===
    test_iter = iter(val_loader)
    examples = [next(test_iter) for _ in range(2)]

    for batch in examples:
        for idx in range(len(batch["slide_ids"])):
            slide_id = batch["slide_ids"][idx]
            if "NEG_" in slide_id:
                wsi_path = os.path.join(config.negative_wsi_folder,f"{slide_id.replace('NEG_', '')}.tif")
                continue
            else:
                wsi_path = os.path.join(config.positive_wsi_folder,f"{slide_id.replace('POS_', '')}.tif")

            example={
                "features": batch["features"][idx:idx+1],
                "mask": batch["mask"][idx:idx+1],
                "labels": batch["labels"][idx:idx+1],
                "coordinates": [batch["coordinates"][idx]],
                "slide_ids": [batch["slide_ids"][idx]]
            }

            predict_and_heatmap(model, example, wsi_path)