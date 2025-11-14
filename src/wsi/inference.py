# --------------------------------------------------------------
# HEATMAPS: 3 TEST SLIDES (POS + NEG)
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from matplotlib.colors import LinearSegmentedColormap
import wsi.config_wsi as config
from wsi.make_bags_dataset import make_bags_dataset
import os


@torch.no_grad()
def predict_and_heatmap(model, batch, slide_path):
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

    # === THUMBNAIL + SCALING ===
    slide = openslide.OpenSlide(slide_path)

    # Level 0 dimensions
    level0_w, level0_h = slide.dimensions

    # Thumbnail
    thumb_level = int(slide.level_count - 1)
    thumb = slide.read_region((0, 0), thumb_level, slide.level_dimensions[thumb_level])
    thumb = np.array(thumb.convert("RGB"))
    thumb_w, thumb_h = thumb.shape[1], thumb.shape[0]

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
    plt.colorbar(sc, label="Attention (softmax)", shrink=0.6)
    ax.set_title(f"{batch['slide_ids'][0]}\nPred: {prob:.3f} → {'CANCER' if prob > 0.5 else 'NORMAL'}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # === LOAD MODEL & DATA ===

    train_loader, val_loader, test_loader = make_bags_dataset()
    from torchmil.models import ABMIL
    criterion = torch.nn.BCEWithLogitsLoss()
    model = ABMIL(in_shape=(config.INPUT_DIM,), criterion=criterion).to(config.DEVICE)
    model.load_state_dict(torch.load(config.trained_model_file, map_location=config.DEVICE))

    # === PICK 3 TEST SLIDES ===
    test_iter = iter(val_loader)
    examples = [next(test_iter) for _ in range(1)]

    for batch in examples:
        for idx in range(len(batch["slide_ids"])):
            slide_id = batch["slide_ids"][idx]
            if "NEG_" in slide_id:
                wsi_path = os.path.join(config.negative_wsi_folder,f"{slide_id.replace('NEG_', '')}.tif")
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