# --------------------------------------------------------------
# PATCH-LEVEL EVALUATION: Attention vs. WILDS Labels
# --------------------------------------------------------------
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import pandas as pd
import wsi.config_wsi as config

def evaluate_patch_auc(model, mil_bag, wilds_patches_df):
    """
    wilds_patches_df: columns = [slide_id, x_coord, y_coord, label]
    Only for positive slides in mil_bag
    """
    model.eval()
    feats = mil_bag["features"].to(config.DEVICE)
    mask = mil_bag["mask"].to(config.DEVICE)
    coords = mil_bag["coordinates"].numpy()

    _, att_raw = model(feats, mask=mask, return_att=True)
    att_raw = att_raw.squeeze()
    att_masked = att_raw.masked_fill(mask.squeeze() == 0, float('-inf'))
    att_norm = torch.softmax(att_masked, dim=0).cpu().numpy()

    # Map WILDS patches to your 224×224 patches
    ds_factor = 4  # level 2
    patch_size = 224

    pred_labels = []
    true_labels = []

    for _, row in wilds_patches_df.iterrows():
        if row["label"] == 0:
            continue  # skip negatives

        x_w, y_w = row["x_coord"], row["y_coord"]
        # WILDS coords are in level 0
        x_l2 = x_w // ds_factor
        y_l2 = y_w // ds_factor

        # Find closest 224×224 patch
        dists = np.sum((coords - np.array([x_l2, y_l2])) ** 2, axis=1)
        idx = np.argmin(dists)

        pred_labels.append(att_norm[idx])
        true_labels.append(1)  # positive

    return roc_auc_score(true_labels, pred_labels)