# --------------------------------------------------------------
# NEGATIVE BAG EXTRACTION – IDENTICAL TO Camelyon17-WILDS
# --------------------------------------------------------------
import random
import numpy as np
import torch
import openslide
import cv2
from tqdm.auto import tqdm
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
import os
import wsi.config_wsi as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === hibou-L embedder ===
hibou = AutoModel.from_pretrained(
    "histai/hibou-L",
    trust_remote_code=True
)
hibou.to(DEVICE)
hibou.eval()
processor = AutoImageProcessor.from_pretrained(
        "histai/hibou-L",
        trust_remote_code=True,
    )
# if not os.path.exists(os.path.join(config.embedding_model_cache_directory, "config.json")):
#     os.makedirs(config.embedding_model_cache_directory, exist_ok=True)
#     processor = AutoImageProcessor.from_pretrained(
#         "histai/hibou-L",
#         trust_remote_code=True,
#     )
#     hibou = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
#
#     hibou.save_pretrained(config.embedding_model_cache_directory)
#     processor.save_pretrained(config.embedding_model_cache_directory)
#
#     hibou.eval()
# else:
#     processor = AutoImageProcessor.from_pretrained(
#         config.embedding_model_cache_directory,
#         trust_remote_code=True,
#     )
#     hibou = AutoModel.from_pretrained(
#         config.embedding_model_cache_directory,
#         trust_remote_code=True,
#     )
#     hibou.eval()

# For hibou-L (handles 224 natively)
@torch.no_grad()
def embed_patch(patch_np):  # (224,224,3)
    inputs = processor(images=patch_np, return_tensors="pt").to(DEVICE)
    out = hibou(**inputs)
    return out.last_hidden_state[:, 0].cpu()


def get_tissue_roi_wilds_style(slide, level=2, thumb_size=1024, min_area=500, padding=1000):
    """
    Returns (x1, y1, x2, y2) in LEVEL 0 coordinates.
    """
    # 1. Thumbnail
    thumb = slide.get_thumbnail((thumb_size, thumb_size))
    gray = np.array(thumb.convert("L"))

    # 2. Otsu
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
    # # debug
    # # After get_tissue_roi_wilds_style()
    # thumb = slide.get_thumbnail((1024, 1024))
    # thumb_np = np.array(thumb)
    # x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
    # cv2.rectangle(thumb_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # plt.imshow(thumb_np);
    # plt.title("Tissue ROI");
    # plt.show()

    # 3. Remove small objects
    num_labels, labels = cv2.connectedComponents(mask)
    for i in range(1, num_labels):
        if np.sum(labels == i) < min_area:
            mask[labels == i] = 0

    # 4. Largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 5. Scale to level 1
    scale = slide.level_dimensions[level][0] / thumb_size
    x1 = int(x * scale)
    y1 = int(y * scale)
    x2 = int((x + w) * scale)
    y2 = int((y + h) * scale)

    # 6. Add padding + clamp
    ds_factor = int(slide.level_downsamples[level])
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(slide.level_dimensions[level][0], x2 + padding)
    y2 = min(slide.level_dimensions[level][1], y2 + padding)

    # 7. Convert to level 0
    return (
        x1 * ds_factor,
        y1 * ds_factor,
        x2 * ds_factor,
        y2 * ds_factor
    )

# --------------------------------------------------------------
# WILDS-EXACT: 224×224 @ Level 2
# --------------------------------------------------------------
def prepare_negative_mil_bags_wilds_exact(
    negative_wsi_folder: str | Path,
    negative_mil_cache: str | Path,
    patch_size: int = 224,
    level: int = 2,           # 4x downsample
    stride: int = 224,
    tissue_threshold: float = 0.5,
    max_slides: int = 50,
):
    negative_wsi_folder = Path(negative_wsi_folder)
    negative_mil_cache   = Path(negative_mil_cache)
    negative_mil_cache.parent.mkdir(parents=True, exist_ok=True)

    wsi_files = sorted([f for f in negative_wsi_folder.glob("*.tif") if not f.name.startswith(".")])[:max_slides]
    negative_bags = []

    for wsi_path in tqdm(wsi_files, desc="Negative bags (WILDS-EXACT)"):
        slide = openslide.OpenSlide(str(wsi_path))
        slide_id = f"NEG_{wsi_path.stem}"

        # 1. ROI in level 0
        roi = get_tissue_roi_wilds_style(slide, level=level, padding=1000)

        if roi is None:
            slide.close()
            continue
        x1_l0, y1_l0, x2_l0, y2_l0 = roi

        # 2. Convert to level 2
        ds_factor = int(slide.level_downsamples[level])
        x1_l2 = x1_l0 // ds_factor
        y1_l2 = y1_l0 // ds_factor
        x2_l2 = x2_l0 // ds_factor
        y2_l2 = y2_l0 // ds_factor

        # 3. Grid
        xs = range(x1_l2, x2_l2 - patch_size + 1, stride)
        ys = range(y1_l2, y2_l2 - patch_size + 1, stride)
        grid = [(x, y) for y in ys for x in xs]
        print(f"{slide_id}: {len(grid)} candidate patches")

        # 4. Extract + filter
        feats, coords_l0 = [], []
        for x_l2, y_l2 in grid:
            region = slide.read_region((x_l2 * ds_factor, y_l2 * ds_factor), level, (patch_size, patch_size))
            patch = np.array(region.convert("RGB"))

            # Tissue % (fast)
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            if np.mean(gray < 220) < tissue_threshold:
                continue

            emb = embed_patch(patch)  # hibou-L or ResNet
            feats.append(emb)
            coords_l0.append((x_l2 * ds_factor, y_l2 * ds_factor))

        if len(feats) == 0:
            slide.close()
            continue

        feats = torch.stack(feats)
        coords = torch.tensor(coords_l0, dtype=torch.long)

        negative_bags.append({
            "features": feats,
            "coordinates": coords,
            "label": torch.zeros(len(feats), dtype=torch.float),
            "slide_label": torch.tensor(0.0),
            "soft_label": torch.tensor(0.0),
            "slide_id": slide_id
        })

        slide.close()
        if len(negative_bags) >= max_slides:
            break

    torch.save(negative_bags, negative_mil_cache)
    print(f"Saved {len(negative_bags)} bags to {negative_mil_cache}")
    return negative_bags


def get_min_max_coordinates(mil_data):
    all_coords = torch.cat([ex['coordinates'] for ex in mil_data], dim=0)  # (total_patches, 2)
    min_coords = torch.min(all_coords, dim=0).values  # (2,)
    max_coords = torch.max(all_coords, dim=0).values  # (2,)
    return min_coords, max_coords


if __name__ == "__main__":
    wsi_path = os.path.join(config.negative_wsi_folder, "patient_002_node_0.tif")
    slide = openslide.OpenSlide(str(wsi_path))
    print("Vendor:", slide.properties.get("openslide.vendor"))  # should be "hamamatsu"
    print("Dimensions:", slide.dimensions)

    negative_bags = prepare_negative_mil_bags_wilds_exact(
        negative_wsi_folder = config.negative_wsi_folder,
        negative_mil_cache = config.negative_mil_cache,
        #patch_size = config.patch_size,
        #level=2,
        #stride = config.stride,
        #max_slides = 50,
    )

    get_min_max_coordinates(negative_bags)
    print(f"Negative bags coordinate range: {get_min_max_coordinates(negative_bags)}")
    print(f"Created {len(negative_bags)} negative bags.")
    print(f"Saving them as {config.negative_mil_cache}")