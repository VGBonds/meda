from datasets import load_dataset, DatasetDict, concatenate_datasets
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

import wsi.config_wsi as config
# from wsi.embed_patch import embed_patch
from wsi.wsi_utils import load_mil_data, save_mil_data

from transformers import AutoImageProcessor, AutoModel

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


# For hibou-L (handles 224 natively)
@torch.no_grad()
def embed_patch(patch_np):  # (224,224,3)
    inputs = processor(images=patch_np, return_tensors="pt").to(DEVICE)
    out = hibou(**inputs)
    return out.last_hidden_state[:, 0].cpu()

def load_from_hf(dataset_id):
    ds_full = load_dataset(dataset_id,
                           split=['train', 'validation', 'test'],
                           cache_dir=config.cache_directory)

    print(f"Loaded: train={len(ds_full[0])}, val={len(ds_full[1])}, test={len(ds_full[2])} patches (streaming)")
    print("Merge the splits. The splits will be done bag-wise on a later stage")
    merged_ds = concatenate_datasets([ds_full[0], ds_full[1], ds_full[2]])
    print(f"merged the data. Total length:{len(merged_ds)}")
    print(merged_ds)
    return merged_ds

def add_wsi_label(ds):
    # get the slide label: slide label is positive when at least one patch is labelled as positive
    slide_to_label = {}
    for ex in tqdm(ds, desc="calculating slide to label mapping"):
        sid = ex["slide"]
        label = ex["label"]
        slide_to_label[sid] = slide_to_label.get(sid, 0) or label  # OR → any 1 wins

    # add slide label to dataset
    def add_slide_label(example):
        example["slide_label"] = slide_to_label[example["slide"]]
        return example
    ds = ds.map(add_slide_label)
    return ds




def create_mil_bags(ds):
    slide_bags = defaultdict(lambda: {
        "features": [], "coordinates": [], "label": [], "id": None, "slide_label": None
    })

    print("Building MIL bags from train / val / test splits...")

    for example in tqdm(ds, desc="merged dataset patches"):
        slide_id = example["slide"]
        label = example["label"]
        slide_label = example["slide_label"]
        patch_rgba = np.array(example["image"])  # (96,96,4)
        patch = patch_rgba[..., :3]  # → (96,96,3)
        coord = (example["x_coord"], example["y_coord"])

        # Embed
        emb = embed_patch(patch)

        # Store
        slide_bags[slide_id]["features"].append(emb)
        slide_bags[slide_id]["coordinates"].append(coord)
        slide_bags[slide_id]["label"].append(label)
        slide_bags[slide_id]["slide_label"] = slide_label
        slide_bags[slide_id]["id"] = slide_id

    # Stack into final MIL dataset
    mil_data = []
    for sid, bag in slide_bags.items():
        if len(bag["features"]) == 0:
            continue
        feats = torch.stack(bag["features"])
        coordinates = torch.tensor(bag["coordinates"], dtype=torch.long)
        label = torch.tensor(bag["label"], dtype=torch.float)
        slide_label = torch.tensor(bag["slide_label"], dtype=torch.float)
        mil_data.append({
            "features": feats,
            "coordinates": coordinates,
            "label": label,
            "slide_label": slide_label,
            "slide_id": sid
        })

    print(f"\nCreated {len(mil_data)} slide-level bags.")
    print(
        f"Example: {mil_data[0]['slide_id']} → {mil_data[0]['features'].shape[0]} patches, slide label={mil_data[0]['slide_id'].item()}")

    return mil_data


def add_soft_label(mil_data):
    for ex in mil_data:
        ex["soft_label"] = torch.mean(ex['label'])

    return mil_data


def prepare_positive_wsi(dataset_id=config.patch_dataset_id):

    # check if positive mil data are already available
    if os.path.exists(config.positive_mil_cache):
        mil_bags = load_mil_data(config.positive_mil_cache)
    else:
        ds_full = load_from_hf(dataset_id)
        ds_full = add_wsi_label(ds_full)
        # Build slide - level MIL bags
        mil_bags = create_mil_bags(ds_full)
        # add soft label
        mil_bags = add_soft_label(mil_bags)
        # serialize data
        torch.save(mil_bags, config.positive_mil_cache)
        print(f"Saved {len(mil_bags)} bags to {config.positive_mil_cache}")
        # save_mil_data(cache_path=config.positive_mil_cache, mil_data=mil_bags)

    return mil_bags
def get_min_max_coordinates(mil_data):
    all_coords = torch.cat([ex['coords'] for ex in mil_data], dim=0)  # (total_patches, 2)
    min_coords = torch.min(all_coords, dim=0).values  # (2,)
    max_coords = torch.max(all_coords, dim=0).values  # (2,)
    per_example_mins = [torch.min(ex['coords'], dim=0).values for ex in mil_data]
    per_example_maxs = [torch.max(ex['coords'], dim=0).values for ex in mil_data]
    return min_coords, max_coords, per_example_mins, per_example_maxs
if __name__ == "__main__":
    mil_bags = prepare_positive_wsi()
    print("Positive WSI MIL bags are ready!")
    min_coords, max_coords, per_example_mins, per_example_maxs = get_min_max_coordinates(mil_bags)
    print(min_coords, max_coords)
    print(per_example_mins)
    print(per_example_maxs)

