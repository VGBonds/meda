# --------------------------------------------------------------
# MERGE POSITIVE & NEGATIVE BAGS + SLIDE-LEVEL SPLIT
# --------------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import random
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

import wsi.config_wsi as config

class MILBagDataset(Dataset):
    def __init__(self, mil_data, indices):
        self.mil_data = mil_data
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        bag = self.mil_data[real_idx]
        return {
            "features": bag["features"],            # (global_max_N, D)
            "mask": bag["mask"],                    # (global_max_N,)
            "label": bag["label"],                  # scalar
            "slide_id": bag["slide_id"],
            "coordinates": bag["coordinates"]       # (global_max_N, 2)
        }


# Collate function for DataLoader in order to transform list of dicts into dict of batched tensors
def mil_collate_fn(batch):
    features = torch.stack([item["features"] for item in batch])  # (B, MAX_N, D)
    masks = torch.stack([item["mask"] for item in batch])        # (B, MAX_N)
    labels = torch.stack([item["label"] for item in batch])      # (B,)
    slide_ids = [item["slide_id"] for item in batch]
    coordinates = [item["coordinates"] for item in batch]
    return {
        "features": features,
        "mask": masks,
        "labels": labels,
        "slide_ids": slide_ids,
        "coordinates": coordinates
    }


def make_bags_dataset(positive_bags_path=config.positive_mil_cache, negative_bags_path=config.negative_mil_cache):
    # === 1. LOAD BAGS ===
    print("Loading positive and negative bags...")
    pos_bags = torch.load(positive_bags_path, map_location="cpu")
    neg_bags = torch.load(negative_bags_path, map_location="cpu")

    print(f"Positives: {len(pos_bags)} bags, {sum(b['features'].shape[0] for b in pos_bags):,} patches")
    print(f"Negatives: {len(neg_bags)} bags, {sum(b['features'].shape[0] for b in neg_bags):,} patches")

    # === 2. MERGE ===
    mil_data = pos_bags + neg_bags
    # Remove a superfluous tensor dimension. todo: fix this in bag creation
    for bag in mil_data:
        bag["features"] = torch.squeeze(bag["features"], dim=1)
    print(f"Merged: {len(mil_data)} total bags")

    # === 2.5 PRE-PAD ===
    mil_data = pad_mil_data(mil_data)

    # === 3. SLIDE-LEVEL SPLIT (NO LEAKAGE) ===
    seed = config.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract slide IDs
    slide_ids = [bag["slide_id"] for bag in mil_data]
    random.shuffle(slide_ids)

    n = len(slide_ids)
    train_end = int(config.train_ratio * n)
    val_end = int((config.train_ratio + config.val_ratio) * n)

    train_ids = set(slide_ids[:train_end])
    val_ids = set(slide_ids[train_end:val_end])
    test_ids = set(slide_ids[val_end:])

    # Map to indices
    train_idx = [i for i, bag in enumerate(mil_data) if bag["slide_id"] in train_ids]
    val_idx = [i for i, bag in enumerate(mil_data) if bag["slide_id"] in val_ids]
    test_idx = [i for i, bag in enumerate(mil_data) if bag["slide_id"] in test_ids]

    print(f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} slides")

    # === 5. CREATE DATASETS ===
    train_dataset = MILBagDataset(mil_data, train_idx)
    val_dataset = MILBagDataset(mil_data, val_idx)
    test_dataset = MILBagDataset(mil_data, test_idx)

    # === 6. DATALOADERS (batch_size=1 → one bag per batch).
    # Use collation function to map batch of dicts to dict of tensors ===
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=mil_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=mil_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False,
                             num_workers=2, pin_memory=True, collate_fn=mil_collate_fn)

    print("DataLoaders ready. Training can begin.")
    return train_loader, val_loader, test_loader


def pad_mil_data(mil_data):
    # --------------------------------------------------------------
    # PRE-PAD ALL BAGS TO GLOBAL MAX (ONE-TIME)
    # --------------------------------------------------------------
    from torch.nn.utils.rnn import pad_sequence

    # === FIND GLOBAL MAX N ===
    all_N = [bag["features"].shape[0] for bag in mil_data]
    global_max_N = max(all_N)
    print(f"Global max patches per bag: {global_max_N}")

    # === PRE-PAD ALL BAGS ===
    padded_mil_data = []
    for bag in mil_data:
        N = bag["features"].shape[0]
        if N < global_max_N:
            # Pad features
            pad_len = global_max_N - N
            pad_features = torch.zeros(pad_len, bag["features"].shape[1])
            features_padded = torch.cat([bag["features"], pad_features], dim=0)

            # Pad coordinates (with -1 to mark invalid)
            pad_coords = torch.full((pad_len, 2), -1, dtype=torch.long)
            coords_padded = torch.cat([bag["coordinates"], pad_coords], dim=0)

            # Create mask: 1 for real, 0 for pad
            mask = torch.cat([torch.ones(N), torch.zeros(pad_len)])
        else:
            features_padded = bag["features"]
            coords_padded = bag["coordinates"]
            mask = torch.ones(N)

        padded_mil_data.append({
            "features": features_padded,  # (global_max_N, D)
            "mask": mask,  # (global_max_N,)
            "label": bag["slide_label"],
            "slide_id": bag["slide_id"],
            "coordinates": coords_padded  # (global_max_N, 2), -1 for pads
        })

    print(f"Pre-padded {len(padded_mil_data)} bags to size {global_max_N}")
    return padded_mil_data  # overwrite



if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_bags_dataset()
    print("Sample batch from train_loader:")
    batch = next(iter(train_loader))
    print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
