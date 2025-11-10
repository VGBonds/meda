import wsi.config_wsi as config



def make_bags(positive_bags_path=config.positive_mil_cache, negative_bags_path=config.negative_mil_cache):
    pass






# Merge
mil_data_balanced = mil_data + negative_bags
print(f"Balanced dataset: {len(mil_data_balanced)} bags (50 pos + {len(negative_bags)} neg)")


# Update SlideMILDataset to include slide_label/soft_label
class SlideMILDataset(torch.utils.data.Dataset):
    def __len__(self): return len(mil_data_balanced)

    def __getitem__(self, idx):
        item = mil_data_balanced[idx]
        return {
            "features": item["features"],
            "label": item["slide_label"],  # Use slide-level for MIL target
            "slide_id": item["slide_id"],
            "coords": item["coords"]
        }


full_ds = SlideMILDataset()

# Slide-level split (80/10/10, no leakage)
random.seed(42)
slide_ids = [item["slide_id"] for item in mil_data_balanced]
random.shuffle(slide_ids)

n = len(slide_ids)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_ids = set(slide_ids[:train_end])
val_ids = set(slide_ids[train_end:val_end])
test_ids = set(slide_ids[val_end:])

train_idx = [i for i, item in enumerate(mil_data_balanced) if item["slide_id"] in train_ids]
val_idx = [i for i, item in enumerate(mil_data_balanced) if item["slide_id"] in val_ids]
test_idx = [i for i, item in enumerate(mil_data_balanced) if item["slide_id"] in test_ids]

train_ds = Subset(full_ds, train_idx)
val_ds = Subset(full_ds, val_idx)
test_ds = Subset(full_ds, test_idx)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

print(f"Split: train={len(train_ds)} slides | val={len(val_ds)} | test={len(test_ds)}")
print("Ready for ABMIL training (Cells 6-9 unchanged)!")





# Your existing embed_patch (from Cell 3) and device
# ...

NEG_DIR = Path("../data/camelyon17_test_negative")
PATCH_SIZE = 96
N_PATCHES_PER_SLIDE = 4000  # Matches your positive bags

# Load positives
mil_data = torch.load("../data/camelyon17_mil_bags.pt", map_location="cpu")
print(f"Loaded {len(mil_data)} positive bags.")

# List of negative slide files (update with your downloads)
negative_files = list(NEG_DIR.glob("*.tif"))[:50]  # Or filter from CSV
if len(negative_files) < 50:
    print(f"Warning: Only {len(negative_files)} negatives found. Adjust as needed.")

negative_bags = []
random.seed(42)  # Reproducible sampling

for wsi_path in tqdm(negative_files, desc="Extracting negative bags"):
    slide = openslide.OpenSlide(str(wsi_path))
    slide_id = f"NEG_{wsi_path.stem}"

    # Random patch locations (avoid borders)
    w, h = slide.dimensions
    patch_coords = []
    patch_embs = []

    for _ in range(N_PATCHES_PER_SLIDE):
        x = random.randint(0, w - PATCH_SIZE)
        y = random.randint(0, h - PATCH_SIZE)
        region = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
        patch_rgb = np.array(region.convert("RGB"))  # (96,96,3)
        emb = embed_patch(patch_rgb)  # Your function
        patch_embs.append(emb)
        patch_coords.append((x, y))

    feats = torch.stack(patch_embs)  # (4000, 512)
    coords = torch.tensor(patch_coords, dtype=torch.long)  # (4000, 2)

    negative_bags.append({
        "features": feats,
        "coords": coords,
        "label": torch.zeros(N_PATCHES_PER_SLIDE, dtype=torch.float),  # All patch labels 0
        "slide_label": torch.tensor(0.0),
        "soft_label": torch.tensor(0.0),
        "slide_id": slide_id
    })

    slide.close()  # Free memory
    if len(negative_bags) >= 50:
        break

print(f"Created {len(negative_bags)} negative bags.")