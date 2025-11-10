import torch
from pathlib import Path

def load_mil_data(cache_path):
    from pathlib import Path
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading MIL bags from {cache_path} …")
        mil_data = torch.load(cache_path, map_location="cpu")
        print(f"Loaded {len(mil_data)} slide bags.")
    else:
        raise FileNotFoundError(
            f"{cache_path} not found. Run the bag-building loop first.")
    return mil_data


def save_mil_data(cache_path, mil_data):
    cache_path = Path(cache_path)
    print(f"Saving {len(mil_data)} slide bags to {cache_path} …")
    torch.save(mil_data, cache_path)
    print("Save complete!")
