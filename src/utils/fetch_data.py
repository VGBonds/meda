import os
from datasets import load_dataset
from src.config import config_medgemma_4b_it_nih_cxr

DATASET_ID = "hf-vision/chest-xray-pneumonia"


def download_data_chest_xray_pneumonia(
        cache_dir=config_medgemma_4b_it_nih_cxr.dataset_cache_directory
):
    os.makedirs(cache_dir, exist_ok=True)
    print("Downloading chest xray pneumonia dataset...")
    dataset=load_dataset(config_medgemma_4b_it_nih_cxr.dataset_id, cache_dir=cache_dir)
    print("Finished downloading chest xray pneumonia dataset.")
    print(dataset)


def load_data_chest_xray_pneumonia(
        cache_dir=config_medgemma_4b_it_nih_cxr.dataset_cache_directory
):
    if (not os.path.isdir(cache_dir)) or (not os.listdir(cache_dir)):
        print("Cache directory is missing or empty. Downloading dataset.")
        download_data_chest_xray_pneumonia(cache_dir)

    dataset = load_dataset(path=cache_dir)
    print(dataset)
    return dataset


if __name__ == "__main__":
    # get the project root folder

    cache_directory = config_medgemma_4b_it_nih_cxr.dataset_cache_directory
    _ = load_data_chest_xray_pneumonia(cache_directory)