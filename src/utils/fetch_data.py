import os
import requests
import zipfile

from datasets import load_dataset, DatasetDict
import config.config_medgemma_4b_it_nih_cxr as config_medgemma_4b_it_nih_cxr
import config.config_medgemma_4b_it_nct_crc_he as config_medgemma_4b_it_nct_crc_he
import urllib.request


def download_zip_from_url(dataset_url: str, dataset_cache_path: str) -> None:
    if not os.path.exists(dataset_cache_path):
        os.makedirs(os.path.dirname(dataset_cache_path), exist_ok=True)
        urllib.request.urlretrieve(dataset_url, filename=dataset_cache_path)
    with zipfile.ZipFile(dataset_cache_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(dataset_cache_path))
    print(f"ZIP file unpacked successfully at: {os.path.dirname(dataset_cache_path)}")


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

def load_data_nct_crc_he(test_size, val_size, max_examples):
    cache_dir = config_medgemma_4b_it_nct_crc_he.dataset_cache_directory_train
    if (not os.path.isdir(cache_dir)) or (not os.listdir(cache_dir)):
        print("Cache directory is missing or empty. Downloading dataset.")
        url = config_medgemma_4b_it_nct_crc_he.dataset_url_train
        dataset_cache_directory = config_medgemma_4b_it_nct_crc_he.dataset_cache_directory
        file_path = os.path.join(dataset_cache_directory, url.split("/")[-1])
        download_zip_from_url(dataset_url=url, dataset_cache_path=file_path)

    dataset = load_dataset(path=cache_dir)
    print("Dataset structure")
    print(dataset)
    dataset = split_dataset(dataset, test_size=test_size, val_size=val_size, seed=841, max_examples=max_examples)
    print("Dataset structure after splitting")
    print(dataset)
    return dataset

# python
def split_dataset(
        ds: DatasetDict,
        test_size=0.02,
        val_size=0.02,
        seed: int | None = None,
        max_examples: int | None = None) -> DatasetDict:
    """
    Shuffle the training split, optionally keep only `max_examples`, then split into train/test/valid.
    Args:
        ds: DatasetDict with a 'train' split.
        test_size: fraction for final test set.
        val_size: fraction for final validation set.
        seed: optional int to make shuffling reproducible.
        max_examples: optional int to limit number of examples used (rest discarded).
    Returns:
        DatasetDict with 'train', 'test', and 'valid' splits.
    """
    if 'train' not in ds:
        raise ValueError("DatasetDict must contain a 'train' split to split from.")

    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be a positive integer or None.")

    # 1) Shuffle the training split
    shuffled_train = ds['train'].shuffle(seed=seed)

    # 2) Optionally restrict to only the first `max_examples` examples
    if max_examples is not None:
        n_keep = min(max_examples, len(shuffled_train))
        if n_keep < 2:
            raise ValueError("Not enough examples after applying max_examples.")
        small_train = shuffled_train.select(range(n_keep))
    else:
        small_train = shuffled_train

    # 3) First split: separate out (test + val) from the selected train subset
    combined_size = test_size + val_size
    if not 0.0 <= combined_size < 1.0:
        raise ValueError("test_size + val_size must be in range [0.0, 1.0).")

    train_testvalid = small_train.train_test_split(test_size=combined_size, seed=seed)

    # 4) Second split: split the (test + val) into test and valid
    # compute relative fraction of val within the combined portion
    if combined_size == 0:
        # no test/val requested -> all goes to train
        return DatasetDict({'train': train_testvalid['train'], 'test': train_testvalid['test'], 'valid': train_testvalid['test']})
    relative_val = val_size / combined_size if combined_size > 0 else 0.0
    test_valid = train_testvalid['test'].train_test_split(test_size=relative_val, seed=seed)

    # 5) Assemble DatasetDict
    return DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train'],
    })

if __name__ == "__main__":
    # # NIH Chest X-Ray Pneumonia Dataset
    # cache_directory = config_medgemma_4b_it_nih_cxr.dataset_cache_directory
    # dataset = load_data_chest_xray_pneumonia(cache_directory)
    # print(dataset)

    # NCT_CRC_HE dataset:
    zip_url_train = config_medgemma_4b_it_nct_crc_he.dataset_url_train
    zip_url_test = config_medgemma_4b_it_nct_crc_he.dataset_url_test

    output_directory = config_medgemma_4b_it_nct_crc_he.dataset_cache_directory
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
    save_path_train = os.path.join(output_directory, zip_url_train.split("/")[-1])
    save_path_test = os.path.join(output_directory, zip_url_test.split("/")[-1])

    dataset = load_data_nct_crc_he(test_size=0.02, val_size=0.02, max_examples=25000)
    print(dataset)
    # # download test data
    # download_zip_from_url(url=zip_url_train, save_path=save_path_train)

