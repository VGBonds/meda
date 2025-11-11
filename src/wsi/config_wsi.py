from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

# Load environment variables from .env file
load_dotenv()

# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# login into huggingface hub if HF_TOKEN is set in environment variables
hf_token = os.environ['HF_TOKEN']
login(token=hf_token)

# get the project root folder
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


negative_wsi_folder="/Volumes/Data2/negWSI"

embedding_model = "hibou-L" #"hibou-L" . "ResNet50"

patch_dataset_id = "wltjr1007/Camelyon17-WILDS"
cache_directory = os.path.join(project_root, "data", "camelyon17")
positive_mil_cache = os.path.join(project_root,  "data", "camelyon17_bags","positive_mil_bags.pt")
negative_mil_cache = os.path.join(project_root,  "data", "camelyon17_bags", "negative_mil_bags.pt")

embedding_model_cache_directory = os.path.join(project_root, "models", embedding_model)

trained_model_cache_directory = os.path.join(project_root, "models", "saved_models")
trained_model_file = os.path.join(trained_model_cache_directory, "abmil_hibou_best.pth")

random_seed = 841

n_patches_per_slide = None # pick all patches
patch_size = 224
level = 2
stride = 224

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

# === Model CONFIG ===
INPUT_DIM = 1024        # hibou-L output
EPOCHS = 20
LR = 2e-4
PATIENCE = 5
MODEL_SAVE = "abmil_hibou_best.pth"

