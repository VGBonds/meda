from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

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

random_seed = 841

n_patches_per_slide = None # pick all patches
patch_size = 224
level = 2
stride = 224
