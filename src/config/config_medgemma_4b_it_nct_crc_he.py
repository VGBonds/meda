import os

import torch
from trl import SFTConfig
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from huggingface_hub import login
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# login into huggingface hub if HF_TOKEN is set in environment variables
hf_token = os.environ['HF_TOKEN']
login(token=hf_token)

# get the project root folder
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Dataset ID and cache directory for the NIH Chest X-ray Pneumonia dataset
dataset_it = "nct_crc_he"
dataset_url_train = "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip"
dataset_url_test = "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip"
dataset_cache_directory = os.path.join(project_root, "data", "nct_crc_he")
dataset_cache_directory_train = os.path.join(dataset_cache_directory, "NCT-CRC-HE-100K")
dataset_cache_directory_test = os.path.join(dataset_cache_directory, "NCT-CRC-HE-7K")

# Model ID,  cache directory for storing pre-trained models and fine-tuned versions
base_model_id = "google/medgemma-4b-it"
model_folder_base = os.path.join(project_root, "models", "medgemma-4b-it")
model_folder_pneumonia_ft_adapter = os.path.join(project_root, "models", "medgemma-4b-it-nct-crc-he-adapter")
model_folder_pneumonia_ft_full = os.path.join(project_root, "models", "medgemma-4b-it-nct-crc-he-finetuned-merged")
# Model loading keyword arguments
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# !!! ONLY FOR 4-BIT QUANTIZED MODELS !!!  Only use for high end GPUs
# which support bfloat16 compute in 4-bit quantization (e.g., NVIDIA H100, A100)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

# prompt templates
condition_findings = [
    "A: adipose",
    "B: background",
    "C: debris",
    "D: lymphocytes",
    "E: mucus",
    "F: smooth muscle",
    "G: normal colon mucosa",
    "H: cancer-associated stroma",
    "I: colorectal adenocarcinoma epithelium"
]
condition_findings_str = "\n".join(condition_findings)
prompt_template = {
    "system_message": "You are a medical AI expert analyzing image patches from hematoxylin & eosin \
    stained histological images of human colorectal cancer (CRC) and normal tissue.",
    "user_prompt": f"What is the most likely tissue type shown in the histopathology image? \n {condition_findings_str}"
}


# set the fine-tuning configuration
sft_args = SFTConfig(
    output_dir="medgemma-4b-it-nct-crc-he",                     # Directory and Hub repository id to save the model to
    num_train_epochs=1,                                         # Number of training epochs
    per_device_train_batch_size=4,                              # Batch size per device during training
    per_device_eval_batch_size=4,                               # Batch size per device during evaluation
    gradient_accumulation_steps=4,                              # Number of steps before performing a backward/update pass
    gradient_checkpointing=True,                                # Enable gradient checkpointing to reduce memory usage
    optim="adamw_torch_fused",                                  # Use fused AdamW optimizer for better performance
    logging_steps=50,                                           # Number of steps between logs
    save_strategy="epoch",                                      # Save checkpoint every epoch
    eval_strategy="steps",                                      # Evaluate every `eval_steps`
    eval_steps=50,                                              # Number of steps between evaluations
    learning_rate=2e-4,                                         # Learning rate based on QLoRA paper
    bf16=True,                                                  # Use bfloat16 precision
    max_grad_norm=0.3,                                          # Max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                                          # Warmup ratio based on QLoRA paper
    lr_scheduler_type="linear",                                 # Use linear learning rate scheduler
    push_to_hub=False,                                          # Push model to Hub
    report_to="tensorboard",                                    # Report metrics to tensorboard
    gradient_checkpointing_kwargs={"use_reentrant": False},     # Set gradient checkpointing to non-reentrant to avoid issues
    dataset_kwargs={"skip_prepare_dataset": True},              # Skip default dataset preparation to preprocess manually
    remove_unused_columns = False,                              # Columns are unused for training but needed for data collator
    label_names=["labels"],                                     # Input keys that correspond to the labels
)



lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,  # Rank of the adaptation matrices
    bias="none",  # biases won't be fine-tuned
    target_modules="all-linear", # linear layers will be fine-tuned
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)


# lora_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.05,
#     r=16,
#     bias="none",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Specific attention and MLP layers
#     task_type="CAUSAL_LM",
#     modules_to_save=[
#         "lm_head",
#         "embed_tokens",
#     ],
# )
