import os
from sys import base_prefix

import torch
from sympy.strategies import condition
from trl import SFTConfig
from peft import LoraConfig

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
dataset_id = "hf-vision/chest-xray-pneumonia"
dataset_cache_directory = os.path.join(project_root, "data", "chest_xray_pneumonia")

# Model ID,  cache directory for storing pre-trained models and fine-tuned versions
base_model_id = "google/medgemma-4b-it"
model_folder_base = os.path.join(project_root, "models", "medgemma-4b-it")
model_folder_pneumonia_ft_adapter = os.path.join(project_root, "models", "medgemma-4b-it-pneumonia-finetuned-adapter")
model_folder_pneumonia_ft_full = os.path.join(project_root, "models", "medgemma-4b-it-pneumonia-finetuned-merged")
# Model loading keyword arguments
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# prompt templates
condition_findings = ['A: NORMAL', 'B: PNEUMONIA']
condition_findings_str = "\n".join(condition_findings)
prompt_template = {
    "system_message": "You are a medical AI expert analyzing chest X-rays \
    for pneumonia detection.",
    "user_prompt": f"Analyze this chest X-ray image for signs of pneumonia \
    or normal lungs. Do you detect: \n {condition_findings_str}"
}

# set the fine-tuning configuration
sft_args  = SFTConfig(
    output_dir="./medgemma-pneumonia-x-ray",                    # Directory and Hub repository id to save the model to
    num_train_epochs=5,                                         # Number of training epochs
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
