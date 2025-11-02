from typing import Any
import config.config_medgemma_4b_it_nih_cxr as config_medgemma_4b_it_nih_cxr
import config.config_medgemma_4b_it_nct_crc_he as config_medgemma_4b_it_nct_crc_he
from datasets import load_dataset


def format_data_medgemma_nct_crc_he(example: dict[str, Any]) -> dict[str, Any]:
    example["messages"] = [
        {"role": "system",
         "content": [
             {"type": "text",
              "text": config_medgemma_4b_it_nct_crc_he.prompt_template["system_message"]
              }
         ]
         },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": config_medgemma_4b_it_nct_crc_he.prompt_template["user_prompt"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": config_medgemma_4b_it_nct_crc_he.condition_findings[example["label"]],
                },
            ],
        },
    ]
    return example

# To prepare the dataset for fine-tuning, we will create a new column called "messages".
# This column will contain structured data representing a system context message,
# user query (the prompt) and assistant response (the correct label).
def format_data_medgemma_nih_chest_x_ray(example: dict[str, Any]) -> dict[str, Any]:
    example["messages"] = [
        {"role": "system",
         "content": [
             {"type": "text",
              "text": config_medgemma_4b_it_nih_cxr.prompt_template["system_message"]
              }
         ]
         },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": config_medgemma_4b_it_nih_cxr.condition_findings[example["label"]],
                },
            ],
        },
    ]
    return example

