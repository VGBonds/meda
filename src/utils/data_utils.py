from src.config import config_medgemma_4b_it_nih_cxr

# To prepare the dataset for fine-tuning, we will create a new column called "messages".
# This column will contain structured data representing a system context message,
# user query (the prompt) and assistant response (the correct label).

def format_data_medgemma_nih_chest_x_ray(example: dict[str, any]) -> dict[str, any]:
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
