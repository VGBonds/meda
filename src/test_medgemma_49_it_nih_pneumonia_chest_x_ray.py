import torch
from src.utils import model_utils
from src.config import config_medgemma_4b_it_nih_cxr
from src.utils.fetch_data import load_data_chest_xray_pneumonia

class test_medgemma_4b_it_nih_pneumonia_chest_x_ray:
    def __init__(self, model_id, model_folder, model_kwargs, max_new_tokens=250):

        self.model, self.processor = model_utils.load_model_and_processor(model_id,
                                                                          model_folder,
                                                                          model_kwargs)

        self.messages_template = [
            {
                "role": "system",
                "content": [{"type": "text", "text": config_medgemma_4b_it_nih_cxr.prompt_template["system_message"]}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"]},  # X-ray
                    {"type": "image", "image": None}
                ]
            }
        ]

    @staticmethod
    def make_prompt(system_message, user_prompt, image):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},  # X-ray
                    {"type": "image", "image": image}
                ]
            }
        ]
        return messages

    def chat(self, messages):

        # Process inputs (apply chat template, tokenize text, preprocess image)
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Generate output (inference mode for efficiency)
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False  # Greedy decoding; use do_sample=True for varied outputs
            )
            generation = generation[0][input_len:]

        # Decode the generated text
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        print("Generated Description:", decoded)
        return decoded


if __name__ == "__main__":

    model_kwargs=config_medgemma_4b_it_nih_cxr.model_kwargs
    test_instance_base = test_medgemma_4b_it_nih_pneumonia_chest_x_ray(
        model_id=config_medgemma_4b_it_nih_cxr.base_model_id,
        model_folder=config_medgemma_4b_it_nih_cxr.model_folder_base,
        model_kwargs=model_kwargs,
        max_new_tokens=250
    )


    dataset = load_data_chest_xray_pneumonia()
    messages = test_instance_base.make_prompt(
        system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
        user_prompt=config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"],
        image=dataset["test"][-7]["image"])
    print(f"True finding:{config_medgemma_4b_it_nih_cxr.condition_findings[dataset['test'][-7]['label']]}")
    base_assistant_message = test_instance_base.chat(messages)
    print(f"Baseline assistant message:{base_assistant_message}")

    # test with the fine-tuned model
    test_instance_ft = test_medgemma_4b_it_nih_pneumonia_chest_x_ray(
        model_id="peft",
        model_folder=config_medgemma_4b_it_nih_cxr.model_folder_pneumonia_ft_full,
        model_kwargs=model_kwargs,
        max_new_tokens=250
    )
    messages = test_instance_ft.make_prompt(
        system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
        user_prompt=config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"],
        image=dataset["test"][-7]["image"])
    print(f"True finding:{config_medgemma_4b_it_nih_cxr.condition_findings[dataset['test'][-7]['label']]}")
    ft_assistant_message = test_instance_ft.chat(messages)
    print(f"Fine-tuned assistant message:{ft_assistant_message}")

    augmented_user_prompt = (config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"]
                             + f"\nIn your answer, please consider that a fine-tuned model ML model on a pneumonia"
                               f"\nchest X-ray dataset does predict the label **{ft_assistant_message}**  in this case. "
                               f"\nProvide a detailed explanation for your diagnosis.")
    augmented_messages = test_instance_ft.make_prompt(
        system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
        user_prompt=augmented_user_prompt,
        image=dataset["test"][-7]["image"])
    augmented_assistant_message = test_instance_base.chat(augmented_messages)

