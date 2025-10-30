import torch
from src.utils import model_utils
from src.config import config_medgemma_4b_it_nih_cxr
from src.utils.fetch_data import load_data_chest_xray_pneumonia
from src.utils.prompt_utils import make_prompt_with_image, make_prompt_without_image

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

        self.max_new_tokens = max_new_tokens


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
                max_new_tokens=self.max_new_tokens,
                do_sample=False  # Greedy decoding; use do_sample=True for varied outputs
            )
            generation = generation[0][input_len:]

        # Decode the generated text
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        # print("Generated Description:", decoded)
        return decoded

    def chat_v1(self, messages, images):

        # Process inputs (apply chat template, tokenize text, preprocess image)
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            return_dict=True, return_tensors="pt"
        ).strip()#.to(self.model.device, dtype=torch.bfloat16)
        inputs = self.processor(text=[text], images=[images], return_tensors="pt", padding=True).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        # Generate output (inference mode for efficiency)
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
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
    # messages = make_prompt_with_image(
    #     system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
    #     user_prompt=config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"],
    #     image=dataset["test"][-7]["image"])
    messages = make_prompt_without_image(
        system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
        user_prompt=config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"])
    print(f"True finding:{config_medgemma_4b_it_nih_cxr.condition_findings[dataset['test'][-7]['label']]}")
    # base_assistant_message = test_instance_base.chat(messages)
    base_assistant_message = test_instance_base.chat_v1(messages, dataset["test"][-7]["image"])

    print(f"Baseline assistant message:{base_assistant_message}")

    # test with the fine-tuned model
    test_instance_ft = test_medgemma_4b_it_nih_pneumonia_chest_x_ray(
        model_id="peft",
        model_folder=config_medgemma_4b_it_nih_cxr.model_folder_pneumonia_ft_full,
        model_kwargs=model_kwargs,
        max_new_tokens=250
    )
    # messages = make_prompt_with_image(
    #     system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
    #     user_prompt=config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"],
    #     image=dataset["test"][-7]["image"])
    messages = make_prompt_without_image(
        system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
        user_prompt=config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"])
    print(f"True finding:{config_medgemma_4b_it_nih_cxr.condition_findings[dataset['test'][-7]['label']]}")
    # ft_assistant_message = test_instance_ft.chat(messages)
    ft_assistant_message = test_instance_ft.chat_v1(messages, dataset["test"][-7]["image"])
    print(f"Fine-tuned assistant message:{ft_assistant_message}")

    augmented_user_prompt = (config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"]
                             + f"\nIn your answer, please consider that a fine-tuned model ML model on a pneumonia"
                               f"\nchest X-ray dataset does predict the label **{ft_assistant_message}**  in this case. "
                               f"\nProvide a detailed explanation for your diagnosis.")
    # augmented_messages = make_prompt_with_image(
    #     system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
    #     user_prompt=augmented_user_prompt,
    #     image=dataset["test"][-7]["image"])
    augmented_messages = make_prompt_without_image(
        system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
        user_prompt=augmented_user_prompt)
    # augmented_assistant_message = test_instance_base.chat(augmented_messages)
    augmented_assistant_message = test_instance_base.chat_v1(augmented_messages, dataset["test"][-7]["image"])

    print(f"Augmented-tuned assistant message:{ft_assistant_message}")