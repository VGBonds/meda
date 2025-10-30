import gc

import torch
from src.utils import model_utils
from src.config import config_medgemma_4b_it_nih_cxr
from src.utils.fetch_data import load_data_chest_xray_pneumonia
from src.utils.prompt_utils import make_prompt_without_image
from sklearn.metrics import f1_score, accuracy_score

class evaluate_medgemma_4b_it_nih_pneumonia_chest_x_ray:
    def __init__(self, model_id, model_folder, model_kwargs, max_new_tokens=250):

        # set up model and processor
        self.model, self.processor = model_utils.load_model_and_processor(model_id,
                                                                          model_folder,
                                                                          model_kwargs)

        self.max_new_tokens = max_new_tokens

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

    # def predict(model, processor, example):
    #
    #     # if print_message:
    #     #     print(f"EXAMPLE MESSAGE:{messages}")
    #     #     print_message = False
    #     with torch.no_grad():  # Reduces memory by not storing gradients/activations
    #         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #         inputs = processor(text=[text], images=[example["image"]], return_tensors="pt", padding=True).to(
    #             model.device)
    #         output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    #         # output = processor.decode(output_ids[0], skip_special_tokens=True).strip().lower()
    #         output = processor.decode(output_ids[0], skip_special_tokens=False).strip().lower()
    #         return 1 if "b: pneumonia" in output else 0  # Map to label index

    def evaluate(self, dataset_split):

        true_labels = dataset_split["label"]
        predicted_labels = []

        for example in dataset_split:
            # Prepare messages with the image
            messages = make_prompt_without_image(
                system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
                user_prompt=config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"],
            )

            # Get model prediction
            response = self.predict(messages, example["image"])

            predicted_labels.append(self.evaluate_response(response))

        accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_labels)
        f1 = f1_score(y_true=true_labels, y_pred=predicted_labels, average="weighted")
        print(f"Accuracy: {accuracy})")
        print(f"F1 Metric: {f1})")

        return accuracy, f1

    def predict(self, messages, images):

        # Process inputs (apply chat template, tokenize text, preprocess image)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False).strip()
        inputs = self.processor(
            text=[text],
            images=[images],
            return_tensors="pt",
            padding=True).to(self.model.device)

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
        decoded = self.processor.decode(generation, skip_special_tokens=True).strip().lower()
        print("Generated Description:", decoded)
        return decoded

    @staticmethod
    def evaluate_response(response, reference="b:pneumonia"):
        # Simple evaluation: check if the response contains the correct label
        response = response.lower()
        reference = reference.lower()
        result = 1 if reference in response else 0
        return result


if __name__ == "__main__":

    model_kwargs = config_medgemma_4b_it_nih_cxr.model_kwargs
    evaluator = evaluate_medgemma_4b_it_nih_pneumonia_chest_x_ray(
        model_id=config_medgemma_4b_it_nih_cxr.base_model_id,
        model_folder=config_medgemma_4b_it_nih_cxr.model_folder_base,
        model_kwargs=model_kwargs,
        max_new_tokens=250
    )

    dataset = load_data_chest_xray_pneumonia()
    dataset_test = dataset["test"]
    dataset_test = dataset_test.select(list(range(10)) + list(range(len(dataset_test)-10,len(dataset_test), 1 )))
    accuracy_baseline, f1_baseline = evaluator.evaluate(dataset_test)

    print(f"Base Model Evaluation - Accuracy: {accuracy_baseline}, F1 Score: {f1_baseline}")

    del evaluator
    gc.collect()
    torch.clear_autocast_cache()

    evaluator = evaluate_medgemma_4b_it_nih_pneumonia_chest_x_ray(
        model_id="peft",
        model_folder=config_medgemma_4b_it_nih_cxr.model_folder_pneumonia_ft_full,
        model_kwargs=model_kwargs,
        max_new_tokens=250
    )

    accuracy_ft, f1_ft = evaluator.evaluate(dataset_test)

    print(f"Base Model Evaluation - Accuracy: {accuracy_ft}, F1 Score: {f1_ft}")





