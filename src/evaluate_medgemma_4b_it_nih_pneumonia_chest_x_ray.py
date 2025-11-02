import gc

import torch
import utils.model_utils as model_utils
import config.config_medgemma_4b_it_nih_cxr as config_medgemma_4b_it_nih_cxr
from utils.fetch_data import load_data_chest_xray_pneumonia
from utils.prompt_utils import make_prompt_without_image
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image

class evaluate_medgemma_4b_it_nih_pneumonia_chest_x_ray:
    def __init__(self, model_id, model_folder, model_kwargs, max_new_tokens=250):
        # set up model and processor
        self.model, self.processor = model_utils.load_model_and_processor(
            model_id,
            model_folder,
            model_kwargs,
        )

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

    def evaluate(self, dataset_split, batch_size=8):
        """
        Evaluate on dataset_split in minibatches.
        batch_size: number of examples per minibatch
        """
        true_labels = list(dataset_split["label"])
        predicted_labels = []

        n = len(dataset_split)

        # prepare base messages (no image) once
        base_messages = make_prompt_without_image(
            system_message=config_medgemma_4b_it_nih_cxr.prompt_template["system_message"],
            user_prompt=config_medgemma_4b_it_nih_cxr.prompt_template["user_prompt"],
        )

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = dataset_split.select(list(range(start,end)))

            # extract images for the batch
            images_batch = [[ex["image"]] for ex in batch]

            # create text prompts for each item in the batch (same template repeated)
            texts = [
                self.processor.apply_chat_template(base_messages, add_generation_prompt=True, tokenize=False).strip()
                for _ in images_batch
            ]

            # Get model predictions for the minibatch (list of decoded strings)
            decoded_list = self.predict(texts, images_batch)

            # convert decoded outputs to label indices and append
            for decoded in decoded_list:
                predicted_labels.append(self.evaluate_response(decoded))

        accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_labels)
        f1 = f1_score(y_true=true_labels, y_pred=predicted_labels, average="weighted")
        print(f"Accuracy: {accuracy})")
        print(f"F1 Metric: {f1})")

        return accuracy, f1


    def predict(self, texts, images):

        # normalize texts
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            texts = list(texts)

        # normalize images to a flat list first
        if isinstance(images, (Image.Image, torch.Tensor, dict)):
            images = [images]
        elif not isinstance(images, list):
            images = list(images)

        # If images is a flat list like [img1, img2, ...], convert to list-of-lists:
        #   [[img1], [img2], ...] so Gemma3Processor treats each inner list as one sample.
        if not any(isinstance(el, (list, tuple)) for el in images):
            images = [[el] for el in images]
        else:
            # ensure each element is a list, not tuple
            images = [list(el) for el in images]

        # final length check
        if len(texts) != len(images):
            raise ValueError(
                f"Mismatch between text and image batch sizes: {len(texts)} texts vs {len(images)} image-lists")

        # call processor (returns tensors / BatchFeature)
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        # move tensor items to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )

        generations = generation[:, input_len:]

        decoded_list = []
        for i, gen in enumerate(generations):
            decoded = self.processor.decode(gen, skip_special_tokens=True).strip().lower()
            decoded_list.append(decoded)

        return decoded_list

    @staticmethod
    def evaluate_response(response, references=("b:pneumonia", "b: pneumonia")):
        # Simple evaluation: check if the response contains the correct label
        response = response.lower().strip()
        references = [reference.lower().strip() for reference in references]
        result = [1 if reference in response else 0 for reference in references]
        result = 1 if sum(result)>=1  else 0
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
    #dataset_test = dataset_test.select(list(range(10)) + list(range(len(dataset_test)-10, len(dataset_test), 1)))
    accuracy_baseline, f1_baseline = evaluator.evaluate(dataset_test, batch_size=4)

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

    accuracy_ft, f1_ft = evaluator.evaluate(dataset_test, batch_size=4)

    print(f"Fine tuned Model Evaluation - Accuracy: {accuracy_ft}, F1 Score: {f1_ft}")
