import gc

import torch
import utils.model_utils as  model_utils
import config.config_medgemma_4b_it_brain_mri as config_medgemma_4b_it_brain_mri
from utils.fetch_data import load_data_brain_mri
from utils.prompt_utils import make_prompt_without_image
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
import re

class evaluate_medgemma_4b_it_brain_mri:
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
                "content": [{"type": "text", "text": config_medgemma_4b_it_brain_mri.prompt_template["system_message"]}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config_medgemma_4b_it_brain_mri.prompt_template["user_prompt"]},  # X-ray
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
            system_message=config_medgemma_4b_it_brain_mri.prompt_template["system_message"],
            user_prompt=config_medgemma_4b_it_brain_mri.prompt_template["user_prompt"],
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

    # @staticmethod
    # def evaluate_response(response):
    #     # Simple evaluation: check if the response contains the correct label
    #     references = [tissue_type.lower().strip()
    #                   for tissue_type in config_medgemma_4b_it_nct_crc_he.condition_findings]
    #
    #     response = response.lower().strip()
    #     result=-1
    #     for k,ref in enumerate(references):
    #         if ref in response:
    #             result = k
    #     return result



    @staticmethod
    def evaluate_response(response):
        references = [finding.lower().strip()
                    for finding in config_medgemma_4b_it_brain_mri.condition_findings]

        if not response:
            return -1

        resp = response.lower()
        matches = []

        for idx, ref in enumerate(references):
            if not ref:
                continue
            parts = ref.split()
            # allow any amount of whitespace between reference tokens, ensure word boundaries
            pattern = r'(?<!\w)' + r'\s+'.join(re.escape(p) for p in parts) + r'(?!\w)'
            if re.search(pattern, resp):
                matches.append(idx)

        return matches[0] if len(matches) == 1 else -1



if __name__ == "__main__":

    model_kwargs = config_medgemma_4b_it_brain_mri.model_kwargs
    evaluator = evaluate_medgemma_4b_it_brain_mri(
        model_id=config_medgemma_4b_it_brain_mri.base_model_id,
        model_folder=config_medgemma_4b_it_brain_mri.model_folder_base,
        model_kwargs=model_kwargs,
        max_new_tokens=250
    )

    dataset = load_data_brain_mri(test_size=0.025, val_size=0.025, max_examples=10000)
    dataset_test = dataset["test"]
    #dataset_test = dataset_test.select(list(range(10)) + list(range(len(dataset_test)-10, len(dataset_test), 1)))
    accuracy_baseline, f1_baseline = evaluator.evaluate(dataset_test, batch_size=4)

    print(f"Base Model Evaluation - Accuracy: {accuracy_baseline}, F1 Score: {f1_baseline}")

    del evaluator
    gc.collect()
    torch.clear_autocast_cache()

    evaluator = evaluate_medgemma_4b_it_brain_mri(
        model_id="peft",
        model_folder=config_medgemma_4b_it_brain_mri.model_folder_bmri_ft_full,
        model_kwargs=model_kwargs,
        max_new_tokens=250
    )

    accuracy_ft, f1_ft = evaluator.evaluate(dataset_test, batch_size=4)

    print(f"Fine tuned Model Evaluation - Accuracy: {accuracy_ft}, F1 Score: {f1_ft}")
