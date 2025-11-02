from trl import SFTTrainer

import config.config_medgemma_4b_it_nih_cxr as config_medgemma_4b_it_nih_cxr
import utils.model_utils  as model_utils
from utils.fetch_data import load_data_chest_xray_pneumonia
from utils.data_utils import format_data_medgemma_nih_chest_x_ray
from typing import Any


def fine_tune_medgemma_nih_pneumonia_chest_x_ray(
        model,
        processor,
        data,
        base_model_directory,
        save_directory_adapters,
        save_directory_full_model,
        model_kwargs,
):
    """
    Fine-tunes the MedGemma model on the NIH Pneumonia Chest X-Ray dataset.
    Saves both the fine-tuned model adapters and the full fine-tuned model.

    Parameters:
    - model: Pre-trained MedGemma model.
    - processor: MedGemma processor for text and image processing.
    - data: Dictionary containing training and validation datasets.
    - base_model_directory: Directory of the base MedGemma model.
    - save_directory_adapters: Directory to save the fine-tuned model adapters.
    - save_directory_full_model: Directory to save the full fine-tuned model.
    - model_kwargs: Additional keyword arguments for the model.

    Returns:
    - .
    """

    # function to transform the training examples into the format accepted by MedGemma

    def collate_fn(examples: list[dict[str, Any]]):
        texts = []
        images = []
        for example in examples:
            images.append([example["image"]])
            texts.append(
                processor.apply_chat_template(
                    example["messages"], add_generation_prompt=False, tokenize=False
                ).strip()
            )

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, with the padding and image tokens masked in
        # the loss computation
        labels = batch["input_ids"].clone()

        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        # Mask tokens that are not used in the loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    trainer = SFTTrainer(
        model=model,
        args=config_medgemma_4b_it_nih_cxr.sft_args,
        train_dataset=data["train"],
        # eval_dataset=formatted_data["validation"].shuffle().select(range(10)),
        eval_dataset=data["validation"],
        peft_config=config_medgemma_4b_it_nih_cxr.lora_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    trainer.train()

    # Save the fine-tuned model adapters and processor
    model_utils.save_ft_model_adapters_and_processor(trainer, processor, save_directory_adapters)
    # Save the full fine-tuned model (merged with adapters) and processor
    model_utils.save_ft_model_and_processor(
        save_directory=save_directory_full_model,
        base_model_directory=base_model_directory,
        ft_adapters_directory=save_directory_adapters,
        model_kwargs=model_kwargs
    )

if __name__ == "__main__":
    # Load and preprocess dataset
    dataset = load_data_chest_xray_pneumonia()
    # Format the dataset
    formatted_dataset = dataset.map(format_data_medgemma_nih_chest_x_ray)

    # Load base model and processor
    model_id = "medgemma-4b-it"
    base_model, processor = model_utils.load_model_and_processor(
        model_id=config_medgemma_4b_it_nih_cxr.base_model_id,
        model_directory=config_medgemma_4b_it_nih_cxr.model_folder_base,
        model_kwargs=config_medgemma_4b_it_nih_cxr.model_kwargs,
    )


    # Fine-tune the model
    fine_tune_medgemma_nih_pneumonia_chest_x_ray(
        model=base_model,
        processor=processor,
        data=formatted_dataset,
        base_model_directory=config_medgemma_4b_it_nih_cxr.model_folder_base,
        save_directory_adapters=config_medgemma_4b_it_nih_cxr.model_folder_pneumonia_ft_adapter,
        save_directory_full_model=config_medgemma_4b_it_nih_cxr.model_folder_pneumonia_ft_full,
        model_kwargs=config_medgemma_4b_it_nih_cxr.model_kwargs,
    )



