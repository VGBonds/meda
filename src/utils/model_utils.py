import os
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel


def load_model_and_processor(model_id, model_directory, model_kwargs):
    # For PEFT adapters we expect a local directory; fail fast if loading fails
    if model_id == 'peft':
        try:
            return load_model_and_processor_from_disk(model_directory, model_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load PEFT model from disk: {e}")

    # For known hub models, try loading from disk first and fall back to downloading
    elif model_id == "google/medgemma-4b-it":
        # Use provided directory or a default directory derived from model_id
        save_directory = model_directory or f"./{model_id.replace('/', '_')}-model"

        try:
            return load_model_and_processor_from_disk(save_directory, model_kwargs)
        except Exception as e:
            print(f"Could not load model from {save_directory}: {e}. Downloading from Hugging Face Hub...")
            # Ensure the download directory exists and download
            download_model_and_processor(model_id, save_directory, model_kwargs)
            # Retry loading from disk after download
            return load_model_and_processor_from_disk(save_directory, model_kwargs)

    else:
        raise NotImplementedError(f"Model id {model_id} not supported")


def download_model_and_processor(model_id, save_directory, model_kwargs):
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Download and save the model (with optimizations for efficiency)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **model_kwargs
    )
    model.save_pretrained(save_directory)

    # Download and save the processor
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    processor.save_pretrained(save_directory)

    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"
    print(f"Model and processor saved to {save_directory}")
    return model, processor


def load_model_and_processor_from_disk(save_directory, model_kwargs):
    # Load the model
    model = AutoModelForImageTextToText.from_pretrained(
        save_directory,
        **model_kwargs
    )
    # Load the processor
    processor = AutoProcessor.from_pretrained(save_directory, use_fast=True)
    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"

    print(f"Model and processor loaded from {save_directory}")
    return model, processor


def save_ft_model_adapters_and_processor(trainer, processor, save_directory):
    # save the  LoRA adapters

    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Save LoRA adapters and processor
    trainer.save_model(save_directory)
    processor.save_pretrained(save_directory)

    print(f"Fine-tuned model adapters and processor saved to {save_directory}")


def load_ft_model_adapters_and_processor_from_disk(
        base_model_directory,
        ft_adapters_directory,
        model_kwargs):
    # Load the fine-tuned model adapters and processor from local disk

    # Load base model
    base_model, _ = load_model_and_processor_from_disk(
        save_directory=base_model_directory,
        model_kwargs=model_kwargs
    )
    print(f"Base model loaded from {base_model_directory}")
    # base_model.to("cuda")

    # Load fine tuned model LoRA adapters
    ft_adapters = PeftModel.from_pretrained(base_model, ft_adapters_directory)
    # Merge the adapters with the base model
    ft_model = ft_adapters.merge_and_unload()

    # Load the processor
    ft_processor = AutoProcessor.from_pretrained(ft_adapters_directory, use_fast=True)
    # Use right padding to avoid issues during training
    ft_processor.tokenizer.padding_side = "right"

    print(f"Fine-tuned model adapters and processor loaded from {ft_adapters_directory}")
    return ft_model, ft_processor


# merge and save a standalone model. Use only CPU so that any GPY errors are avoided
def save_ft_model_and_processor(
        save_directory,
        base_model_directory,
        ft_adapters_directory,
        model_kwargs):

    # Load base model without quantization
    base_model, _ = load_model_and_processor_from_disk(
        save_directory=base_model_directory,
        model_kwargs=model_kwargs)
    # base_model.to("cpu") for CPU loading and saving

    # Load LoRA adapters
    ft_adapters = PeftModel.from_pretrained(base_model, ft_adapters_directory)
    # Merge adapters
    ft_model = ft_adapters.merge_and_unload()

    # Load the processor
    ft_processor = AutoProcessor.from_pretrained(ft_adapters_directory, use_fast=True)
    # Use right padding to avoid issues during training
    ft_processor.tokenizer.padding_side = "right"

    # Save merged model
    ft_model.save_pretrained(save_directory)
    ft_processor.save_pretrained(save_directory)


if __name__ == "__main__":
    # Example usage
    model_id = "google/medgemma-4b-it"
    model_directory = "./medgemma-4b-it-model"
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype="auto",
        device_map="auto",
    )

    # Download and load model and processor
    model, processor = load_model_and_processor(model_id, model_directory, model_kwargs)
