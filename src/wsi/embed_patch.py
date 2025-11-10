from torchvision import models, transforms
import torch
import torch.nn as nn
import wsi.config_wsi as config
from transformers import AutoModel, AutoImageProcessor
import wsi.config_wsi as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === hibou-L embedder (once) ===
processor = AutoImageProcessor.from_pretrained("histai/hibou-L")
hibou = AutoModel.from_pretrained("histai/hibou-L").to(DEVICE)
hibou.save_pretrained(config.positive_mil_cache)

hibou.eval()

@torch.no_grad()
def get_model_and_processor():
    if config.embedding_model == "ResNet50":
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        backbone = models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        backbone.eval()
        backbone.to(DEVICE)
    elif config.embedding_model == "hibou-L":
        preprocess = AutoImageProcessor.from_pretrained("histai/hibou-L")
        backbone = AutoModel.from_pretrained("histai/hibou-L").to(DEVICE)
        backbone.eval()
    else:
        # NotImplemented
        raise NotImplementedError("Embedding model not implemented")

    return backbone, preprocess



#hibou embedder
@torch.no_grad()
def embed_patch_hibou(patch_np, processor, hibou):  # (96,96,3) uint8
    inputs = processor(images=patch_np, return_tensors="pt").to(DEVICE)
    out = hibou(**inputs)
    return out.last_hidden_state[:, 0].cpu()  # (1024,)

# resnet50 embedder
@torch.no_grad()
def embed_patch_resnet(patch_np, processor, resnet):  # (96,96,3) uint8
    tensor = processor(patch_np).unsqueeze(0).to(DEVICE)
    emb = resnet(tensor).squeeze()  # (512,)
    return emb.cpu()

