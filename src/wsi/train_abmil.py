# --------------------------------------------------------------
# 2. ABMIL MODEL + TRAINING LOOP
# --------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torchmil.models import ABMIL
from torchmil.utils import Trainer
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import wsi.config_wsi as config
from wsi.make_bags_dataset import make_bags_dataset
from tqdm.auto import tqdm

# # === CONFIG ===
# INPUT_DIM = 1024        # hibou-L output
# EPOCHS = 20
# LR = 2e-4
# PATIENCE = 5
# MODEL_SAVE = "abmil_hibou_best.pth"

def train_abmil(train_loader, val_loader):
    # === MODEL ===
    device = config.DEVICE
    criterion = nn.BCEWithLogitsLoss()
    model = ABMIL(in_shape=(config.INPUT_DIM,), criterion=criterion).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # === TRAINING CONFIG ===
    EPOCHS = 30
    PATIENCE = 7
    BEST_AUC = 0.0
    MODEL_PATH = "abmil_hibou_final.pth"
    patience_counter = 0

    print("Starting custom training loop...")

    for epoch in range(config.EPOCHS):
        # ------------------- TRAIN -------------------
        model.train()
        train_losses = []
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            feats = batch["features"].to(device)  # (B, MAX_N, D)
            mask = batch["mask"].to(device)  # (B, MAX_N)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            # logits, _ = model(feats, mask=mask, return_attention=false)
            logits, att = model(feats, mask=mask, return_att=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds = train_preds + torch.sigmoid(logits).cpu().detach().numpy().tolist()
            train_labels = train_labels + labels.cpu().numpy().tolist()

        train_auc = roc_auc_score(train_labels, train_preds)
        train_loss = np.mean(train_losses)

        # ------------------- VALIDATION -------------------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(feats, mask=mask)
                val_preds = val_preds + torch.sigmoid(logits).cpu().detach().numpy().tolist()
                val_labels = val_labels + labels.cpu().numpy().tolist()

        val_auc = roc_auc_score(val_labels, val_preds)

        # ------------------- EARLY STOPPING -------------------
        if val_auc > BEST_AUC:
            BEST_AUC = val_auc
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
            print(f"NEW BEST: AUC={val_auc:.4f} → saved")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f} | Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {BEST_AUC:.4f}")
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Best model loaded.")
    return model


# def train_abmil(train_loader, val_loader):
#     # === MODEL ===
#     criterion = nn.BCEWithLogitsLoss()
#     model = ABMIL(in_shape=(config.INPUT_DIM,), criterion=criterion).to(config.DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=config.LR)
#
#     train_abmil(train_loader, val_loader, model)
#
#     # Instantiate the Trainer
#     # Trainer expects: features (B, N, D), labels (B,)
#     trainer = Trainer(
#         model=model,
#         optimizer=optimizer,
#         device=config.DEVICE,
#         obj_metric="roc_auc",
#         obj_metric_mode = "max",
#         early_stop_patience = config.PATIENCE,
#
#     )
#
#     # === TRAIN ===
#     print("Starting training...")
#     trainer.train(
#         train_dataloader=train_loader,
#         val_dataloader=val_loader,
#         max_epochs=config.EPOCHS,
#     )
#     print("Training complete.")
#
#     # Save the model
#     # torch.save(model.state_dict(), config.trained_model_file)
#     torch.save(trainer.get_best_model_state_dict(), config.trained_model_file)
#     print(f"Model saved to {config.trained_model_file}")
#     return trainer.get_best_model()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_bags_dataset()
    best_model = train_abmil(train_loader, val_loader)
    print("Best model loaded.")