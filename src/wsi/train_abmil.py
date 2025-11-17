# --------------------------------------------------------------
# 2. ABMIL MODEL + TRAINING LOOP
# --------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchmil.models import ABMIL
from torchmil.utils import Trainer
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import wsi.config_wsi as config
from wsi.make_bags_dataset import make_bags_dataset
from tqdm.auto import tqdm
from wsi.abmil import ABMIL

def train_abmil(train_loader, val_loader):
    # === MODEL ===

    criterion = nn.BCEWithLogitsLoss()
    # model = ABMIL(in_shape=(config.INPUT_DIM,), criterion=criterion).to(device)
    # torchmil implementation with gated attention
    # model = ABMIL(in_shape=(config.INPUT_DIM,), gated=True, att_dim=512, att_act="relu", criterion=criterion, ).to(device)
    model = ABMIL(input_size=config.INPUT_DIM,
                  hidden_dim=512,
                  dropout=config.DROPOUT_RATE,
                  dropout_attn=config.DROPOUT_RATE_ATTN).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    print(model)

    # === TRAINING CONFIG ===
    EPOCHS = config.EPOCHS
    PATIENCE = config.PATIENCE
    BEST_AUC = 0.0
    BEST_VAL_LOSS = float("inf")
    BEST_TRAIN_LOSS = float("inf")

    MODEL_PATH = config.trained_model_file
    os.makedirs(config.trained_model_cache_directory, exist_ok=True)
    patience_counter = 0
    epsilon = 1e-2

    print("Starting custom training loop...")

    for epoch in range(config.EPOCHS):
        # ------------------- TRAIN -------------------
        model.train()
        train_losses = []
        train_preds, train_labels = [], []
        max_att = []
        min_att = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            feats = batch["features"].to(config.DEVICE)  # (B, MAX_N, D)
            mask = batch["mask"].to(config.DEVICE)  # (B, MAX_N)
            labels = batch["labels"].to(config.DEVICE)
            optimizer.zero_grad()
            # logits, _ = model(feats, mask=mask, return_attention=false)
            # logits, att = model(feats, mask=mask, return_att=True)
            logits, att = model(feats, mask=mask, return_attention=True)

            loss_ce = criterion(logits, labels)
            entropy = -torch.sum(att * torch.log(att + 1e-8), dim=1).mean()
            loss = loss_ce + config.LAMBDA_ENTROPY * entropy
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds = train_preds + torch.sigmoid(logits).cpu().detach().numpy().tolist()
            train_labels = train_labels + labels.cpu().numpy().tolist()
            max_att_t = torch.max(att , dim=1).values.detach().cpu().numpy().tolist()
            min_att_t = torch.min(att , dim=1).values.detach().cpu().numpy().tolist()
            max_att = max_att + [round(v,4) for v in max_att_t]
            min_att = min_att +[round(v,4) for v in min_att_t]

        print(f"max attention: {max_att})")
        print(f", min attention: {min_att}")

        train_auc = roc_auc_score(train_labels, train_preds)
        train_loss = np.mean(train_losses)

        # ------------------- VALIDATION -------------------
        model.eval()
        val_preds, val_labels, val_losses = [], [], []
        val_preds_np = []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch["features"].to(config.DEVICE)
                mask = batch["mask"].to(config.DEVICE)
                labels = batch["labels"].to(config.DEVICE)
                logits, att = model(feats, mask=mask, return_attention=True)
                loss_ce = criterion(logits, labels)
                entropy = -torch.sum(att * torch.log(att + 1e-8), dim=1).mean()
                loss = loss_ce + config.LAMBDA_ENTROPY * entropy
                val_losses.append(loss.item())
                val_preds_np.append(torch.sigmoid(logits).cpu().detach().numpy())
                val_preds = val_preds + torch.sigmoid(logits).cpu().detach().numpy().tolist()
                val_labels = val_labels + labels.cpu().numpy().tolist()

        val_preds_np = np.concatenate(val_preds_np)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_accuracy = accuracy_score(y_true=val_labels,
                                      y_pred=np.where(val_preds_np > 0.5, 1, 0))
        val_loss = np.mean(val_losses)
        # ------------------- EARLY STOPPING -------------------
        if  train_loss<BEST_TRAIN_LOSS: #val_loss<BEST_VAL_LOSS: # and not(val_auc < (BEST_AUC-epsilon)):
            BEST_AUC = val_auc
            BEST_VAL_LOSS = val_loss
            BEST_TRAIN_LOSS = train_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
            print(f"NEW BEST: TRAIN LOSS={train_loss}, VAL LOSS={val_loss},AUC={val_auc:.4f}, Accuracy={val_accuracy:.4f} → saved")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f},  | Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {BEST_AUC:.4f}")
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Best model loaded.")
    return model


if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_bags_dataset()
    best_model = train_abmil(train_loader, val_loader)
    print("Best model loaded.")