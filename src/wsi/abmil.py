import torch
import torch.nn as nn
from transformers.models.prophetnet.modeling_prophetnet import softmax


class TriplePathologistABMIL(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        # 1. High-recall (cautious) — low pos_weight → penalizes FN
        self.recall_abmil = ABMIL(input_size)
        # 2. High-precision (strict) — high pos_weight → penalizes FP
        self.precision_abmil = ABMIL(input_size)
        # 3. Arbiter — sees features + both attentions
        self.arbiter = nn.Sequential(
            nn.Linear(input_size + 2, 256),  # +2 for two attentions
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, mask=None):
        # Recall & Precision predictions
        recall_logits, recall_att = self.recall_abmil(x, mask, return_attention=True)
        precision_logits, precision_att = self.precision_abmil(x, mask, return_attention=True)

        # Arbiter input: weighted average + original features
        att_combined = torch.stack([recall_att, precision_att], dim=-1)  # (B, N, 2)
        x_with_att = torch.cat([x, att_combined], dim=-1)  # (B, N, D+2)
        bag_rep = (x_with_att * recall_att.unsqueeze(-1)).sum(dim=1)  # use recall att as base

        final_logits = self.arbiter(bag_rep)

        return {
            "final": final_logits,
            "recall": recall_logits,
            "precision": precision_logits,
            "att_recall": recall_att,
            "att_precision": precision_att
        }
class PR_ABMIL(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_dim=512,
            dropout=0.25,
            dropout_attn=0.0,
            patch_dropout=0.10,  # NEW: 10% patch dropout
            permute=True  # NEW: permute every forward
    ):
        super().__init__()
        self.patch_dropout = patch_dropout
        self.permute = permute

        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_attn),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None, return_attention=False):
        """
        x: (B, N, D)
        mask: (B, N) — 1=real, 0=pad
        """
        B, N, D = x.shape

        # === 1. RANDOM PERMUTATION (per epoch) ===
        if self.permute and self.training:
            perm = torch.randperm(N, device=x.device)
            x = x[:, perm]
            if mask is not None:
                mask = mask[:, perm]

        # === 2. PATCH DROPOUT (only on real patches) ===
        if self.training and self.patch_dropout > 0.0 and mask is not None:
            # Dropout mask: only on real patches
            keep = torch.rand(B, N, device=x.device) > self.patch_dropout
            keep = keep & (mask == 1)  # don't drop padded
            keep[:, 0] = True  # keep at least one

            # Apply dropout
            x = x * keep.unsqueeze(-1).float()
            # Update mask for attention
            mask = mask.float()
            mask = mask * keep.float()
            mask = mask.bool()

        # === 3. ABMIL ATTENTION ===
        att = self.attention(x).squeeze(-1)  # (B, N)

        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(att, dim=1)
        bag_rep = (x * att.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(bag_rep)
        logits = logits.squeeze(-1)
        if return_attention:
            return logits, att
        return logits


# In ABMIL model — modify forward to use mask
class ABMIL(nn.Module):
    def __init__(self, input_size, hidden_dim=512, dropout=0.25, dropout_attn=0.0):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.Tanh(), # or ReLU?  nn.Tanh()
            nn.Dropout(dropout_attn),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None, return_attention=False):
        # x: (B, N, D), mask: (B, N)
        att = self.attention(x).squeeze(-1)  # (B, N)

        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(att, dim=1)  # (B, N)
        bag_rep = (x * att.unsqueeze(-1)).sum(dim=1)  # (B, D)
        logits = self.classifier(bag_rep)
        logits = logits.squeeze(-1)
        if return_attention:
            return logits, att
        return logits

class ABMIL_standard(torch.nn.Module):
    def __init__(self, iinput_size, emb_dim, att_dim):
        super().__init__()

        # Feature extractor
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(iinput_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, emb_dim),
        )

        self.fc1 = torch.nn.Linear(emb_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1)

        self.classifier = torch.nn.Linear(emb_dim, 1)

    def forward(self, X, mask, return_att=False):
        X = self.mlp(X)  # (batch_size, bag_size, emb_dim)
        H = torch.tanh(self.fc1(X))  # (batch_size, bag_size, att_dim)
        att = torch.sigmoid(self.fc2(H))  # (batch_size, bag_size, 1)
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att_s = torch.softmax(att)  # (batch_size, bag_size, 1)
        # att_s = torch.nn.functional.softmax(att, dim=1)
        X = torch.bmm(att_s.transpose(1, 2), X).squeeze(1)  # (batch_size, emb_dim)
        y = self.classifier(X).squeeze(1)  # (batch_size,)
        if return_att:
            return y, att_s
        else:
            return y
