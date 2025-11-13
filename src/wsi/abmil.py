import torch
import torch.nn as nn
from torch import masked_softmax

# In ABMIL model — modify forward to use mask


class ABMIL_Grok(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        # ... (same as torchmil)
        self.attention = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, x, mask=None, return_attention=False):
        # x: (B, N, D), mask: (B, N)
        att = self.attention(x).squeeze(-1)  # (B, N)

        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(att, dim=1)  # (B, N)
        bag_rep = (x * att.unsqueeze(-1)).sum(dim=1)  # (B, D)
        logits = self.classifier(bag_rep)

        if return_attention:
            return logits, att
        return logits

class ABMIL(torch.nn.Module):
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
        att_s = masked_softmax(att, mask)  # (batch_size, bag_size, 1)
        # att_s = torch.nn.functional.softmax(att, dim=1)
        X = torch.bmm(att_s.transpose(1, 2), X).squeeze(1)  # (batch_size, emb_dim)
        y = self.classifier(X).squeeze(1)  # (batch_size,)
        if return_att:
            return y, att_s
        else:
            return y


model = ABMIL(emb_dim=256, att_dim=128)
print(model)