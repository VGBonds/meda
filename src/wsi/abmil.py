import torch
import torch.nn as nn

# In ABMIL model — modify forward to use mask


class ABMIL(nn.Module):
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