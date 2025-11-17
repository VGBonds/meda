import torch
import torch.nn as nn

class SpatialABMIL(nn.Module):
    def __init__(self, feat_dim=1024, coord_dim=2, hidden_dim=256):
        super().__init__()

        # 1. Spatial self-attention (coords → context)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=coord_dim, num_heads=1, batch_first=True
        )
        self.coord_proj = nn.Linear(coord_dim, feat_dim)  # lift coords

        # 2. Feature fusion
        self.fusion = nn.Linear(feat_dim * 2, feat_dim)

        # 3. Classic ABMIL (interpretable)
        self.abmil_attn = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, x, coords, mask=None, return_attention=False):
        # x: (B, N, D), coords: (B, N, 2), mask: (B, N)

        # 1. Spatial context
        coords_lifted = self.coord_proj(coords)  # (B, N, D)
        spatial_ctx, _ = self.spatial_attn(coords_lifted, coords_lifted, coords_lifted)

        # 2. Fuse with patch features
        x_fused = torch.cat([x, spatial_ctx], dim=-1)  # (B, N, 2D)
        x_fused = self.fusion(x_fused)  # (B, N, D)

        # 3. ABMIL attention (on fused features)
        att_raw = self.abmil_attn(x_fused).squeeze(-1)  # (B, N)
        if mask is not None:
            att_raw = att_raw.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att_raw, dim=1)

        bag_rep = (x_fused * att.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(bag_rep)

        if return_attention:
            return logits, att
        return logits