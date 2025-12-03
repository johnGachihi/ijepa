import torch


class DownsampleProjection(torch.nn.Module):
  def __init__(self, in_dim, embed_dim, bottleneck_dim=128):
    super().__init__()

    self.in_dim = in_dim
    self.embed_dim = embed_dim
    self.bottleneck_dim = bottleneck_dim

    self.mlp = torch.nn.Sequential(
      # torch.nn.Linear(in_dim, in_dim),
      # torch.nn.GELU(),
      torch.nn.Linear(in_dim, bottleneck_dim),
      torch.nn.LayerNorm(bottleneck_dim),
      torch.nn.Linear(bottleneck_dim, embed_dim)
    )

  def _init_weights(self):
    """Initialize weights using truncated normal."""
    for m in self.modules():
      if isinstance(m, torch.nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
          torch.nn.init.constant_(m.bias, 0)
      elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

  def forward(self, x):
    return self.mlp(x)
