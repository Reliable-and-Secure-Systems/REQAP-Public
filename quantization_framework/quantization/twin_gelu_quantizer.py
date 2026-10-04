"""
Twin uniform quantizer for post-GELU activations.

Reference: PTQ4ViT (Yuan et al., ECCV 2022, arXiv:2111.12293) — twin uniform
quantization. Implementation details below verified directly against
PTQ4ViT's actual public source (quant_layers/linear.py,
PostGeluPTQSLQuantLinear / PostGeluPTQSLBatchingQuantLinear), not just the
paper text — this session previously lost real time to implementations
built from paraphrased descriptions that turned out to deviate from the
reference in load-bearing ways (see log_quantizer.py's history).

Why post-GELU activations need a different quantizer than the rest of the
network: GELU's output is severely asymmetric. The positive side is
UNBOUNDED (GELU(x) -> x as x -> +inf, so large positive inputs produce
large positive outputs with no fixed ceiling) while the negative side is
BOUNDED BELOW at a fixed constant (GELU has a minimum value of
approximately -0.1700 at x ~ -0.7517 — a property of the GELU function
itself, not of the data or model). A single shared linear scale across both
sides wastes resolution: sized to cover the unbounded positive tail, it
allocates far more range than the negative side ever needs.

Twin uniform quantization splits at exactly zero and gives each side its
own independently-calibrated scale:

  - Positive side: q_max_pos = 2^(bits-1) levels, scale calibrated PER
    CHANNEL from that channel's own max positive value:
        scale_pos = channel_max_positive / (q_max_pos - 0.5)
  - Negative side: q_max_neg = 2^(bits-1) levels, scale is a FIXED
    CONSTANT (not calibrated from data at all), since GELU's negative
    bound is a fixed mathematical property, always the same regardless of
    channel/model/data:
        scale_neg = GELU_NEG_BOUND / q_max_neg

Using bits-1 levels per side (rather than bits levels shared across the
whole range) means the total representable-value count is 2^bits, same
budget as a normal bits-wide quantizer — split into two independently-
scaled halves instead of one shared linear scale. This exactly matches
PTQ4ViT's reference implementation; the only deliberate departure is
granularity: PTQ4ViT calibrates one scale per tensor/chunk (layer-wise or
structured), this implementation calibrates PER CHANNEL, matching this
codebase's existing PerChannelActivationQuantizer and the channel-wise
mixed-precision search's per-channel bit assignment — an architectural
adaptation, not a change to the actual quantization math.
"""

import torch
import torch.nn as nn


class TwinGeluQuantizer(nn.Module):
    """
    Per-channel twin uniform quantizer for post-GELU activations.

    Args:
        channel_bits: list of int, one bit-width per channel. Each channel
                      gets (bits-1) levels per side (positive/negative).
        channel_dim:  index of the channel dimension in the forward tensor.
                      Use 1 for Conv2d (B,C,H,W), -1 (last) for Linear
                      variants — same convention as
                      PerChannelActivationQuantizer.
        eps:          numerical floor to avoid division by zero when a
                      channel's positive-side calibration data is all <= 0.
    """

    # -min(GELU(x)) at x ~ -0.7517 — a fixed mathematical property of the
    # GELU function, independent of data/model. Verified against PTQ4ViT's
    # actual source (quant_layers/linear.py): a_neg_interval is a hardcoded
    # constant, not calibrated, precisely because this bound never changes.
    GELU_NEG_BOUND = 0.16997124254703522

    def __init__(self, channel_bits: list, channel_dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.channel_bits = channel_bits
        self.channel_dim = channel_dim
        self.eps = eps
        self.num_channels = len(channel_bits)
        self.calibrated = False

        q_max_pos = [2 ** (b - 1) for b in channel_bits]
        self.register_buffer('q_max_pos', torch.tensor(q_max_pos, dtype=torch.float32))
        # Negative side uses the same per-channel level count, but its
        # scale is fixed (see GELU_NEG_BOUND above), computed once here —
        # never recalibrated from data.
        neg_scale = [self.GELU_NEG_BOUND / q for q in q_max_pos]
        self.register_buffer('scale_neg', torch.tensor(neg_scale, dtype=torch.float32))
        # Positive-side scale: calibrated per channel, one-shot (see
        # calibrate()). Initialized to 1.0, overwritten on first
        # training-mode forward call.
        self.register_buffer('scale_pos', torch.ones(self.num_channels))

    def calibrate(self, x):
        """
        One-shot per-channel positive-side calibration:
        scale_pos = channel_max_positive / (q_max_pos - 0.5)
        Matches PTQ4ViT's reference formula exactly (x.max() / (a_qmax -
        0.5)), applied per channel instead of per tensor/chunk.
        """
        device = x.device
        d = self.channel_dim % x.dim()
        x_perm = x.detach().moveaxis(d, 0).reshape(self.num_channels, -1).float()

        channel_max_pos = x_perm.clamp(min=0.0).max(dim=1).values
        scale_pos = channel_max_pos / (self.q_max_pos.to(device) - 0.5)
        self.scale_pos.copy_(scale_pos.clamp(min=self.eps).to(self.scale_pos.device))
        self.calibrated = True

    def forward(self, x):
        device = x.device
        if self.q_max_pos.device != device:
            self.q_max_pos = self.q_max_pos.to(device)
            self.scale_neg = self.scale_neg.to(device)
            self.scale_pos = self.scale_pos.to(device)

        d = self.channel_dim % x.dim()
        num_channels_actual = x.shape[d]
        if num_channels_actual != self.num_channels:
            raise RuntimeError(
                f'\nTwinGeluQuantizer: channel count mismatch.\n'
                f'  Expected : {self.num_channels} channels (from channel_bits)\n'
                f'  Got      : {num_channels_actual} at dim {d} of tensor shape {tuple(x.shape)}\n'
                f'  channel_dim was set to {self.channel_dim}\n'
            )

        if self.training and not self.calibrated:
            self.calibrate(x)

        view_shape = [1] * x.dim()
        view_shape[d] = self.num_channels
        q_max_pos = self.q_max_pos.view(view_shape)
        scale_pos = self.scale_pos.view(view_shape)
        scale_neg = self.scale_neg.view(view_shape)

        x_f32 = x.float()

        # Positive side: [0, q_max_pos - 1], scale = calibrated per channel.
        x_pos_idx = torch.round(x_f32 / scale_pos)
        x_pos_idx = torch.minimum(torch.maximum(x_pos_idx, torch.zeros_like(x_pos_idx)),
                                  q_max_pos - 1)
        x_pos = x_pos_idx * scale_pos

        # Negative side: [-q_max_pos, 0], scale = fixed GELU-bound constant.
        x_neg_idx = torch.round(x_f32 / scale_neg)
        x_neg_idx = torch.minimum(torch.maximum(x_neg_idx, -q_max_pos),
                                  torch.zeros_like(x_neg_idx))
        x_neg = x_neg_idx * scale_neg

        # Values >= 0 use the positive branch, values < 0 use the negative
        # branch — matches PTQ4ViT's split-at-zero exactly (clamp ranges
        # above already ensure x_pos=0 contribution for negative inputs and
        # vice versa, but select explicitly for clarity and to avoid
        # relying on clamp-to-zero coincidence).
        is_positive = (x_f32 >= 0).float()
        x_dq = x_pos * is_positive + x_neg * (1.0 - is_positive)

        return x_dq.to(x.dtype)

    def extra_repr(self):
        bits_summary = {}
        for b in self.channel_bits:
            bits_summary[b] = bits_summary.get(b, 0) + 1
        return (f"channels={self.num_channels}, channel_dim={self.channel_dim}, "
                f"bits_dist={bits_summary}, gelu_neg_bound={self.GELU_NEG_BOUND}")
