import torch
import torch.nn as nn

class ActivationQuantizer(nn.Module):
    """
    Module to quantize activations during Forward Pass.
    Used for QAT (Fake Quantization) and Calibration.
    """
    def __init__(self, bit_width=8, method='asymmetric', momentum=0.9):
        super().__init__()
        self.bit_width = bit_width
        self.method = method
        self.momentum = momentum
        
        # State buffers
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
        
        # Flags
        self.initialized = False
        
    def forward(self, x):
        # CRITICAL: Ensure all buffers are on the same device as input
        target_device = x.device
        if self.running_min.device != target_device:
            self.running_min = self.running_min.to(target_device)
            self.running_max = self.running_max.to(target_device)
            self.scale = self.scale.to(target_device)
            self.zero_point = self.zero_point.to(target_device)
        
        if self.training:
            # Update ranges
            current_min = x.detach().min()
            current_max = x.detach().max()
            
            if not self.initialized:
                self.running_min.copy_(current_min)
                self.running_max.copy_(current_max)
                self.initialized = True
            else:
                self.running_min.mul_(self.momentum).add_(current_min * (1 - self.momentum))
                self.running_max.mul_(self.momentum).add_(current_max * (1 - self.momentum))
        
        # Calculate Scale/ZP based on running stats
        q_min = 0
        q_max = 2 ** self.bit_width - 1
        
        scale = (self.running_max - self.running_min) / (q_max - q_min)
        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
        
        zero_point = -self.running_min / scale
        zero_point = torch.round(zero_point).clamp(q_min, q_max)
        
        # Update buffers in-place to preserve registration and device
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        
        # Fake Quantize using the registered buffers
        # x_int = round(x / s + z)
        # x_dequant = (x_int - z) * s
        
        x_int = torch.round(x / self.scale + self.zero_point).clamp(q_min, q_max)
        x_dq = (x_int - self.zero_point) * self.scale
        
        return x_dq

    def extra_repr(self):
        return f"bit_width={self.bit_width}, method={self.method}"


class PerChannelActivationQuantizer(nn.Module):
    """
    Per-output-channel activation quantizer for filter-level W=A co-optimization.

    Each output channel has its own bit-width and calibrated min/max range.

    Calibration uses **percentile clipping** instead of raw min/max to handle
    outlier activations.  Transformer attention layers produce activations with
    extreme outliers (a few values 50-100x larger than the median).  With raw
    min/max, the quantization range spans this huge interval and most of the
    quantization levels are wasted on empty space — effectively reducing
    precision to 1-2 bits for the bulk of values.  Percentile clipping
    (default 99.99th) sets the range where the data actually lives, giving
    true N-bit precision at the cost of clipping rare outliers.  This is the
    standard approach in production quantization frameworks (TensorRT, ONNX
    Runtime, PyTorch FX quantization).

    channel_dim must be supplied by the caller (set at insertion time, not inferred
    from tensor shape at runtime).  This makes the quantizer model-agnostic:

        Conv1d/2d/3d, ConvTranspose* → channel_dim=1   output: (B, C, ...)
        Linear (any rank)            → channel_dim=-1  output: (B, C) | (B, N, C) | (B, H, W, C)
                                        channels are always at the last dim for Linear
    """

    def __init__(self, channel_bits: list, channel_dim: int = -1,
                 percentile: float = 99.99):
        """
        Args:
            channel_bits: list of int, one bit-width per output channel.
            channel_dim:  index of the channel dimension in the forward tensor.
                          Use 1 for Conv2d (B,C,H,W), -1 (last) for all Linear
                          variants regardless of how many extra dims they produce.
            percentile:   percentile for range clipping during calibration
                          (default 99.99).  The range is set to the
                          [100-percentile, percentile] interval per channel,
                          clipping extreme outliers that would otherwise waste
                          quantization levels on empty space.
        """
        super().__init__()
        self.channel_bits = channel_bits
        self.channel_dim = channel_dim
        self.percentile = percentile
        num_channels = len(channel_bits)

        self.register_buffer('running_min', torch.zeros(num_channels))
        self.register_buffer('running_max', torch.ones(num_channels))
        self.initialized = False

    def forward(self, x):
        device = x.device
        if self.running_min.device != device:
            self.running_min = self.running_min.to(device)
            self.running_max = self.running_max.to(device)

        # Resolve negative channel_dim to a positive index
        ch_dim = self.channel_dim % x.dim()
        num_channels = x.shape[ch_dim]

        # Sanity check: catch channel_dim mismatches early with an actionable message
        if num_channels != len(self.channel_bits):
            raise RuntimeError(
                f'\nPerChannelActivationQuantizer: channel count mismatch.\n'
                f'  Expected : {len(self.channel_bits)} channels (from channel_bits)\n'
                f'  Got      : {num_channels} at dim {ch_dim} of tensor shape {tuple(x.shape)}\n'
                f'  channel_dim was set to {self.channel_dim}\n'
                f'\nHow to fix:\n'
                f'  1. Find which dim of the tensor holds the {len(self.channel_bits)} channels.\n'
                f'     Tensor shape: {tuple(x.shape)}\n'
                f'  2. If channels are at dim 1 (Conv-style), add the layer type to\n'
                f'     _CONV_TYPES in validate_config_granular.py.\n'
                f'  3. If channels are at the last dim (Linear-style), no change needed\n'
                f'     — check that channel_dim=-1 is being passed correctly.\n'
            )

        if self.training:
            # Move channel dim to front, then flatten everything else
            # so we can compute per-channel statistics without special-casing layouts
            x_perm = x.detach().moveaxis(ch_dim, 0)           # (C, ...)
            x_flat = x_perm.reshape(num_channels, -1)          # (C, B*H*W*...)

            # Percentile clipping: use the [lo, hi] percentile range instead of
            # raw min/max.  This prevents outlier activations (common in
            # transformer attention layers) from stretching the quantization
            # range and wasting levels on empty space.
            hi = self.percentile / 100.0    # e.g. 0.9999
            lo = 1.0 - hi                   # e.g. 0.0001
            ch_min = torch.quantile(x_flat.float(), lo, dim=1)
            ch_max = torch.quantile(x_flat.float(), hi, dim=1)

            if not self.initialized:
                self.running_min.copy_(ch_min)
                self.running_max.copy_(ch_max)
                self.initialized = True
            else:
                momentum = 0.9
                self.running_min.mul_(momentum).add_(ch_min * (1 - momentum))
                self.running_max.mul_(momentum).add_(ch_max * (1 - momentum))

        # Per-channel asymmetric fake-quantization — fully vectorised, no Python loop.
        # q_min is always 0 (asymmetric); q_max varies per channel based on bit-width.
        bits_t = torch.tensor(self.channel_bits, dtype=torch.float32, device=device)
        q_max = (2.0 ** bits_t - 1.0)                          # (C,)

        scale = (self.running_max - self.running_min) / q_max   # (C,)
        scale = scale.clamp(min=1e-8)
        zp = (-self.running_min / scale).round()
        zp = torch.minimum(torch.maximum(zp, torch.zeros_like(zp)), q_max)  # clamp(0, q_max)

        # Reshape (C,) vectors for broadcasting against x at ch_dim
        view_shape = [1] * x.dim()
        view_shape[ch_dim] = num_channels
        scale  = scale.view(view_shape)
        zp     = zp.view(view_shape)
        q_max  = q_max.view(view_shape)

        x_int = (x / scale + zp).round()
        x_int = torch.maximum(x_int, torch.zeros_like(x_int))  # clamp min=0
        x_int = torch.minimum(x_int, q_max)                    # clamp max=q_max (per-channel)
        return (x_int - zp) * scale

    def extra_repr(self):
        bits_summary = {}
        for b in self.channel_bits:
            bits_summary[b] = bits_summary.get(b, 0) + 1
        return (f"channels={len(self.channel_bits)}, "
                f"channel_dim={self.channel_dim}, "
                f"percentile={self.percentile}, bits_dist={bits_summary}")
