"""
Ground-truth BOPs (bit-operations) computation, shared by both the
CIFAR/GTSRB and ImageNet pipelines. Replaces the three independent
hand-rolled MAC counters (validate_config_granular.py, validate_config_v2.py,
joint_granular_search_v2.py's compute_corrected_bops) that undercounted
MaxPool-downsampled convs, ignored the Linear token dimension on
transformers, and weighted the Phase-2 split formula by channel count
instead of MACs. See REMEDIATION_PLAN.md Phase A1 / A2.

layer_macs and other_macs come from profile_macs.py (CIFAR/GTSRB) or its
ImageNet counterpart — real per-layer MAC counts from
torch.utils.flop_counter.FlopCounterMode, not hand-rolled spatial tracking.
"""


def compute_bops(layer_macs, other_macs, weight_config, activation_config):
    """
    layer_macs: {layer_name: total_macs} for each quantized Conv2d/Linear
        leaf, from profile_macs.py.
    other_macs: MACs not attributed to any quantized leaf (e.g. attention
        QK^T / softmax@V matmuls) — always executes at FP32.
    weight_config: {layer_name: [bits_per_output_channel, ...]}.
    activation_config: {layer_name: [bits_per_output_channel, ...]} — a
        layer present here is a W=A layer (weight and activation share the
        per-channel bit-width); a layer in weight_config but ABSENT from
        activation_config is a Phase-2 rollback layer (weight-only,
        activation stays FP32).

    Returns dict with quantized_bops, fp32_bops, reduction, and
    fp32_act_mac_fraction (MAC-weighted fraction of the model executing
    with FP32 activations — Phase-2 layers + other_macs).
    """
    total_macs = sum(layer_macs.values()) + other_macs
    fp32_bops = total_macs * 32 * 32

    quantized_bops = 0.0
    fp32_act_macs = other_macs  # attention matmuls etc. are always FP32-activation

    for layer_name, bits_per_channel in weight_config.items():
        macs = layer_macs.get(layer_name)
        if macs is None:
            raise KeyError(f'{layer_name!r} in weight_config has no MAC count '
                            f'in layer_macs — profile_macs.py output is stale '
                            f'or from a different model architecture')
        n_ch = len(bits_per_channel)
        macs_per_channel = macs / n_ch
        is_wa_layer = layer_name in activation_config

        if is_wa_layer:
            act_bits_per_channel = activation_config[layer_name]
            for w_bits, a_bits in zip(bits_per_channel, act_bits_per_channel):
                quantized_bops += macs_per_channel * w_bits * a_bits
        else:
            fp32_act_macs += macs
            for w_bits in bits_per_channel:
                quantized_bops += macs_per_channel * w_bits * 32

    quantized_bops += other_macs * 32 * 32

    return {
        'total_macs': total_macs,
        'fp32_bops': fp32_bops,
        'quantized_bops': quantized_bops,
        'reduction': fp32_bops / quantized_bops if quantized_bops else float('inf'),
        'fp32_act_mac_fraction': fp32_act_macs / total_macs if total_macs else 0.0,
    }
