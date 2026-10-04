/* Auto-generated Complete Swin-T Pipeline (Packed MQF) */
int8_t *unpacked_fm = fm_B;
int8_t *packed_fm = fm_A;
int8_t *tmp_ptr;

int8_t q_buf[4096];
int8_t k_buf[4096];
int8_t v_buf[4096];
int8_t attn_out[4096];
int8_t tmp_window_buf[30000];
int32_t dummy_rel_pos[4096] = {0};

// Initial state: unpacked_fm has the image.
for (int i=0; i<3072; i++) unpacked_fm[i] = cifar_image_0[i];

// patch_embed.proj
run_conv2d_8bit(unpacked_fm, packed_fm, patch_embed_proj_weights,
    32, 32, 3,
    8, 8, 96,
    4, 4, 0, 4,
    patch_embed_proj_scale, patch_embed_proj_bias);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
printf("Finished patch_embed.proj\n");

// --- layers_0_blocks_0 ---
for(int hw=0; hw<64; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*96, packed_fm + hw*96, 96, layers_0_blocks_0_norm1_weight, layers_0_blocks_0_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<64; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*96), packed_fm + hw*288, layers_0_blocks_0_attn_qkv_weights,
        layers_0_blocks_0_attn_qkv_pos, layers_0_blocks_0_attn_qkv_mask, layers_0_blocks_0_attn_qkv_slots, layers_0_blocks_0_attn_qkv_in_act_bits, layers_0_blocks_0_attn_qkv_in_offset_table,
        96, 288,
        layers_0_blocks_0_attn_qkv_scale, layers_0_blocks_0_attn_qkv_bias, layers_0_blocks_0_attn_qkv_MAX_D, layers_0_blocks_0_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<64; hw++) {
    for(int d=0; d<96; d++) {
        q_buf[hw*96 + d] = unpacked_fm[hw*288 + d];
        k_buf[hw*96 + d] = unpacked_fm[hw*288 + 96 + d];
        v_buf[hw*96 + d] = unpacked_fm[hw*288 + 2*96 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 8, 8, 96, 7);
window_partition_8bit(k_buf, unpacked_fm, 8, 8, 96, 7);
window_partition_8bit(v_buf, tmp_window_buf, 8, 8, 96, 7);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*4704, unpacked_fm + nw*4704, tmp_window_buf + nw*4704,
        attn_out + nw*4704, dummy_rel_pos,
        7, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 8, 8, 96, 7);
for(int hw=0; hw<64; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*96), packed_fm + hw*96, layers_0_blocks_0_attn_proj_weights,
        layers_0_blocks_0_attn_proj_pos, layers_0_blocks_0_attn_proj_mask, layers_0_blocks_0_attn_proj_slots, layers_0_blocks_0_attn_proj_in_act_bits, layers_0_blocks_0_attn_proj_in_offset_table,
        96, 96,
        layers_0_blocks_0_attn_proj_scale, layers_0_blocks_0_attn_proj_bias, layers_0_blocks_0_attn_proj_MAX_D, layers_0_blocks_0_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<64; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*96, packed_fm + hw*96, 96, layers_0_blocks_0_norm2_weight, layers_0_blocks_0_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<64; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*96), packed_fm + hw*384, layers_0_blocks_0_mlp_fc1_weights,
        layers_0_blocks_0_mlp_fc1_pos, layers_0_blocks_0_mlp_fc1_mask, layers_0_blocks_0_mlp_fc1_slots, layers_0_blocks_0_mlp_fc1_in_act_bits, layers_0_blocks_0_mlp_fc1_in_offset_table,
        96, 384,
        layers_0_blocks_0_mlp_fc1_scale, layers_0_blocks_0_mlp_fc1_bias, layers_0_blocks_0_mlp_fc1_MAX_D, layers_0_blocks_0_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 24576);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<64; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*96, layers_0_blocks_0_mlp_fc2_weights,
        layers_0_blocks_0_mlp_fc2_pos, layers_0_blocks_0_mlp_fc2_mask, layers_0_blocks_0_mlp_fc2_slots, layers_0_blocks_0_mlp_fc2_in_act_bits, layers_0_blocks_0_mlp_fc2_in_offset_table,
        384, 96,
        layers_0_blocks_0_mlp_fc2_scale, layers_0_blocks_0_mlp_fc2_bias, layers_0_blocks_0_mlp_fc2_MAX_D, layers_0_blocks_0_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_0_blocks_1 ---
for(int hw=0; hw<64; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*96, packed_fm + hw*96, 96, layers_0_blocks_1_norm1_weight, layers_0_blocks_1_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<64; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*96), packed_fm + hw*288, layers_0_blocks_1_attn_qkv_weights,
        layers_0_blocks_1_attn_qkv_pos, layers_0_blocks_1_attn_qkv_mask, layers_0_blocks_1_attn_qkv_slots, layers_0_blocks_1_attn_qkv_in_act_bits, layers_0_blocks_1_attn_qkv_in_offset_table,
        96, 288,
        layers_0_blocks_1_attn_qkv_scale, layers_0_blocks_1_attn_qkv_bias, layers_0_blocks_1_attn_qkv_MAX_D, layers_0_blocks_1_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 8, 8, 288, 3);
for(int i=0; i<18432; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<64; hw++) {
    for(int d=0; d<96; d++) {
        q_buf[hw*96 + d] = unpacked_fm[hw*288 + d];
        k_buf[hw*96 + d] = unpacked_fm[hw*288 + 96 + d];
        v_buf[hw*96 + d] = unpacked_fm[hw*288 + 2*96 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 8, 8, 96, 7);
window_partition_8bit(k_buf, unpacked_fm, 8, 8, 96, 7);
window_partition_8bit(v_buf, tmp_window_buf, 8, 8, 96, 7);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*4704, unpacked_fm + nw*4704, tmp_window_buf + nw*4704,
        attn_out + nw*4704, dummy_rel_pos,
        7, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 8, 8, 96, 7);
reverse_cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 8, 8, 96, 3);
for(int i=0; i<6144; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<64; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*96), packed_fm + hw*96, layers_0_blocks_1_attn_proj_weights,
        layers_0_blocks_1_attn_proj_pos, layers_0_blocks_1_attn_proj_mask, layers_0_blocks_1_attn_proj_slots, layers_0_blocks_1_attn_proj_in_act_bits, layers_0_blocks_1_attn_proj_in_offset_table,
        96, 96,
        layers_0_blocks_1_attn_proj_scale, layers_0_blocks_1_attn_proj_bias, layers_0_blocks_1_attn_proj_MAX_D, layers_0_blocks_1_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<64; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*96, packed_fm + hw*96, 96, layers_0_blocks_1_norm2_weight, layers_0_blocks_1_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<64; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*96), packed_fm + hw*384, layers_0_blocks_1_mlp_fc1_weights,
        layers_0_blocks_1_mlp_fc1_pos, layers_0_blocks_1_mlp_fc1_mask, layers_0_blocks_1_mlp_fc1_slots, layers_0_blocks_1_mlp_fc1_in_act_bits, layers_0_blocks_1_mlp_fc1_in_offset_table,
        96, 384,
        layers_0_blocks_1_mlp_fc1_scale, layers_0_blocks_1_mlp_fc1_bias, layers_0_blocks_1_mlp_fc1_MAX_D, layers_0_blocks_1_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 24576);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<64; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*96, layers_0_blocks_1_mlp_fc2_weights,
        layers_0_blocks_1_mlp_fc2_pos, layers_0_blocks_1_mlp_fc2_mask, layers_0_blocks_1_mlp_fc2_slots, layers_0_blocks_1_mlp_fc2_in_act_bits, layers_0_blocks_1_mlp_fc2_in_offset_table,
        384, 96,
        layers_0_blocks_1_mlp_fc2_scale, layers_0_blocks_1_mlp_fc2_bias, layers_0_blocks_1_mlp_fc2_MAX_D, layers_0_blocks_1_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- Patch Merging Stage 0 -> 1 ---
patch_merging_8bit(unpacked_fm, packed_fm, 8, 8, 96);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_1_downsample_norm_weight, layers_1_downsample_norm_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*192, layers_1_downsample_reduction_weights,
        layers_1_downsample_reduction_pos, layers_1_downsample_reduction_mask, layers_1_downsample_reduction_slots, layers_1_downsample_reduction_in_act_bits, layers_1_downsample_reduction_in_offset_table,
        384, 192,
        layers_1_downsample_reduction_scale, layers_1_downsample_reduction_bias, layers_1_downsample_reduction_MAX_D, layers_1_downsample_reduction_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_1_blocks_0 ---
for(int hw=0; hw<16; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*192, packed_fm + hw*192, 192, layers_1_blocks_0_norm1_weight, layers_1_blocks_0_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*192), packed_fm + hw*576, layers_1_blocks_0_attn_qkv_weights,
        layers_1_blocks_0_attn_qkv_pos, layers_1_blocks_0_attn_qkv_mask, layers_1_blocks_0_attn_qkv_slots, layers_1_blocks_0_attn_qkv_in_act_bits, layers_1_blocks_0_attn_qkv_in_offset_table,
        192, 576,
        layers_1_blocks_0_attn_qkv_scale, layers_1_blocks_0_attn_qkv_bias, layers_1_blocks_0_attn_qkv_MAX_D, layers_1_blocks_0_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    for(int d=0; d<192; d++) {
        q_buf[hw*192 + d] = unpacked_fm[hw*576 + d];
        k_buf[hw*192 + d] = unpacked_fm[hw*576 + 192 + d];
        v_buf[hw*192 + d] = unpacked_fm[hw*576 + 2*192 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 4, 4, 192, 4);
window_partition_8bit(k_buf, unpacked_fm, 4, 4, 192, 4);
window_partition_8bit(v_buf, tmp_window_buf, 4, 4, 192, 4);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*3072, unpacked_fm + nw*3072, tmp_window_buf + nw*3072,
        attn_out + nw*3072, dummy_rel_pos,
        4, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 4, 4, 192, 4);
for(int hw=0; hw<16; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*192), packed_fm + hw*192, layers_1_blocks_0_attn_proj_weights,
        layers_1_blocks_0_attn_proj_pos, layers_1_blocks_0_attn_proj_mask, layers_1_blocks_0_attn_proj_slots, layers_1_blocks_0_attn_proj_in_act_bits, layers_1_blocks_0_attn_proj_in_offset_table,
        192, 192,
        layers_1_blocks_0_attn_proj_scale, layers_1_blocks_0_attn_proj_bias, layers_1_blocks_0_attn_proj_MAX_D, layers_1_blocks_0_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*192, packed_fm + hw*192, 192, layers_1_blocks_0_norm2_weight, layers_1_blocks_0_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*192), packed_fm + hw*768, layers_1_blocks_0_mlp_fc1_weights,
        layers_1_blocks_0_mlp_fc1_pos, layers_1_blocks_0_mlp_fc1_mask, layers_1_blocks_0_mlp_fc1_slots, layers_1_blocks_0_mlp_fc1_in_act_bits, layers_1_blocks_0_mlp_fc1_in_offset_table,
        192, 768,
        layers_1_blocks_0_mlp_fc1_scale, layers_1_blocks_0_mlp_fc1_bias, layers_1_blocks_0_mlp_fc1_MAX_D, layers_1_blocks_0_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 12288);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*768), packed_fm + hw*192, layers_1_blocks_0_mlp_fc2_weights,
        layers_1_blocks_0_mlp_fc2_pos, layers_1_blocks_0_mlp_fc2_mask, layers_1_blocks_0_mlp_fc2_slots, layers_1_blocks_0_mlp_fc2_in_act_bits, layers_1_blocks_0_mlp_fc2_in_offset_table,
        768, 192,
        layers_1_blocks_0_mlp_fc2_scale, layers_1_blocks_0_mlp_fc2_bias, layers_1_blocks_0_mlp_fc2_MAX_D, layers_1_blocks_0_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_1_blocks_1 ---
for(int hw=0; hw<16; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*192, packed_fm + hw*192, 192, layers_1_blocks_1_norm1_weight, layers_1_blocks_1_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*192), packed_fm + hw*576, layers_1_blocks_1_attn_qkv_weights,
        layers_1_blocks_1_attn_qkv_pos, layers_1_blocks_1_attn_qkv_mask, layers_1_blocks_1_attn_qkv_slots, layers_1_blocks_1_attn_qkv_in_act_bits, layers_1_blocks_1_attn_qkv_in_offset_table,
        192, 576,
        layers_1_blocks_1_attn_qkv_scale, layers_1_blocks_1_attn_qkv_bias, layers_1_blocks_1_attn_qkv_MAX_D, layers_1_blocks_1_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 4, 4, 576, 2);
for(int i=0; i<9216; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<16; hw++) {
    for(int d=0; d<192; d++) {
        q_buf[hw*192 + d] = unpacked_fm[hw*576 + d];
        k_buf[hw*192 + d] = unpacked_fm[hw*576 + 192 + d];
        v_buf[hw*192 + d] = unpacked_fm[hw*576 + 2*192 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 4, 4, 192, 4);
window_partition_8bit(k_buf, unpacked_fm, 4, 4, 192, 4);
window_partition_8bit(v_buf, tmp_window_buf, 4, 4, 192, 4);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*3072, unpacked_fm + nw*3072, tmp_window_buf + nw*3072,
        attn_out + nw*3072, dummy_rel_pos,
        4, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 4, 4, 192, 4);
reverse_cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 4, 4, 192, 2);
for(int i=0; i<3072; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<16; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*192), packed_fm + hw*192, layers_1_blocks_1_attn_proj_weights,
        layers_1_blocks_1_attn_proj_pos, layers_1_blocks_1_attn_proj_mask, layers_1_blocks_1_attn_proj_slots, layers_1_blocks_1_attn_proj_in_act_bits, layers_1_blocks_1_attn_proj_in_offset_table,
        192, 192,
        layers_1_blocks_1_attn_proj_scale, layers_1_blocks_1_attn_proj_bias, layers_1_blocks_1_attn_proj_MAX_D, layers_1_blocks_1_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*192, packed_fm + hw*192, 192, layers_1_blocks_1_norm2_weight, layers_1_blocks_1_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*192), packed_fm + hw*768, layers_1_blocks_1_mlp_fc1_weights,
        layers_1_blocks_1_mlp_fc1_pos, layers_1_blocks_1_mlp_fc1_mask, layers_1_blocks_1_mlp_fc1_slots, layers_1_blocks_1_mlp_fc1_in_act_bits, layers_1_blocks_1_mlp_fc1_in_offset_table,
        192, 768,
        layers_1_blocks_1_mlp_fc1_scale, layers_1_blocks_1_mlp_fc1_bias, layers_1_blocks_1_mlp_fc1_MAX_D, layers_1_blocks_1_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 12288);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<16; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*768), packed_fm + hw*192, layers_1_blocks_1_mlp_fc2_weights,
        layers_1_blocks_1_mlp_fc2_pos, layers_1_blocks_1_mlp_fc2_mask, layers_1_blocks_1_mlp_fc2_slots, layers_1_blocks_1_mlp_fc2_in_act_bits, layers_1_blocks_1_mlp_fc2_in_offset_table,
        768, 192,
        layers_1_blocks_1_mlp_fc2_scale, layers_1_blocks_1_mlp_fc2_bias, layers_1_blocks_1_mlp_fc2_MAX_D, layers_1_blocks_1_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- Patch Merging Stage 1 -> 2 ---
patch_merging_8bit(unpacked_fm, packed_fm, 4, 4, 192);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*768, packed_fm + hw*768, 768, layers_2_downsample_norm_weight, layers_2_downsample_norm_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*768), packed_fm + hw*384, layers_2_downsample_reduction_weights,
        layers_2_downsample_reduction_pos, layers_2_downsample_reduction_mask, layers_2_downsample_reduction_slots, layers_2_downsample_reduction_in_act_bits, layers_2_downsample_reduction_in_offset_table,
        768, 384,
        layers_2_downsample_reduction_scale, layers_2_downsample_reduction_bias, layers_2_downsample_reduction_MAX_D, layers_2_downsample_reduction_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_2_blocks_0 ---
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_0_norm1_weight, layers_2_blocks_0_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1152, layers_2_blocks_0_attn_qkv_weights,
        layers_2_blocks_0_attn_qkv_pos, layers_2_blocks_0_attn_qkv_mask, layers_2_blocks_0_attn_qkv_slots, layers_2_blocks_0_attn_qkv_in_act_bits, layers_2_blocks_0_attn_qkv_in_offset_table,
        384, 1152,
        layers_2_blocks_0_attn_qkv_scale, layers_2_blocks_0_attn_qkv_bias, layers_2_blocks_0_attn_qkv_MAX_D, layers_2_blocks_0_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    for(int d=0; d<384; d++) {
        q_buf[hw*384 + d] = unpacked_fm[hw*1152 + d];
        k_buf[hw*384 + d] = unpacked_fm[hw*1152 + 384 + d];
        v_buf[hw*384 + d] = unpacked_fm[hw*1152 + 2*384 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 2, 2, 384, 2);
window_partition_8bit(k_buf, unpacked_fm, 2, 2, 384, 2);
window_partition_8bit(v_buf, tmp_window_buf, 2, 2, 384, 2);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*1536, unpacked_fm + nw*1536, tmp_window_buf + nw*1536,
        attn_out + nw*1536, dummy_rel_pos,
        2, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 2, 2, 384, 2);
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*384, layers_2_blocks_0_attn_proj_weights,
        layers_2_blocks_0_attn_proj_pos, layers_2_blocks_0_attn_proj_mask, layers_2_blocks_0_attn_proj_slots, layers_2_blocks_0_attn_proj_in_act_bits, layers_2_blocks_0_attn_proj_in_offset_table,
        384, 384,
        layers_2_blocks_0_attn_proj_scale, layers_2_blocks_0_attn_proj_bias, layers_2_blocks_0_attn_proj_MAX_D, layers_2_blocks_0_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_0_norm2_weight, layers_2_blocks_0_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1536, layers_2_blocks_0_mlp_fc1_weights,
        layers_2_blocks_0_mlp_fc1_pos, layers_2_blocks_0_mlp_fc1_mask, layers_2_blocks_0_mlp_fc1_slots, layers_2_blocks_0_mlp_fc1_in_act_bits, layers_2_blocks_0_mlp_fc1_in_offset_table,
        384, 1536,
        layers_2_blocks_0_mlp_fc1_scale, layers_2_blocks_0_mlp_fc1_bias, layers_2_blocks_0_mlp_fc1_MAX_D, layers_2_blocks_0_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 6144);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*1536), packed_fm + hw*384, layers_2_blocks_0_mlp_fc2_weights,
        layers_2_blocks_0_mlp_fc2_pos, layers_2_blocks_0_mlp_fc2_mask, layers_2_blocks_0_mlp_fc2_slots, layers_2_blocks_0_mlp_fc2_in_act_bits, layers_2_blocks_0_mlp_fc2_in_offset_table,
        1536, 384,
        layers_2_blocks_0_mlp_fc2_scale, layers_2_blocks_0_mlp_fc2_bias, layers_2_blocks_0_mlp_fc2_MAX_D, layers_2_blocks_0_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_2_blocks_1 ---
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_1_norm1_weight, layers_2_blocks_1_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1152, layers_2_blocks_1_attn_qkv_weights,
        layers_2_blocks_1_attn_qkv_pos, layers_2_blocks_1_attn_qkv_mask, layers_2_blocks_1_attn_qkv_slots, layers_2_blocks_1_attn_qkv_in_act_bits, layers_2_blocks_1_attn_qkv_in_offset_table,
        384, 1152,
        layers_2_blocks_1_attn_qkv_scale, layers_2_blocks_1_attn_qkv_bias, layers_2_blocks_1_attn_qkv_MAX_D, layers_2_blocks_1_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 2, 2, 1152, 1);
for(int i=0; i<4608; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<4; hw++) {
    for(int d=0; d<384; d++) {
        q_buf[hw*384 + d] = unpacked_fm[hw*1152 + d];
        k_buf[hw*384 + d] = unpacked_fm[hw*1152 + 384 + d];
        v_buf[hw*384 + d] = unpacked_fm[hw*1152 + 2*384 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 2, 2, 384, 2);
window_partition_8bit(k_buf, unpacked_fm, 2, 2, 384, 2);
window_partition_8bit(v_buf, tmp_window_buf, 2, 2, 384, 2);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*1536, unpacked_fm + nw*1536, tmp_window_buf + nw*1536,
        attn_out + nw*1536, dummy_rel_pos,
        2, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 2, 2, 384, 2);
reverse_cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 2, 2, 384, 1);
for(int i=0; i<1536; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*384, layers_2_blocks_1_attn_proj_weights,
        layers_2_blocks_1_attn_proj_pos, layers_2_blocks_1_attn_proj_mask, layers_2_blocks_1_attn_proj_slots, layers_2_blocks_1_attn_proj_in_act_bits, layers_2_blocks_1_attn_proj_in_offset_table,
        384, 384,
        layers_2_blocks_1_attn_proj_scale, layers_2_blocks_1_attn_proj_bias, layers_2_blocks_1_attn_proj_MAX_D, layers_2_blocks_1_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_1_norm2_weight, layers_2_blocks_1_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1536, layers_2_blocks_1_mlp_fc1_weights,
        layers_2_blocks_1_mlp_fc1_pos, layers_2_blocks_1_mlp_fc1_mask, layers_2_blocks_1_mlp_fc1_slots, layers_2_blocks_1_mlp_fc1_in_act_bits, layers_2_blocks_1_mlp_fc1_in_offset_table,
        384, 1536,
        layers_2_blocks_1_mlp_fc1_scale, layers_2_blocks_1_mlp_fc1_bias, layers_2_blocks_1_mlp_fc1_MAX_D, layers_2_blocks_1_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 6144);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*1536), packed_fm + hw*384, layers_2_blocks_1_mlp_fc2_weights,
        layers_2_blocks_1_mlp_fc2_pos, layers_2_blocks_1_mlp_fc2_mask, layers_2_blocks_1_mlp_fc2_slots, layers_2_blocks_1_mlp_fc2_in_act_bits, layers_2_blocks_1_mlp_fc2_in_offset_table,
        1536, 384,
        layers_2_blocks_1_mlp_fc2_scale, layers_2_blocks_1_mlp_fc2_bias, layers_2_blocks_1_mlp_fc2_MAX_D, layers_2_blocks_1_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_2_blocks_2 ---
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_2_norm1_weight, layers_2_blocks_2_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1152, layers_2_blocks_2_attn_qkv_weights,
        layers_2_blocks_2_attn_qkv_pos, layers_2_blocks_2_attn_qkv_mask, layers_2_blocks_2_attn_qkv_slots, layers_2_blocks_2_attn_qkv_in_act_bits, layers_2_blocks_2_attn_qkv_in_offset_table,
        384, 1152,
        layers_2_blocks_2_attn_qkv_scale, layers_2_blocks_2_attn_qkv_bias, layers_2_blocks_2_attn_qkv_MAX_D, layers_2_blocks_2_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    for(int d=0; d<384; d++) {
        q_buf[hw*384 + d] = unpacked_fm[hw*1152 + d];
        k_buf[hw*384 + d] = unpacked_fm[hw*1152 + 384 + d];
        v_buf[hw*384 + d] = unpacked_fm[hw*1152 + 2*384 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 2, 2, 384, 2);
window_partition_8bit(k_buf, unpacked_fm, 2, 2, 384, 2);
window_partition_8bit(v_buf, tmp_window_buf, 2, 2, 384, 2);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*1536, unpacked_fm + nw*1536, tmp_window_buf + nw*1536,
        attn_out + nw*1536, dummy_rel_pos,
        2, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 2, 2, 384, 2);
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*384, layers_2_blocks_2_attn_proj_weights,
        layers_2_blocks_2_attn_proj_pos, layers_2_blocks_2_attn_proj_mask, layers_2_blocks_2_attn_proj_slots, layers_2_blocks_2_attn_proj_in_act_bits, layers_2_blocks_2_attn_proj_in_offset_table,
        384, 384,
        layers_2_blocks_2_attn_proj_scale, layers_2_blocks_2_attn_proj_bias, layers_2_blocks_2_attn_proj_MAX_D, layers_2_blocks_2_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_2_norm2_weight, layers_2_blocks_2_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1536, layers_2_blocks_2_mlp_fc1_weights,
        layers_2_blocks_2_mlp_fc1_pos, layers_2_blocks_2_mlp_fc1_mask, layers_2_blocks_2_mlp_fc1_slots, layers_2_blocks_2_mlp_fc1_in_act_bits, layers_2_blocks_2_mlp_fc1_in_offset_table,
        384, 1536,
        layers_2_blocks_2_mlp_fc1_scale, layers_2_blocks_2_mlp_fc1_bias, layers_2_blocks_2_mlp_fc1_MAX_D, layers_2_blocks_2_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 6144);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*1536), packed_fm + hw*384, layers_2_blocks_2_mlp_fc2_weights,
        layers_2_blocks_2_mlp_fc2_pos, layers_2_blocks_2_mlp_fc2_mask, layers_2_blocks_2_mlp_fc2_slots, layers_2_blocks_2_mlp_fc2_in_act_bits, layers_2_blocks_2_mlp_fc2_in_offset_table,
        1536, 384,
        layers_2_blocks_2_mlp_fc2_scale, layers_2_blocks_2_mlp_fc2_bias, layers_2_blocks_2_mlp_fc2_MAX_D, layers_2_blocks_2_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_2_blocks_3 ---
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_3_norm1_weight, layers_2_blocks_3_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1152, layers_2_blocks_3_attn_qkv_weights,
        layers_2_blocks_3_attn_qkv_pos, layers_2_blocks_3_attn_qkv_mask, layers_2_blocks_3_attn_qkv_slots, layers_2_blocks_3_attn_qkv_in_act_bits, layers_2_blocks_3_attn_qkv_in_offset_table,
        384, 1152,
        layers_2_blocks_3_attn_qkv_scale, layers_2_blocks_3_attn_qkv_bias, layers_2_blocks_3_attn_qkv_MAX_D, layers_2_blocks_3_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 2, 2, 1152, 1);
for(int i=0; i<4608; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<4; hw++) {
    for(int d=0; d<384; d++) {
        q_buf[hw*384 + d] = unpacked_fm[hw*1152 + d];
        k_buf[hw*384 + d] = unpacked_fm[hw*1152 + 384 + d];
        v_buf[hw*384 + d] = unpacked_fm[hw*1152 + 2*384 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 2, 2, 384, 2);
window_partition_8bit(k_buf, unpacked_fm, 2, 2, 384, 2);
window_partition_8bit(v_buf, tmp_window_buf, 2, 2, 384, 2);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*1536, unpacked_fm + nw*1536, tmp_window_buf + nw*1536,
        attn_out + nw*1536, dummy_rel_pos,
        2, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 2, 2, 384, 2);
reverse_cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 2, 2, 384, 1);
for(int i=0; i<1536; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*384, layers_2_blocks_3_attn_proj_weights,
        layers_2_blocks_3_attn_proj_pos, layers_2_blocks_3_attn_proj_mask, layers_2_blocks_3_attn_proj_slots, layers_2_blocks_3_attn_proj_in_act_bits, layers_2_blocks_3_attn_proj_in_offset_table,
        384, 384,
        layers_2_blocks_3_attn_proj_scale, layers_2_blocks_3_attn_proj_bias, layers_2_blocks_3_attn_proj_MAX_D, layers_2_blocks_3_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_3_norm2_weight, layers_2_blocks_3_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1536, layers_2_blocks_3_mlp_fc1_weights,
        layers_2_blocks_3_mlp_fc1_pos, layers_2_blocks_3_mlp_fc1_mask, layers_2_blocks_3_mlp_fc1_slots, layers_2_blocks_3_mlp_fc1_in_act_bits, layers_2_blocks_3_mlp_fc1_in_offset_table,
        384, 1536,
        layers_2_blocks_3_mlp_fc1_scale, layers_2_blocks_3_mlp_fc1_bias, layers_2_blocks_3_mlp_fc1_MAX_D, layers_2_blocks_3_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 6144);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*1536), packed_fm + hw*384, layers_2_blocks_3_mlp_fc2_weights,
        layers_2_blocks_3_mlp_fc2_pos, layers_2_blocks_3_mlp_fc2_mask, layers_2_blocks_3_mlp_fc2_slots, layers_2_blocks_3_mlp_fc2_in_act_bits, layers_2_blocks_3_mlp_fc2_in_offset_table,
        1536, 384,
        layers_2_blocks_3_mlp_fc2_scale, layers_2_blocks_3_mlp_fc2_bias, layers_2_blocks_3_mlp_fc2_MAX_D, layers_2_blocks_3_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_2_blocks_4 ---
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_4_norm1_weight, layers_2_blocks_4_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1152, layers_2_blocks_4_attn_qkv_weights,
        layers_2_blocks_4_attn_qkv_pos, layers_2_blocks_4_attn_qkv_mask, layers_2_blocks_4_attn_qkv_slots, layers_2_blocks_4_attn_qkv_in_act_bits, layers_2_blocks_4_attn_qkv_in_offset_table,
        384, 1152,
        layers_2_blocks_4_attn_qkv_scale, layers_2_blocks_4_attn_qkv_bias, layers_2_blocks_4_attn_qkv_MAX_D, layers_2_blocks_4_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    for(int d=0; d<384; d++) {
        q_buf[hw*384 + d] = unpacked_fm[hw*1152 + d];
        k_buf[hw*384 + d] = unpacked_fm[hw*1152 + 384 + d];
        v_buf[hw*384 + d] = unpacked_fm[hw*1152 + 2*384 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 2, 2, 384, 2);
window_partition_8bit(k_buf, unpacked_fm, 2, 2, 384, 2);
window_partition_8bit(v_buf, tmp_window_buf, 2, 2, 384, 2);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*1536, unpacked_fm + nw*1536, tmp_window_buf + nw*1536,
        attn_out + nw*1536, dummy_rel_pos,
        2, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 2, 2, 384, 2);
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*384, layers_2_blocks_4_attn_proj_weights,
        layers_2_blocks_4_attn_proj_pos, layers_2_blocks_4_attn_proj_mask, layers_2_blocks_4_attn_proj_slots, layers_2_blocks_4_attn_proj_in_act_bits, layers_2_blocks_4_attn_proj_in_offset_table,
        384, 384,
        layers_2_blocks_4_attn_proj_scale, layers_2_blocks_4_attn_proj_bias, layers_2_blocks_4_attn_proj_MAX_D, layers_2_blocks_4_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_4_norm2_weight, layers_2_blocks_4_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1536, layers_2_blocks_4_mlp_fc1_weights,
        layers_2_blocks_4_mlp_fc1_pos, layers_2_blocks_4_mlp_fc1_mask, layers_2_blocks_4_mlp_fc1_slots, layers_2_blocks_4_mlp_fc1_in_act_bits, layers_2_blocks_4_mlp_fc1_in_offset_table,
        384, 1536,
        layers_2_blocks_4_mlp_fc1_scale, layers_2_blocks_4_mlp_fc1_bias, layers_2_blocks_4_mlp_fc1_MAX_D, layers_2_blocks_4_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 6144);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*1536), packed_fm + hw*384, layers_2_blocks_4_mlp_fc2_weights,
        layers_2_blocks_4_mlp_fc2_pos, layers_2_blocks_4_mlp_fc2_mask, layers_2_blocks_4_mlp_fc2_slots, layers_2_blocks_4_mlp_fc2_in_act_bits, layers_2_blocks_4_mlp_fc2_in_offset_table,
        1536, 384,
        layers_2_blocks_4_mlp_fc2_scale, layers_2_blocks_4_mlp_fc2_bias, layers_2_blocks_4_mlp_fc2_MAX_D, layers_2_blocks_4_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_2_blocks_5 ---
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_5_norm1_weight, layers_2_blocks_5_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1152, layers_2_blocks_5_attn_qkv_weights,
        layers_2_blocks_5_attn_qkv_pos, layers_2_blocks_5_attn_qkv_mask, layers_2_blocks_5_attn_qkv_slots, layers_2_blocks_5_attn_qkv_in_act_bits, layers_2_blocks_5_attn_qkv_in_offset_table,
        384, 1152,
        layers_2_blocks_5_attn_qkv_scale, layers_2_blocks_5_attn_qkv_bias, layers_2_blocks_5_attn_qkv_MAX_D, layers_2_blocks_5_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 2, 2, 1152, 1);
for(int i=0; i<4608; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<4; hw++) {
    for(int d=0; d<384; d++) {
        q_buf[hw*384 + d] = unpacked_fm[hw*1152 + d];
        k_buf[hw*384 + d] = unpacked_fm[hw*1152 + 384 + d];
        v_buf[hw*384 + d] = unpacked_fm[hw*1152 + 2*384 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 2, 2, 384, 2);
window_partition_8bit(k_buf, unpacked_fm, 2, 2, 384, 2);
window_partition_8bit(v_buf, tmp_window_buf, 2, 2, 384, 2);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*1536, unpacked_fm + nw*1536, tmp_window_buf + nw*1536,
        attn_out + nw*1536, dummy_rel_pos,
        2, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 2, 2, 384, 2);
reverse_cyclic_shift_8bit(unpacked_fm, tmp_window_buf, 2, 2, 384, 1);
for(int i=0; i<1536; i++) unpacked_fm[i] = tmp_window_buf[i];
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*384, layers_2_blocks_5_attn_proj_weights,
        layers_2_blocks_5_attn_proj_pos, layers_2_blocks_5_attn_proj_mask, layers_2_blocks_5_attn_proj_slots, layers_2_blocks_5_attn_proj_in_act_bits, layers_2_blocks_5_attn_proj_in_offset_table,
        384, 384,
        layers_2_blocks_5_attn_proj_scale, layers_2_blocks_5_attn_proj_bias, layers_2_blocks_5_attn_proj_MAX_D, layers_2_blocks_5_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*384, packed_fm + hw*384, 384, layers_2_blocks_5_norm2_weight, layers_2_blocks_5_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*384), packed_fm + hw*1536, layers_2_blocks_5_mlp_fc1_weights,
        layers_2_blocks_5_mlp_fc1_pos, layers_2_blocks_5_mlp_fc1_mask, layers_2_blocks_5_mlp_fc1_slots, layers_2_blocks_5_mlp_fc1_in_act_bits, layers_2_blocks_5_mlp_fc1_in_offset_table,
        384, 1536,
        layers_2_blocks_5_mlp_fc1_scale, layers_2_blocks_5_mlp_fc1_bias, layers_2_blocks_5_mlp_fc1_MAX_D, layers_2_blocks_5_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 6144);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<4; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*1536), packed_fm + hw*384, layers_2_blocks_5_mlp_fc2_weights,
        layers_2_blocks_5_mlp_fc2_pos, layers_2_blocks_5_mlp_fc2_mask, layers_2_blocks_5_mlp_fc2_slots, layers_2_blocks_5_mlp_fc2_in_act_bits, layers_2_blocks_5_mlp_fc2_in_offset_table,
        1536, 384,
        layers_2_blocks_5_mlp_fc2_scale, layers_2_blocks_5_mlp_fc2_bias, layers_2_blocks_5_mlp_fc2_MAX_D, layers_2_blocks_5_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- Patch Merging Stage 2 -> 3 ---
patch_merging_8bit(unpacked_fm, packed_fm, 2, 2, 384);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*1536, packed_fm + hw*1536, 1536, layers_3_downsample_norm_weight, layers_3_downsample_norm_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*1536), packed_fm + hw*768, layers_3_downsample_reduction_weights,
        layers_3_downsample_reduction_pos, layers_3_downsample_reduction_mask, layers_3_downsample_reduction_slots, layers_3_downsample_reduction_in_act_bits, layers_3_downsample_reduction_in_offset_table,
        1536, 768,
        layers_3_downsample_reduction_scale, layers_3_downsample_reduction_bias, layers_3_downsample_reduction_MAX_D, layers_3_downsample_reduction_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_3_blocks_0 ---
for(int hw=0; hw<1; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*768, packed_fm + hw*768, 768, layers_3_blocks_0_norm1_weight, layers_3_blocks_0_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*768), packed_fm + hw*2304, layers_3_blocks_0_attn_qkv_weights,
        layers_3_blocks_0_attn_qkv_pos, layers_3_blocks_0_attn_qkv_mask, layers_3_blocks_0_attn_qkv_slots, layers_3_blocks_0_attn_qkv_in_act_bits, layers_3_blocks_0_attn_qkv_in_offset_table,
        768, 2304,
        layers_3_blocks_0_attn_qkv_scale, layers_3_blocks_0_attn_qkv_bias, layers_3_blocks_0_attn_qkv_MAX_D, layers_3_blocks_0_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    for(int d=0; d<768; d++) {
        q_buf[hw*768 + d] = unpacked_fm[hw*2304 + d];
        k_buf[hw*768 + d] = unpacked_fm[hw*2304 + 768 + d];
        v_buf[hw*768 + d] = unpacked_fm[hw*2304 + 2*768 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 1, 1, 768, 1);
window_partition_8bit(k_buf, unpacked_fm, 1, 1, 768, 1);
window_partition_8bit(v_buf, tmp_window_buf, 1, 1, 768, 1);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*768, unpacked_fm + nw*768, tmp_window_buf + nw*768,
        attn_out + nw*768, dummy_rel_pos,
        1, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 1, 1, 768, 1);
for(int hw=0; hw<1; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*768), packed_fm + hw*768, layers_3_blocks_0_attn_proj_weights,
        layers_3_blocks_0_attn_proj_pos, layers_3_blocks_0_attn_proj_mask, layers_3_blocks_0_attn_proj_slots, layers_3_blocks_0_attn_proj_in_act_bits, layers_3_blocks_0_attn_proj_in_offset_table,
        768, 768,
        layers_3_blocks_0_attn_proj_scale, layers_3_blocks_0_attn_proj_bias, layers_3_blocks_0_attn_proj_MAX_D, layers_3_blocks_0_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*768, packed_fm + hw*768, 768, layers_3_blocks_0_norm2_weight, layers_3_blocks_0_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*768), packed_fm + hw*3072, layers_3_blocks_0_mlp_fc1_weights,
        layers_3_blocks_0_mlp_fc1_pos, layers_3_blocks_0_mlp_fc1_mask, layers_3_blocks_0_mlp_fc1_slots, layers_3_blocks_0_mlp_fc1_in_act_bits, layers_3_blocks_0_mlp_fc1_in_offset_table,
        768, 3072,
        layers_3_blocks_0_mlp_fc1_scale, layers_3_blocks_0_mlp_fc1_bias, layers_3_blocks_0_mlp_fc1_MAX_D, layers_3_blocks_0_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 3072);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*3072), packed_fm + hw*768, layers_3_blocks_0_mlp_fc2_weights,
        layers_3_blocks_0_mlp_fc2_pos, layers_3_blocks_0_mlp_fc2_mask, layers_3_blocks_0_mlp_fc2_slots, layers_3_blocks_0_mlp_fc2_in_act_bits, layers_3_blocks_0_mlp_fc2_in_offset_table,
        3072, 768,
        layers_3_blocks_0_mlp_fc2_scale, layers_3_blocks_0_mlp_fc2_bias, layers_3_blocks_0_mlp_fc2_MAX_D, layers_3_blocks_0_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// --- layers_3_blocks_1 ---
for(int hw=0; hw<1; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*768, packed_fm + hw*768, 768, layers_3_blocks_1_norm1_weight, layers_3_blocks_1_norm1_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*768), packed_fm + hw*2304, layers_3_blocks_1_attn_qkv_weights,
        layers_3_blocks_1_attn_qkv_pos, layers_3_blocks_1_attn_qkv_mask, layers_3_blocks_1_attn_qkv_slots, layers_3_blocks_1_attn_qkv_in_act_bits, layers_3_blocks_1_attn_qkv_in_offset_table,
        768, 2304,
        layers_3_blocks_1_attn_qkv_scale, layers_3_blocks_1_attn_qkv_bias, layers_3_blocks_1_attn_qkv_MAX_D, layers_3_blocks_1_attn_qkv_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    for(int d=0; d<768; d++) {
        q_buf[hw*768 + d] = unpacked_fm[hw*2304 + d];
        k_buf[hw*768 + d] = unpacked_fm[hw*2304 + 768 + d];
        v_buf[hw*768 + d] = unpacked_fm[hw*2304 + 2*768 + d];
    }
}
window_partition_8bit(q_buf, packed_fm, 1, 1, 768, 1);
window_partition_8bit(k_buf, unpacked_fm, 1, 1, 768, 1);
window_partition_8bit(v_buf, tmp_window_buf, 1, 1, 768, 1);
for(int nw=0; nw<1; nw++) {
    swar_window_attention_8bit(
        packed_fm + nw*768, unpacked_fm + nw*768, tmp_window_buf + nw*768,
        attn_out + nw*768, dummy_rel_pos,
        1, 32, 1.0, 1.0);
}
window_reverse_8bit(attn_out, unpacked_fm, 1, 1, 768, 1);
for(int hw=0; hw<1; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*768), packed_fm + hw*768, layers_3_blocks_1_attn_proj_weights,
        layers_3_blocks_1_attn_proj_pos, layers_3_blocks_1_attn_proj_mask, layers_3_blocks_1_attn_proj_slots, layers_3_blocks_1_attn_proj_in_act_bits, layers_3_blocks_1_attn_proj_in_offset_table,
        768, 768,
        layers_3_blocks_1_attn_proj_scale, layers_3_blocks_1_attn_proj_bias, layers_3_blocks_1_attn_proj_MAX_D, layers_3_blocks_1_attn_proj_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    swar_layer_norm_16bit(unpacked_fm + hw*768, packed_fm + hw*768, 768, layers_3_blocks_1_norm2_weight, layers_3_blocks_1_norm2_bias);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*768), packed_fm + hw*3072, layers_3_blocks_1_mlp_fc1_weights,
        layers_3_blocks_1_mlp_fc1_pos, layers_3_blocks_1_mlp_fc1_mask, layers_3_blocks_1_mlp_fc1_slots, layers_3_blocks_1_mlp_fc1_in_act_bits, layers_3_blocks_1_mlp_fc1_in_offset_table,
        768, 3072,
        layers_3_blocks_1_mlp_fc1_scale, layers_3_blocks_1_mlp_fc1_bias, layers_3_blocks_1_mlp_fc1_MAX_D, layers_3_blocks_1_mlp_fc1_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
swar_gelu_approx_16bit(unpacked_fm, packed_fm, 3072);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;
for(int hw=0; hw<1; hw++) {
    run_linear_general((const uint8_t*)(unpacked_fm + hw*3072), packed_fm + hw*768, layers_3_blocks_1_mlp_fc2_weights,
        layers_3_blocks_1_mlp_fc2_pos, layers_3_blocks_1_mlp_fc2_mask, layers_3_blocks_1_mlp_fc2_slots, layers_3_blocks_1_mlp_fc2_in_act_bits, layers_3_blocks_1_mlp_fc2_in_offset_table,
        3072, 768,
        layers_3_blocks_1_mlp_fc2_scale, layers_3_blocks_1_mlp_fc2_bias, layers_3_blocks_1_mlp_fc2_MAX_D, layers_3_blocks_1_mlp_fc2_WORDS_PER_FILTER);
}
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// Global Average Pool
global_average_pool_2d(unpacked_fm, packed_fm, 1, 1, 768);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// head.0
swar_layer_norm_16bit(unpacked_fm, packed_fm, 768, head_0_weight, head_0_bias);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

// head.1
run_linear_general((const uint8_t*)unpacked_fm, packed_fm, head_1_weights,
    head_1_pos, head_1_mask, head_1_slots, head_1_in_act_bits, head_1_in_offset_table,
    768, 100,
    head_1_scale, head_1_bias, head_1_MAX_D, head_1_WORDS_PER_FILTER);
tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;

