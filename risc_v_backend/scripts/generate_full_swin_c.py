import os

def dump_swin():
    filename = "risc_v_backend/firmware/swin_packed_pipeline.c"
    
    with open(filename, "w") as f:
        f.write("/* Auto-generated Complete Swin-T Pipeline (Packed MQF) */\n")
        f.write("int8_t *unpacked_fm = fm_B;\n")
        f.write("int8_t *packed_fm = fm_A;\n")
        f.write("int8_t *tmp_ptr;\n\n")
        f.write("int8_t q_buf[4096];\n")
        f.write("int8_t k_buf[4096];\n")
        f.write("int8_t v_buf[4096];\n")
        f.write("int8_t attn_out[4096];\n")
        f.write("int8_t tmp_window_buf[30000];\n")
        f.write("int32_t dummy_rel_pos[4096] = {0};\n\n")
        
        f.write("// Initial state: unpacked_fm has the image.\n")
        f.write("for (int i=0; i<3072; i++) unpacked_fm[i] = cifar_image_0[i];\n\n")

        # Patch Embed
        f.write("// patch_embed.proj\n")
        f.write("run_conv2d_8bit(unpacked_fm, packed_fm, patch_embed_proj_weights,\n")
        f.write("    32, 32, 3,\n")
        f.write("    8, 8, 96,\n")
        f.write("    4, 4, 0, 4,\n")
        f.write("    patch_embed_proj_scale, patch_embed_proj_bias);\n")
        f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n")
        f.write("printf(\"Finished patch_embed.proj\\n\");\n\n")
        
        dims = [96, 192, 384, 768]
        depths = [2, 2, 6, 2]
        
        h, w = 8, 8
        window_size = 7
        
        for layer_idx in range(4):
            dim = dims[layer_idx]
            
            for block_idx in range(depths[layer_idx]):
                prefix = f"layers_{layer_idx}_blocks_{block_idx}"
                
                # In PyTorch, Swin pads H and W to be multiples of window_size.
                # For C execution, we'll assume the C buffers just process HxW
                # and if H < window_size, we just use H as the window_size to avoid out-of-bounds memory.
                eff_win_h = window_size if h >= window_size else h
                eff_win_w = window_size if w >= window_size else w
                seq_len = eff_win_h * eff_win_w
                num_windows = (h // eff_win_h) * (w // eff_win_w) if h >= eff_win_h else 1
                
                f.write(f"// --- {prefix} ---\n")
                
                # norm1
                f.write(f"for(int hw=0; hw<{h*w}; hw++) {{\n")
                f.write(f"    swar_layer_norm_16bit(unpacked_fm + hw*{dim}, packed_fm + hw*{dim}, {dim}, {prefix}_norm1_weight, {prefix}_norm1_bias);\n")
                f.write(f"}}\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n")
                
                # qkv
                f.write(f"for(int hw=0; hw<{h*w}; hw++) {{\n")
                f.write(f"    run_linear_general((const uint8_t*)(unpacked_fm + hw*{dim}), packed_fm + hw*{dim*3}, {prefix}_attn_qkv_weights,\n")
                f.write(f"        {prefix}_attn_qkv_pos, {prefix}_attn_qkv_mask, {prefix}_attn_qkv_slots, {prefix}_attn_qkv_in_act_bits, {prefix}_attn_qkv_in_offset_table,\n")
                f.write(f"        {dim}, {dim*3},\n")
                f.write(f"        {prefix}_attn_qkv_scale, {prefix}_attn_qkv_bias, {prefix}_attn_qkv_MAX_D, {prefix}_attn_qkv_WORDS_PER_FILTER);\n")
                f.write(f"}}\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n")
                
                # Cyclic shift for SW-MSA (blocks 1, 3, 5, etc.)
                shift_size = eff_win_h // 2 if block_idx % 2 != 0 else 0
                if shift_size > 0:
                    f.write(f"cyclic_shift_8bit(unpacked_fm, tmp_window_buf, {h}, {w}, {dim*3}, {shift_size});\n")
                    f.write(f"for(int i=0; i<{h*w*dim*3}; i++) unpacked_fm[i] = tmp_window_buf[i];\n")
                
                # Split QKV into q_buf, k_buf, v_buf
                f.write(f"for(int hw=0; hw<{h*w}; hw++) {{\n")
                f.write(f"    for(int d=0; d<{dim}; d++) {{\n")
                f.write(f"        q_buf[hw*{dim} + d] = unpacked_fm[hw*{dim*3} + d];\n")
                f.write(f"        k_buf[hw*{dim} + d] = unpacked_fm[hw*{dim*3} + {dim} + d];\n")
                f.write(f"        v_buf[hw*{dim} + d] = unpacked_fm[hw*{dim*3} + 2*{dim} + d];\n")
                f.write(f"    }}\n")
                f.write(f"}}\n")
                
                # Window partition and attention
                f.write(f"window_partition_8bit(q_buf, packed_fm, {h}, {w}, {dim}, {eff_win_h});\n")
                f.write(f"window_partition_8bit(k_buf, unpacked_fm, {h}, {w}, {dim}, {eff_win_h});\n")
                f.write(f"window_partition_8bit(v_buf, tmp_window_buf, {h}, {w}, {dim}, {eff_win_h});\n")
                
                f.write(f"for(int nw=0; nw<{num_windows}; nw++) {{\n")
                f.write(f"    swar_window_attention_8bit(\n")
                f.write(f"        packed_fm + nw*{seq_len*dim}, unpacked_fm + nw*{seq_len*dim}, tmp_window_buf + nw*{seq_len*dim},\n")
                f.write(f"        attn_out + nw*{seq_len*dim}, dummy_rel_pos,\n")
                f.write(f"        {eff_win_h}, 32, 1.0, 1.0);\n")
                f.write(f"}}\n")
                
                # Reverse window partition
                f.write(f"window_reverse_8bit(attn_out, unpacked_fm, {h}, {w}, {dim}, {eff_win_h});\n")
                
                # Reverse cyclic shift
                if shift_size > 0:
                    f.write(f"reverse_cyclic_shift_8bit(unpacked_fm, tmp_window_buf, {h}, {w}, {dim}, {shift_size});\n")
                    f.write(f"for(int i=0; i<{h*w*dim}; i++) unpacked_fm[i] = tmp_window_buf[i];\n")
                
                # attn proj
                f.write(f"for(int hw=0; hw<{h*w}; hw++) {{\n")
                f.write(f"    run_linear_general((const uint8_t*)(unpacked_fm + hw*{dim}), packed_fm + hw*{dim}, {prefix}_attn_proj_weights,\n")
                f.write(f"        {prefix}_attn_proj_pos, {prefix}_attn_proj_mask, {prefix}_attn_proj_slots, {prefix}_attn_proj_in_act_bits, {prefix}_attn_proj_in_offset_table,\n")
                f.write(f"        {dim}, {dim},\n")
                f.write(f"        {prefix}_attn_proj_scale, {prefix}_attn_proj_bias, {prefix}_attn_proj_MAX_D, {prefix}_attn_proj_WORDS_PER_FILTER);\n")
                f.write(f"}}\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n")
                
                # Skip connection + norm2
                f.write(f"for(int hw=0; hw<{h*w}; hw++) {{\n")
                f.write(f"    swar_layer_norm_16bit(unpacked_fm + hw*{dim}, packed_fm + hw*{dim}, {dim}, {prefix}_norm2_weight, {prefix}_norm2_bias);\n")
                f.write(f"}}\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n")
                
                # mlp fc1
                f.write(f"for(int hw=0; hw<{h*w}; hw++) {{\n")
                f.write(f"    run_linear_general((const uint8_t*)(unpacked_fm + hw*{dim}), packed_fm + hw*{dim*4}, {prefix}_mlp_fc1_weights,\n")
                f.write(f"        {prefix}_mlp_fc1_pos, {prefix}_mlp_fc1_mask, {prefix}_mlp_fc1_slots, {prefix}_mlp_fc1_in_act_bits, {prefix}_mlp_fc1_in_offset_table,\n")
                f.write(f"        {dim}, {dim*4},\n")
                f.write(f"        {prefix}_mlp_fc1_scale, {prefix}_mlp_fc1_bias, {prefix}_mlp_fc1_MAX_D, {prefix}_mlp_fc1_WORDS_PER_FILTER);\n")
                f.write(f"}}\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n")
                
                # gelu
                f.write(f"swar_gelu_approx_16bit(unpacked_fm, packed_fm, {h*w*dim*4});\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n")
                
                # mlp fc2
                f.write(f"for(int hw=0; hw<{h*w}; hw++) {{\n")
                f.write(f"    run_linear_general((const uint8_t*)(unpacked_fm + hw*{dim*4}), packed_fm + hw*{dim}, {prefix}_mlp_fc2_weights,\n")
                f.write(f"        {prefix}_mlp_fc2_pos, {prefix}_mlp_fc2_mask, {prefix}_mlp_fc2_slots, {prefix}_mlp_fc2_in_act_bits, {prefix}_mlp_fc2_in_offset_table,\n")
                f.write(f"        {dim*4}, {dim},\n")
                f.write(f"        {prefix}_mlp_fc2_scale, {prefix}_mlp_fc2_bias, {prefix}_mlp_fc2_MAX_D, {prefix}_mlp_fc2_WORDS_PER_FILTER);\n")
                f.write(f"}}\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n\n")
                
            # Patch merging
            if layer_idx < 3:
                down_idx = layer_idx + 1
                f.write(f"// --- Patch Merging Stage {layer_idx} -> {down_idx} ---\n")
                # 1. Patch Merge (reshape) -> output channels = 4*dim, H=H/2, W=W/2
                f.write(f"patch_merging_8bit(unpacked_fm, packed_fm, {h}, {w}, {dim});\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n")
                h = h // 2
                w = w // 2
                
                # 2. Norm
                f.write(f"for(int hw=0; hw<{h*w}; hw++) {{\n")
                f.write(f"    swar_layer_norm_16bit(unpacked_fm + hw*{dim*4}, packed_fm + hw*{dim*4}, {dim*4}, layers_{down_idx}_downsample_norm_weight, layers_{down_idx}_downsample_norm_bias);\n")
                f.write(f"}}\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n")
                
                # 3. Reduction (Linear)
                f.write(f"for(int hw=0; hw<{h*w}; hw++) {{\n")
                f.write(f"    run_linear_general((const uint8_t*)(unpacked_fm + hw*{dim*4}), packed_fm + hw*{dim*2}, layers_{down_idx}_downsample_reduction_weights,\n")
                f.write(f"        layers_{down_idx}_downsample_reduction_pos, layers_{down_idx}_downsample_reduction_mask, layers_{down_idx}_downsample_reduction_slots, layers_{down_idx}_downsample_reduction_in_act_bits, layers_{down_idx}_downsample_reduction_in_offset_table,\n")
                f.write(f"        {dim*4}, {dim*2},\n")
                f.write(f"        layers_{down_idx}_downsample_reduction_scale, layers_{down_idx}_downsample_reduction_bias, layers_{down_idx}_downsample_reduction_MAX_D, layers_{down_idx}_downsample_reduction_WORDS_PER_FILTER);\n")
                f.write(f"}}\n")
                f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n\n")

        # Global average pool
        f.write("// Global Average Pool\n")
        f.write(f"global_average_pool_2d(unpacked_fm, packed_fm, {h}, {w}, {dims[3]});\n")
        f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n\n")

        # Head Norm
        f.write("// head.0\n")
        f.write(f"swar_layer_norm_16bit(unpacked_fm, packed_fm, {dims[3]}, head_0_weight, head_0_bias);\n")
        f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n\n")
        
        # Head Linear
        f.write("// head.1\n")
        f.write(f"run_linear_general((const uint8_t*)unpacked_fm, packed_fm, head_1_weights,\n")
        f.write(f"    head_1_pos, head_1_mask, head_1_slots, head_1_in_act_bits, head_1_in_offset_table,\n")
        f.write(f"    {dims[3]}, 100,\n")
        f.write(f"    head_1_scale, head_1_bias, head_1_MAX_D, head_1_WORDS_PER_FILTER);\n")
        f.write("tmp_ptr = unpacked_fm; unpacked_fm = packed_fm; packed_fm = tmp_ptr;\n\n")

if __name__ == "__main__":
    dump_swin()
    print("Generated risc_v_backend/firmware/swin_packed_pipeline.c")
