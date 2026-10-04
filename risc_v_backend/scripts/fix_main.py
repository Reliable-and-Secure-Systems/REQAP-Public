import re

with open('vc707_deployment/main_vc707_vgg11.c', 'r', encoding='utf-8') as f:
    vgg = f.read()

vgg = vgg.replace('vgg11_baseline_weights.h', 'generated/mobilenet_packed_weights.h')
vgg = vgg.replace('vgg11_packed_weights.h', 'generated/mobilenet_packed_weights.h')
vgg = vgg.replace('VGG-11 (Baseline 8-bit)', 'MobileNet (Baseline 8-bit)')
vgg = vgg.replace('VGG-11 (Packed MQF)', 'MobileNet (Packed MQF)')
vgg = vgg.replace('TEST_VGG11', 'TEST_MOBILENET')
vgg = vgg.replace('run_full_vgg11_baseline_forward_pass', 'run_full_mobilenet_baseline_forward_pass')
vgg = vgg.replace('run_full_vgg11_packed_forward_pass', 'run_full_mobilenet_packed_forward_pass')
vgg = vgg.replace('VGG-11 Baseline 8-bit', 'MobileNet Baseline 8-bit')
vgg = vgg.replace('VGG-11 Packed MQF', 'MobileNet Packed MQF')
vgg = vgg.replace('print_vgg11_profiler_summary', 'print_mobilenet_profiler_summary')
vgg = vgg.replace('run_vgg11_unit_tests', 'run_mobilenet_unit_tests')
vgg = vgg.replace('VGG-11 MQF PACKED', 'MOBILENET MQF PACKED')
vgg = vgg.replace('VGG-11 EXECUTION FINISHED SUCCESSFULLY!', 'MOBILENET EXECUTION FINISHED SUCCESSFULLY!')

# Strip out VGG-specific packed offsets
vgg = re.sub(r'#ifndef BASELINE_MODE\s+uint32_t features_18_offsets.*?#endif', '', vgg, flags=re.DOTALL)

with open('risc_v_backend/scripts/mobilenet_pipeline.txt', 'r', encoding='utf-16le') as f:
    pipe_base = f.read().replace('\ufeff', '')

with open('risc_v_backend/scripts/mobilenet_packed_pipeline.txt', 'r', encoding='utf-16le') as f:
    pipe_pack = f.read().replace('\ufeff', '')

# Replace the body of the function using split lines
lines = vgg.split('\n')
start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "/* --- Layer 1: features.0 (Conv 3->64, 32x32) --- */" in line:
        start_idx = i
    if "=== Classification Results ===" in line:
        end_idx = i - 1  # the printf line
        break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [
        "#ifdef BASELINE_MODE",
        pipe_base,
        "#else",
        pipe_pack,
        "#endif",
        '    printf("\\n=== Classification Results ===\\n");'
    ] + lines[end_idx+1:]
    
    vgg = '\n'.join(new_lines)
    
    printf_fix = """#ifdef BASELINE_MODE
        printf("Class %d: %d\\n", i, in_fm[i]);
#else
        printf("Class %d: %d\\n", i, unpacked_fm[i]);
#endif"""
    vgg = vgg.replace('printf("Class %d: %d\\n", i, fm_B[i]);', printf_fix)
    
    with open('vc707_deployment/main_vc707_mobilenet.c', 'w', encoding='utf-8') as f:
        f.write(vgg)
    print("Done")
else:
    print(f"Failed. start={start_idx}, end={end_idx}")

