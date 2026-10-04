import re
import sys

# Read VGG template
with open("vc707_deployment/main_vc707_vgg11.c", "r", encoding="utf-8") as f:
    vgg_code = f.read()

# Read generated pipelines
with open("risc_v_backend/scripts/mobilenet_pipeline.txt", "r", encoding="utf-16le") as f:
    baseline_pipeline_code = f.read()

with open("risc_v_backend/scripts/mobilenet_packed_pipeline.txt", "r", encoding="utf-16le") as f:
    packed_pipeline_code = f.read()

# Replace header includes
vgg_code = vgg_code.replace("vgg11_baseline_weights.h", "generated/mobilenet_baseline_weights.h")
vgg_code = vgg_code.replace("vgg11_packed_weights.h", "generated/mobilenet_packed_weights.h")
vgg_code = vgg_code.replace("VGG-11 (Baseline 8-bit)", "MobileNet (Baseline 8-bit)")
vgg_code = vgg_code.replace("VGG-11 (Packed MQF)", "MobileNet (Packed MQF)")
vgg_code = vgg_code.replace("TEST_VGG11", "TEST_MOBILENET")

# Replace function names
vgg_code = vgg_code.replace("run_full_vgg11_baseline_forward_pass", "run_full_mobilenet_baseline_forward_pass")
vgg_code = vgg_code.replace("run_full_vgg11_packed_forward_pass", "run_full_mobilenet_packed_forward_pass")
vgg_code = vgg_code.replace("VGG-11 Baseline 8-bit", "MobileNet Baseline 8-bit")
vgg_code = vgg_code.replace("VGG-11 Packed MQF", "MobileNet Packed MQF")
vgg_code = vgg_code.replace("print_vgg11_profiler_summary", "print_mobilenet_profiler_summary")
vgg_code = vgg_code.replace("run_vgg11_unit_tests", "run_mobilenet_unit_tests")
vgg_code = vgg_code.replace("VGG-11 MQF PACKED", "MOBILENET MQF PACKED")
vgg_code = vgg_code.replace("VGG-11 EXECUTION FINISHED SUCCESSFULLY!", "MOBILENET EXECUTION FINISHED SUCCESSFULLY!")

# Inject res_fm
vgg_code = vgg_code.replace("int8_t fm_B[300000];", "int8_t fm_B[300000];\nint8_t res_fm[300000];")

# Inject Baseline Pipeline safely
# Find the start of the baseline function
baseline_func_start = vgg_code.find("static void run_full_mobilenet_baseline_forward_pass(void)")
baseline_block_start = vgg_code.find("/* --- Layer 1: features.0", baseline_func_start)
baseline_block_end = vgg_code.find("printf(\"\\n=== Classification Results ===\\n\");", baseline_block_start)
vgg_code = vgg_code[:baseline_block_start] + baseline_pipeline_code + "\n\n    " + vgg_code[baseline_block_end:]

# Inject Packed Pipeline safely
packed_func_start = vgg_code.find("static void run_full_mobilenet_packed_forward_pass(void)")
packed_block_start = vgg_code.find("/* --- Layer 1: features.0", packed_func_start)
packed_block_end = vgg_code.find("printf(\"\\n=== Classification Results ===\\n\");", packed_block_start)
vgg_code = vgg_code[:packed_block_start] + packed_pipeline_code + "\n\n    " + vgg_code[packed_block_end:]

with open("vc707_deployment/main_vc707_mobilenet.c", "w", encoding="utf-8") as f:
    f.write(vgg_code)

print("Generated main_vc707_mobilenet.c successfully")

