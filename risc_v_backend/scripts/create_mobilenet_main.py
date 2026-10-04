import re
import sys

# Read VGG template
with open("vc707_deployment/main_vc707_vgg11.c", "r") as f:
    vgg_code = f.read()

# Replace header includes
vgg_code = vgg_code.replace("vgg11_baseline_weights.h", "generated/mobilenet_packed_weights.h")
vgg_code = vgg_code.replace("vgg11_packed_weights.h", "generated/mobilenet_packed_weights.h")
vgg_code = vgg_code.replace("VGG-11 (Baseline 8-bit)", "MobileNet (Baseline 8-bit)")
vgg_code = vgg_code.replace("VGG-11 (Packed MQF)", "MobileNet (Packed MQF)")
vgg_code = vgg_code.replace("TEST_VGG11", "TEST_MOBILENET")

# Replace run_full_vgg11_packed_forward_pass with run_full_mobilenet_baseline_forward_pass
vgg_code = vgg_code.replace("run_full_vgg11_baseline_forward_pass", "run_full_mobilenet_baseline_forward_pass")
vgg_code = vgg_code.replace("run_full_vgg11_packed_forward_pass", "run_full_mobilenet_packed_forward_pass")
vgg_code = vgg_code.replace("VGG-11 Baseline 8-bit", "MobileNet Baseline 8-bit")
vgg_code = vgg_code.replace("VGG-11 Packed MQF", "MobileNet Packed MQF")
vgg_code = vgg_code.replace("print_vgg11_profiler_summary", "print_mobilenet_profiler_summary")
vgg_code = vgg_code.replace("run_vgg11_unit_tests", "run_mobilenet_unit_tests")
vgg_code = vgg_code.replace("VGG-11 MQF PACKED", "MOBILENET MQF PACKED")
vgg_code = vgg_code.replace("VGG-11 EXECUTION FINISHED SUCCESSFULLY!", "MOBILENET EXECUTION FINISHED SUCCESSFULLY!")

# Now we need to insert the pipeline into run_full_mobilenet_baseline_forward_pass
# The function starts at:
# printf("\n=== Starting Full %s Forward Pass ===\n", mode_name);

# Read generated pipeline
with open("risc_v_backend/scripts/mobilenet_pipeline.txt", "r", encoding="utf-16le") as f:
    pipeline_code = f.read()

# We can replace everything from "/* --- Layer 1: features.0" down to "printf("\n=== Classification Results ===\n");"
pattern = r"/\* --- Layer 1: features\.0.*?(printf\(\"\\n=== Classification Results ===\\n\"\);)"
replacement = pipeline_code + "\n\n    " + r"\1"

new_code = re.sub(pattern, replacement, vgg_code, flags=re.DOTALL)

with open("vc707_deployment/main_vc707_mobilenet.c", "w") as f:
    f.write(new_code)
print("Generated main_vc707_mobilenet.c")
