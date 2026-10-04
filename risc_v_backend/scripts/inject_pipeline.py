import re
c = open('vc707_deployment/main_vc707_mobilenet.c', encoding='utf-8').read()
pipe = open('risc_v_backend/scripts/mobilenet_pipeline.txt', encoding='utf-16le').read()

# The function has:
#     /* --- Layer 1: features.0 (Conv 3->64, 32x32) --- */
# ...
#     printf("\n=== Classification Results ===\n");

pattern = r"/\* --- Layer 1: features\.0 \(Conv 3->64, 32x32\) ---\*/.*?printf\(\"\\n=== Classification Results ===\\n\"\);"
# Wait, the pattern might be slightly different.
# Let's split by string.
start_str = "    /* --- Layer 1: features.0"
end_str = "    printf(\"\\n=== Classification Results ===\\n\");"

idx1 = c.find(start_str)
idx2 = c.find(end_str)

if idx1 != -1 and idx2 != -1:
    new_c = c[:idx1] + pipe + "\n" + c[idx2:]
    open('vc707_deployment/main_vc707_mobilenet.c', 'w', encoding='utf-8').write(new_c)
    print("Success")
else:
    print("Failed to find boundaries")
