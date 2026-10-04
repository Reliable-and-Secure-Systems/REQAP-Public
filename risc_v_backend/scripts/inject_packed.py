import re
with open('vc707_deployment/main_vc707_mobilenet.c', 'r', encoding='utf-8') as f:
    c = f.read()

with open('risc_v_backend/scripts/mobilenet_packed_pipeline.txt', 'r', encoding='utf-16le') as f:
    pipe = f.read()

c = c.replace('printf("Packed forward pass not implemented yet.\\n");', pipe)

with open('vc707_deployment/main_vc707_mobilenet.c', 'w', encoding='utf-8') as f:
    f.write(c)

print("Injected packed pipeline.")
