import os
import pickle
import numpy as np

data_file = r"C:\Mubashir-BTU\Thesis\Codes\Danial\DATE2027_Eval\VGG11\Python_Reference_Model\data\cifar-10-batches-py\test_batch"
h_file = r"C:\Mubashir-BTU\Thesis\Codes\Danial\DATE2027_Eval\VC707_Live_Demo_UI\firmware\cifar_test_image.h"

with open(data_file, "rb") as f:
    d = pickle.load(f, encoding="bytes")

data = d[b"data"]
labels = d[b"labels"]

mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
std  = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
in_scale = 0.035773955285549164

samples = [
    ("cifar_image_0_cat", 0, "cat", 3),
    ("cifar_image_1_airplane", 3, "airplane", 0),
    ("cifar_image_2_automobile", 6, "automobile", 1),
    ("cifar_image_3_dog", 12, "dog", 5),
    ("cifar_image_4_ship", 1, "ship", 8),
]

lines = ["#ifndef CIFAR_TEST_IMAGE_H", "#define CIFAR_TEST_IMAGE_H", "", "#include <stdint.h>", ""]

for name, idx, cname, label_id in samples:
    raw = data[idx].reshape(3, 32, 32).astype(np.float32) / 255.0
    norm = (raw - mean) / std
    q = np.clip(np.round(norm / in_scale), -128, 127).astype(np.int8)
    flat = q.flatten()
    
    lines.append(f"/* Sample Image {name} | Class: {cname} (Label {label_id}) */")
    lines.append(f"static const int8_t {name}[3072] = {{")
    
    for r in range(0, len(flat), 16):
        chunk = flat[r:r+16]
        c_str = ", ".join(f"{v:4d}" for v in chunk)
        if r + 16 < len(flat):
            lines.append(f"    {c_str},")
        else:
            lines.append(f"    {c_str}")
    lines.append("};")
    lines.append("")

lines.append("/* Array of pointers to test images */")
lines.append("static const int8_t * const cifar_test_images[5] = {")
lines.append("    cifar_image_0_cat,")
lines.append("    cifar_image_1_airplane,")
lines.append("    cifar_image_2_automobile,")
lines.append("    cifar_image_3_dog,")
lines.append("    cifar_image_4_ship")
lines.append("};")
lines.append("")
lines.append("static const int cifar_test_labels[5] = {3, 0, 1, 5, 8};")
lines.append('static const char * const cifar_test_names[5] = {"cat", "airplane", "automobile", "dog", "ship"};')
lines.append("")
lines.append("/* Backward compatibility alias */")
lines.append("#define cifar_image_0 cifar_image_0_cat")
lines.append("")
lines.append("#endif /* CIFAR_TEST_IMAGE_H */")

with open(h_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Successfully generated 5 embedded test images in cifar_test_image.h!")
