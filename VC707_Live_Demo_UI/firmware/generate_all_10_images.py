import os
import pickle
import numpy as np
from PIL import Image

data_file = r"C:\Mubashir-BTU\Thesis\Codes\Danial\DATE2027_Eval\VGG11\Python_Reference_Model\data\cifar-10-batches-py\test_batch"
h_file = r"C:\Mubashir-BTU\Thesis\Codes\Danial\DATE2027_Eval\VC707_Live_Demo_UI\firmware\cifar_test_image.h"
sample_dir = r"C:\Mubashir-BTU\Thesis\Codes\Danial\DATE2027_Eval\VC707_Live_Demo_UI\python_ui\sample_images"
os.makedirs(sample_dir, exist_ok=True)

with open(data_file, "rb") as f:
    d = pickle.load(f, encoding="bytes")

data = d[b"data"]
labels = d[b"labels"]

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class_emojis = [
    "✈️", "🚗", "🐦", "🐱", "🦌",
    "🐶", "🐸", "🐎", "🚢", "🚚"
]

# Find first occurrence of each of the 10 classes
target_indices = {}
for i, lbl in enumerate(labels):
    cname = class_names[lbl]
    if cname not in target_indices:
        target_indices[cname] = (i, lbl)
    if len(target_indices) == 10:
        break

mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
std  = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
in_scale = 0.035773955285549164

lines = ["#ifndef CIFAR_TEST_IMAGE_H", "#define CIFAR_TEST_IMAGE_H", "", "#include <stdint.h>", ""]

for cls_idx, cname in enumerate(class_names):
    test_idx, label_id = target_indices[cname]
    raw_chw = data[test_idx].reshape(3, 32, 32)
    raw_hwc = raw_chw.transpose(1, 2, 0)
    
    # Save crisp PNG preview
    pil_img = Image.fromarray(raw_hwc)
    pil_img_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    png_path = os.path.join(sample_dir, f"sample_{cls_idx}_{cname}.png")
    pil_img_resized.save(png_path)
    print(f"Exported PNG: {png_path}")
    
    # Quantize for C header
    raw_f32 = raw_chw.astype(np.float32) / 255.0
    norm = (raw_f32 - mean) / std
    q = np.clip(np.round(norm / in_scale), -128, 127).astype(np.int8)
    flat = q.flatten()
    
    var_name = f"cifar_image_{cls_idx}_{cname}"
    lines.append(f"/* Sample Image #{cls_idx}: {cname.upper()} (Class ID: {cls_idx}, Test Batch Index: {test_idx}) */")
    lines.append(f"static const int8_t {var_name}[3072] = {{")
    for r in range(0, len(flat), 16):
        chunk = flat[r:r+16]
        c_str = ", ".join(f"{v:4d}" for v in chunk)
        if r + 16 < len(flat):
            lines.append(f"    {c_str},")
        else:
            lines.append(f"    {c_str}")
    lines.append("};")
    lines.append("")

lines.append("/* Pointers to all 10 embedded test images */")
lines.append("static const int8_t * const cifar_test_images[10] = {")
for cls_idx, cname in enumerate(class_names):
    comma = "," if cls_idx < 9 else ""
    lines.append(f"    cifar_image_{cls_idx}_{cname}{comma}")
lines.append("};")
lines.append("")
lines.append("static const int cifar_test_labels[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};")
lines.append('static const char * const cifar_test_names[10] = {')
for cls_idx, cname in enumerate(class_names):
    comma = "," if cls_idx < 9 else ""
    lines.append(f'    "{cname}"{comma}')
lines.append("};")
lines.append("")
lines.append("/* Backward compatibility alias */")
lines.append("#define cifar_image_0 cifar_image_3_cat")
lines.append("")
lines.append("#endif /* CIFAR_TEST_IMAGE_H */")

with open(h_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Successfully generated all 10 embedded test images in {h_file}!")
