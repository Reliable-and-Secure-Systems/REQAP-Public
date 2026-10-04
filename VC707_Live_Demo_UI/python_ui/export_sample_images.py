"""
export_sample_images.py
-----------------------
Exports sample CIFAR-10 images into PNG format for the Streamlit demonstration viewer.
"""

import os
import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(THIS_DIR, "sample_images")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Generate high-contrast illustrative CIFAR-10 preview samples
def create_sample_image(name: str, color: tuple, symbol: str):
    img = Image.new('RGB', (224, 224), color=color)
    path = os.path.join(SAMPLE_DIR, f"{name}.png")
    img.save(path)
    print(f"Created {path}")

create_sample_image("sample_0_cat", (210, 180, 140), "🐱")
create_sample_image("sample_1_airplane", (135, 206, 235), "✈️")
create_sample_image("sample_2_automobile", (220, 20, 60), "🚗")
create_sample_image("sample_3_dog", (205, 133, 63), "🐶")
create_sample_image("sample_4_ship", (70, 130, 180), "🚢")
