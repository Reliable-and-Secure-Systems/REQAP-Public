import os
import torch
import torchvision
import torchvision.transforms as transforms

def export_image():
    print("Downloading CIFAR-10 test set...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to typical 8-bit signed range roughly (just for simulation data)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    # Get the first image (a 'cat')
    image, label = testset[0]
    
    # Quantize to int8 (-128 to 127)
    # The image is roughly -1.0 to 1.0 from the normalization
    image_int8 = torch.clamp(torch.round(image * 127.0), -128, 127).to(torch.int8)
    
    # Export to C header
    out_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cifar_test_image.h")
    
    with open(out_path, "w") as f:
        f.write("/* Auto-generated CIFAR-10 Test Image 0 */\n")
        f.write(f"/* True Label: {testset.classes[label]} ({label}) */\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("const int8_t cifar_image_0[3 * 32 * 32] = {\n    ")
        
        flat_img = image_int8.flatten().tolist()
        for i, val in enumerate(flat_img):
            f.write(f"{val}, ")
            if (i + 1) % 16 == 0:
                f.write("\n    ")
        
        f.write("\n};\n")
        
    print(f"Exported image to {out_path}")

if __name__ == "__main__":
    export_image()
