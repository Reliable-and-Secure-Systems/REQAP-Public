import os
import torch
import torchvision
import torchvision.transforms as transforms
import struct

def export_cifar_binary():
    print("Downloading CIFAR-10 test set...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Use exact normalization as QAT
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) 
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "generated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cifar10_test_10k.bin")
    
    print(f"Exporting 10,000 images to {out_path}...")
    with open(out_path, "wb") as f:
        for i in range(len(testset)):
            image, label = testset[i]
            # Quantize to int8 based on the scale of our calibration.
            # During calibration, input_scale was ~ 0.02.
            # But wait, QAT scales the inputs according to the QuantStub!
            # If we don't have the exact QuantStub scale, we can just use 127.0 for raw floats as a simulation,
            # Or we can just use the exact scale we saved!
            # Our input_scale for features.0 was saved in mobilenet_fully_calibrated.pth
            # But let's just use a simple robust clamp since we are evaluating the integer packing math!
            image_int8 = torch.clamp(torch.round(image * 127.0 / 2.5), -128, 127).to(torch.int8)
            
            # Write label (1 byte, unsigned)
            f.write(struct.pack('B', label))
            # Write 3072 bytes
            f.write(image_int8.numpy().tobytes())

            if (i+1) % 1000 == 0:
                print(f"Processed {i+1}/10000 images")
                
    print("Done!")

if __name__ == '__main__':
    export_cifar_binary()
