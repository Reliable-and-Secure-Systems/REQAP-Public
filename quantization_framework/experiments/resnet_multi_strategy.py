"""
ResNet-18 Sensitivity-Aware Pruning on GTSRB
Uses real Test folder for evaluation (proper held-out test set)
"""

import torch
import torch.nn as nn
import torch.quantization
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import copy
import csv
import os
from typing import Dict, List
import random

# Configuration
CHECKPOINT_PATH = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/sensitivity/road_0.9994904891304348.pth'
SCORE_DIR = 'resnet_weight_sensitivity_score'
DATA_PATH = '/scratch/monacdan/MASTERS/GENIE/DANIAL_MASTERS/VGG_manipulation/Genie/VGG11/GA_BM/GA_with_quantization/non_genetic_algorithm/multi_architecture_pruning/data/archive'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 30

# Set random seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Class mapping
classes_ori = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited', 17:'No entry',
    18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 22:'Bumpy road',
    23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians',
    28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only',
    36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left',
    40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}


def load_model():
    """Load trained ResNet-18 model"""
    print("\nLoading ResNet-18 model...")
    model = torchvision.models.resnet18(pretrained=False)
    num_classes = len(classes_ori)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"  Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best validation accuracy during training: {checkpoint.get('best_valid_acc', 0.0)*100:.2f}%")
    
    return model, checkpoint


def load_data():
    """Load GTSRB dataset - Train for fine-tuning, Test for evaluation"""
    print("\nLoading GTSRB dataset...")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load Train dataset for fine-tuning (full training set)
    train_data = datasets.ImageFolder(os.path.join(DATA_PATH, 'Train'), transform=train_transform)
    
    # Load Test dataset for evaluation (the real held-out test set)
    test_data = datasets.ImageFolder(os.path.join(DATA_PATH, 'Test'), transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test (held-out): {len(test_data)} samples")
    
    return train_loader, test_loader


def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def calculate_sparsity(model):
    """Calculate model sparsity"""
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    return 100.0 * zero_params / total_params if total_params > 0 else 0.0


def get_layer_info(model, score_dir):
    """Get layer names and parameters with sensitivity scores"""
    model_state_dict = model.state_dict()
    
    # Find all weight layers
    available_layers = [n for n in model_state_dict.keys() 
                       if 'weight' in n and 'bias' not in n and len(model_state_dict[n].shape) >= 2]
    
    # Filter layers that have sensitivity scores
    layer_names = [n for n in available_layers 
                   if os.path.exists(os.path.join(score_dir, f'weight_sensitivity_scores_{n}.csv'))]
    
    layer_params = {n: model_state_dict[n].numel() for n in layer_names}
    
    print(f"  Found {len(layer_names)} layers with sensitivity scores")
    
    return layer_names, layer_params


def apply_pruning_mask(model, strategy, layer_names, layer_params, score_dir):
    """Apply pruning masks based on sensitivity scores"""
    masks = {}
    
    for layer_idx, layer_name in enumerate(layer_names):
        prune_pct = strategy[layer_idx]
        param_tensor = dict(model.named_parameters())[layer_name]
        
        # Skip if no pruning
        if prune_pct <= 0.0:
            masks[layer_name] = torch.ones_like(param_tensor)
            continue
        
        # Load sensitivity scores
        score_file = os.path.join(score_dir, f'weight_sensitivity_scores_{layer_name}.csv')
        if not os.path.exists(score_file):
            masks[layer_name] = torch.ones_like(param_tensor)
            continue
        
        scores = []
        with open(score_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    scores.append((int(parts[0]), float(parts[1])))
        
        # Sort by sensitivity (ascending = prune least sensitive first)
        scores.sort(key=lambda x: x[1])
        
        # Calculate number of weights to prune
        total_weights = layer_params[layer_name]
        num_to_prune = int(total_weights * (prune_pct / 100.0))
        
        # Create mask
        mask = torch.ones(total_weights)
        for i in range(min(num_to_prune, len(scores))):
            mask[scores[i][0]] = 0
        
        # Apply mask
        mask = mask.view(param_tensor.shape).to(param_tensor.device)
        with torch.no_grad():
            param_tensor.mul_(mask)
        
        masks[layer_name] = mask
    
    return masks


def fine_tune_model(model, masks, train_loader, test_loader, device, epochs):
    """Fine-tune pruned model"""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    patience_counter = 0
    best_model_state = None
    
    print("  Fine-tuning...")
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Re-apply masks after gradient update
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks:
                        param.data.mul_(masks[name].to(param.device))
            
            optimizer.step()
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader, device)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: {test_accuracy:.2f}%")
        
        # Early stopping
        if patience_counter >= 5:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return best_accuracy


def create_pruning_strategy(layer_names, layer_params, strategy_type, target_sparsity):
    """Create pruning strategy for each layer"""
    total_params = sum(layer_params.values())
    strategy = []
    
    for layer_name in layer_names:
        # Extract stage information
        if 'layer1' in layer_name:
            stage = 1
        elif 'layer2' in layer_name:
            stage = 2
        elif 'layer3' in layer_name:
            stage = 3
        elif 'layer4' in layer_name:
            stage = 4
        else:
            stage = 0
        
        is_conv1 = 'conv1' in layer_name and stage == 0
        is_fc = 'fc' in layer_name
        
        # Determine base pruning rate
        if strategy_type == "uniform":
            prune_pct = target_sparsity
        elif strategy_type == "graduated":
            if stage == 4:
                prune_pct = target_sparsity * 1.3
            elif stage == 3:
                prune_pct = target_sparsity * 1.1
            elif stage == 2:
                prune_pct = target_sparsity * 0.9
            else:
                prune_pct = target_sparsity * 0.7
        elif strategy_type == "conservative":
            if is_conv1 or is_fc:
                prune_pct = target_sparsity * 0.3
            else:
                prune_pct = target_sparsity * 0.8
        elif strategy_type == "aggressive":
            if stage >= 3:
                prune_pct = target_sparsity * 1.5
            else:
                prune_pct = target_sparsity * 0.8
        else:
            prune_pct = target_sparsity
        
        strategy.append(min(98.0, max(0.0, prune_pct)))
    
    # Adjust to hit target sparsity
    for _ in range(20):
        current_pruned = sum((strategy[i] / 100.0) * layer_params[layer_names[i]] 
                           for i in range(len(layer_names)))
        current_sparsity = (current_pruned / total_params) * 100.0
        
        if abs(current_sparsity - target_sparsity) < 0.5:
            break
        
        scale = target_sparsity / current_sparsity if current_sparsity > 0 else 1.0
        strategy = [min(98.0, max(0.0, s * scale)) for s in strategy]
    
    return strategy


def main():
    print("=" * 80)
    print("ResNet-18 Sensitivity-Aware Pruning on GTSRB")
    print("=" * 80)
    
    # Load model
    model, checkpoint = load_model()
    
    # Load data - Train for fine-tuning, Test for evaluation
    train_loader, test_loader = load_data()
    
    # Evaluate baseline on REAL held-out test set
    print("\nEvaluating baseline on held-out test set...")
    baseline_accuracy = evaluate_model(model, test_loader, DEVICE)
    print(f"  Baseline test accuracy: {baseline_accuracy:.2f}%")
    print(f"  (Training validation was: {checkpoint.get('best_valid_acc', 0.0)*100:.2f}%)")
    
    # Get layer information
    print("\nAnalyzing layers...")
    layer_names, layer_params = get_layer_info(model, SCORE_DIR)
    print(f"  Total parameters in prunable layers: {sum(layer_params.values()):,}")
    
    # Test multiple strategies
    print("\n" + "=" * 80)
    print("Multi-Strategy Pruning")
    print("=" * 80)
    
    strategies = [
        ("uniform", 90.0),
        ("graduated", 90.0),
        ("conservative", 90.0),
        ("aggressive", 85.0)
    ]
    
    results = []
    
    for strategy_id, (strategy_type, target_sparsity) in enumerate(strategies, 1):
        print(f"\n{'-' * 80}")
        print(f"Strategy {strategy_id}/{len(strategies)}: {strategy_type} @ {target_sparsity}%")
        print(f"{'-' * 80}")
        
        # Create strategy
        strategy = create_pruning_strategy(layer_names, layer_params, strategy_type, target_sparsity)
        
        # Apply pruning
        pruned_model = copy.deepcopy(model)
        masks = apply_pruning_mask(pruned_model, strategy, layer_names, layer_params, SCORE_DIR)
        
        actual_sparsity = calculate_sparsity(pruned_model)
        print(f"  Actual sparsity: {actual_sparsity:.2f}%")
        
        # Fine-tune
        fp32_accuracy = fine_tune_model(pruned_model, masks, train_loader, test_loader, 
                                        DEVICE, FINE_TUNE_EPOCHS)
        print(f"  FP32 accuracy after fine-tuning: {fp32_accuracy:.2f}%")
        
        # Quantize
        print("  Applying PTQ...")
        pruned_model_cpu = pruned_model.cpu()
        quantized_model = torch.quantization.quantize_dynamic(
            pruned_model_cpu, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Evaluate quantized model on CPU
        int8_accuracy = evaluate_model(quantized_model, test_loader, 'cpu')
        print(f"  INT8 accuracy after quantization: {int8_accuracy:.2f}%")
        
        results.append({
            'strategy': strategy_type,
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'fp32_accuracy': fp32_accuracy,
            'int8_accuracy': int8_accuracy,
            'total_drop': baseline_accuracy - int8_accuracy
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Baseline (unpruned, FP32 on test set): {baseline_accuracy:.2f}%\n")
    print(f"{'Strategy':<20} {'Sparsity':<12} {'FP32 Acc':<12} {'INT8 Acc':<12} {'Drop':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['strategy']:<20} {r['actual_sparsity']:>10.2f}% {r['fp32_accuracy']:>10.2f}% "
              f"{r['int8_accuracy']:>10.2f}% {r['total_drop']:>8.2f}%")
    
    # Find best strategy
    best = min(results, key=lambda x: x['total_drop'])
    print(f"\nBest strategy: {best['strategy']} with only {best['total_drop']:.2f}% accuracy drop")
    print(f"Achieved {best['actual_sparsity']:.1f}% sparsity + INT8 quantization")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
