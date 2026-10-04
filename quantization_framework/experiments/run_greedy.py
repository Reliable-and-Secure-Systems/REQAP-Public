import subprocess
import sys

cmd = [
    sys.executable,
    "quantization_framework/experiments/joint_granular_search.py",
    "--model", "levit",
    "--checkpoint", "models/best_levit_model_cifar100.pth",
    "--sensitivity", "levit_cifar100_channel_sensitivity.csv",
    "--dataset", "cifar100",
    "--bits", "2", "4", "8",
    "--target-drop", "3.0",
    "--max-drop", "3.0",
    "--output", "levit_cifar100_csweep",
    "--act-errors", "levit_cifar100_activation_errors.json",
    "--eval-batches", "20",
]

subprocess.run(cmd)
