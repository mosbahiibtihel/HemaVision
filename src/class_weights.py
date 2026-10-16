import torch
import medmnist
from medmnist import INFO
from collections import Counter

# Load BloodMNIST
info = INFO["bloodmnist"]
DataClass = getattr(medmnist, info["python_class"])

dataset = DataClass(
    split="train",
    download=True
)

class_names = list(info["label"].values())

# Get labels
labels = [int(label[0]) for _, label in dataset]

# Count each class
counts = Counter(labels)

# Total number of samples
total = len(labels)

# Number of classes
num_classes = len(class_names)

# Compute balanced class weights So rare classes automatically receive higher weights.
weights = []

for class_id in range(num_classes):
    weight = total / (num_classes * counts[class_id])
    weights.append(weight)

weights = torch.tensor(weights, dtype=torch.float32)

print("Class weights:\n")

for class_id, weight in enumerate(weights):
    print(f"{class_names[class_id]:60s}: {weight:.3f}")

print("\nPyTorch tensor:")
print(weights)