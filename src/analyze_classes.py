import medmnist
from medmnist import INFO
from collections import Counter
import matplotlib.pyplot as plt

# Load BloodMNIST
info = INFO["bloodmnist"]
DataClass = getattr(medmnist, info["python_class"])

dataset = DataClass(
    split="train",
    download=True
)

class_names = list(info["label"].values())

# Count samples per class
labels = [int(label[0]) for _, label in dataset]
counts = Counter(labels)

# Print counts
print("Training set class distribution:\n")

for class_id in range(8):
    print(f"{class_names[class_id]:60s}: {counts[class_id]}")

# Plot distribution
plt.figure(figsize=(12, 6))

plt.bar(
    [class_names[i] for i in range(8)],
    [counts[i] for i in range(8)]
)

plt.title("BloodMNIST Training Set Class Distribution")
plt.xlabel("Cell Type")
plt.ylabel("Number of Images")
plt.xticks(rotation=35, ha="right")

plt.tight_layout()
plt.savefig("results/class_distribution.png", dpi=150)
plt.show()

print("\nSaved to results/class_distribution.png")