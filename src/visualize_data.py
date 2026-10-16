
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load BloodMNIST
# --------------------------------------------------

info = INFO["bloodmnist"]
DataClass = getattr(medmnist, info["python_class"])

dataset = DataClass(
    split="train",
    download=True
)

# Class names
class_names = list(info["label"].values())

# --------------------------------------------------
# 2. Create a figure
# --------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# --------------------------------------------------
# 3. Find one example of each class
# --------------------------------------------------

found_classes = set()

for image, label in dataset:

    class_id = int(label[0])

    if class_id in found_classes:
        continue

    found_classes.add(class_id)

    row = class_id // 4
    col = class_id % 4

    axes[row, col].imshow(image, interpolation="nearest")
    axes[row, col].set_title(class_names[class_id])
    axes[row, col].axis("off")

    if len(found_classes) == 8:
        break

# --------------------------------------------------
# 4. Save the visualization
# --------------------------------------------------

plt.tight_layout()

plt.savefig(
    "results/bloodmnist_classes.png",
    dpi=150
)

plt.show()

print("Visualization saved to results/bloodmnist_classes.png")

