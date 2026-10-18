import os
import medmnist
from medmnist import INFO
from PIL import Image

DATASET_NAME = "bloodmnist"

CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature_granulocytes",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]

info = INFO[DATASET_NAME]
DataClass = getattr(medmnist, info["python_class"])

dataset = DataClass(
    split="test",
    download=True
)

output_dir = "results/test_samples"
os.makedirs(output_dir, exist_ok=True)

# Find one example of each class
found = {}

for i in range(len(dataset)):

    image, label = dataset[i]

    label = int(label[0])

    if label not in found:

        filename = f"{label}_{CLASS_NAMES[label]}.png"
        path = os.path.join(output_dir, filename)

        image.save(path)

        found[label] = path

        print(f"Saved: {path}")

    if len(found) == 8:
        break

print("\nDone!")
print(f"Saved {len(found)} images.")