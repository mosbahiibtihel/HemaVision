import medmnist
from medmnist import INFO
from torchvision import transforms

# --------------------------------------------------
# 1. Get dataset information
# --------------------------------------------------

info = INFO["bloodmnist"]
DataClass = getattr(medmnist, info["python_class"])

print("Dataset:", info["description"])
print("Classes:", info["label"])
print("Number of classes:", len(info["label"]))

# --------------------------------------------------
# 2. Define preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
])

# --------------------------------------------------
# 3. Download and load the datasets
# --------------------------------------------------

train_dataset = DataClass(
    split="train",
    transform=transform,
    download=True
)

val_dataset = DataClass(
    split="val",
    transform=transform,
    download=True
)

test_dataset = DataClass(
    split="test",
    transform=transform,
    download=True
)

# --------------------------------------------------
# 4. Print dataset information
# --------------------------------------------------

print("\nDataset sizes:")
print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# --------------------------------------------------
# 5. Inspect one sample
# --------------------------------------------------

image, label = train_dataset[0]

print("\nOne sample:")
print("Image shape:", image.shape)
print("Label:", label)
print("Pixel minimum:", image.min().item())
print("Pixel maximum:", image.max().item())
