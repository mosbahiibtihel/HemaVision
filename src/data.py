import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO


DATASET_NAME = "bloodmnist"
BATCH_SIZE = 32
NUM_WORKERS = 0


def get_dataloaders(batch_size=BATCH_SIZE):
    info = INFO[DATASET_NAME]
    DataClass = getattr(medmnist, info["python_class"])

    # ImageNet normalization for the pretrained EfficientNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Validation/test transformations
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = DataClass(
        split="train",
        transform=train_transform,
        download=True,
    )

    val_dataset = DataClass(
        split="val",
        transform=eval_transform,
        download=True,
    )

    test_dataset = DataClass(
        split="test",
        transform=eval_transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)

    print("\nFirst labels:")
    print(labels[:10].squeeze())