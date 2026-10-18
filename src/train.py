import torch
from torch.optim import Adam
from sklearn.metrics import f1_score

from data import get_dataloaders
from model import build_model


# -----------------------------
# Configuration
# -----------------------------

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3

MODEL_PATH = "models/hemavision_efficientnet_b0.pth"

CLASS_WEIGHTS = torch.tensor([
    1.7545,  # basophil
    0.6854,  # eosinophil
    1.3778,  # erythroblast
    0.7378,  # immature granulocytes
    1.7607,  # lymphocyte
    1.5054,  # monocyte
    0.6416,  # neutrophil
    0.9098,  # platelet
], dtype=torch.float32)


# -----------------------------
# Training
# -----------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0
    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        # 1. Reset previous gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(images)

        # 3. Calculate error
        loss = criterion(outputs, labels)

        # 4. Backpropagation
        loss.backward()

        # 5. Update model weights
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    average_loss = total_loss / len(loader)

    accuracy = (
        sum(
            p == y
            for p, y in zip(
                all_predictions,
                all_labels
            )
        )
        / len(all_labels)
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return average_loss, accuracy, macro_f1


# -----------------------------
# Validation
# -----------------------------

def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = total_loss / len(loader)

    accuracy = (
        sum(
            p == y
            for p, y in zip(
                all_predictions,
                all_labels
            )
        )
        / len(all_labels)
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return average_loss, accuracy, macro_f1


# -----------------------------
# Main
# -----------------------------

def main():

    # CPU or GPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Load datasets
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    # Build EfficientNet
    model = build_model()

    model = model.to(device)

    # Class-weighted loss
    class_weights = CLASS_WEIGHTS.to(device)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Optimizer
    optimizer = Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    best_f1 = 0

    # -----------------------------
    # Epoch loop
    # -----------------------------

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        # Training
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Validation
        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f} | "
            f"Train Macro F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("Best model saved!")


if __name__ == "__main__":
    main()