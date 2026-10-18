import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from data import get_dataloaders
from model import build_model
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "models/hemavision_efficientnet_b0.pth"

CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocytes",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]


def main():

    # -----------------------------
    # Device
    # -----------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # -----------------------------
    # Load test data
    # -----------------------------

    _, _, test_loader = get_dataloaders()

    # -----------------------------
    # Load model
    # -----------------------------

    model = build_model()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # -----------------------------
    # Predictions
    # -----------------------------

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.squeeze().long()

            outputs = model(images)

            predictions = outputs.argmax(dim=1).cpu()

            all_predictions.extend(
                predictions.numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    # -----------------------------
    # Metrics
    # -----------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    print("\n===== TEST RESULTS =====")

    print(
        f"Test Accuracy: {accuracy:.4f}"
    )

    print(
        f"Test Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    # -----------------------------
    # Classification report
    # -----------------------------

    print("\n===== CLASSIFICATION REPORT =====")

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )

    print(report)

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("===== CONFUSION MATRIX =====")
    print(cm)

    # -----------------------------
    # Confusion matrix visualization
    # -----------------------------

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues"
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("HemaVision - BloodMNIST Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nConfusion matrix saved to:")
    print("results/confusion_matrix.png")


if __name__ == "__main__":
    main()