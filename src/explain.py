import torch
import numpy as np
import matplotlib.pyplot as plt

import medmnist
from medmnist import INFO

from torchvision import transforms

from model import build_model


DATASET_NAME = "bloodmnist"
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load trained model
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

    # Load BloodMNIST test set
    info = INFO[DATASET_NAME]
    DataClass = getattr(
        medmnist,
        info["python_class"]
    )

    dataset = DataClass(
        split="test",
        transform=None,
        download=True
    )

    # Select an image
    image, label = dataset[0]

    true_class = int(label[0])

    print(
        "True class:",
        CLASS_NAMES[true_class]
    )

    # Transform for EfficientNet
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = eval_transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    # EfficientNet final convolutional layer
    target_layer = model.features[-1][0]

    print(
        "Grad-CAM target layer:",
        target_layer
    )

    # Storage for activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    # Forward pass
    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_class = outputs.argmax(
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    print(
        "Predicted class:",
        CLASS_NAMES[predicted_class]
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    # Backward pass for predicted class
    model.zero_grad()

    target_score = outputs[
        0,
        predicted_class
    ]

    target_score.backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    if not activations:
        raise RuntimeError(
            "No activations were captured."
        )

    if not gradients:
        raise RuntimeError(
            "No gradients were captured."
        )

    activation = activations[0]
    gradient = gradients[0]

    print(
        "Activation shape:",
        activation.shape
    )

    print(
        "Gradient shape:",
        gradient.shape
    )

    # Grad-CAM weights
    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Weighted combination of feature maps
    cam = (
        weights * activation
    ).sum(dim=1)

    # Keep positive influence
    cam = torch.relu(cam)

    cam = cam[0]

    # Convert to NumPy
    cam = cam.detach().cpu().numpy()

    # Normalize
    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    # Resize heatmap to 224x224
    heatmap = torch.tensor(cam)

    heatmap = heatmap.unsqueeze(0).unsqueeze(0)

    heatmap = torch.nn.functional.interpolate(
        heatmap,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    heatmap = heatmap[0, 0].numpy()

    # Prepare original image
    resized_image = transforms.Resize(
        (224, 224)
    )(image)

    rgb_image = np.array(
        resized_image
    ).astype(np.float32) / 255.0

    # Create colored heatmap
    cmap = plt.get_cmap("jet")

    colored_heatmap = cmap(
        heatmap
    )[:, :, :3]

    # Overlay
    visualization = (
        0.5 * rgb_image
        + 0.5 * colored_heatmap
    )

    visualization = np.clip(
        visualization,
        0,
        1
    )

    # Save result
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title(
        f"Original\nTrue: {CLASS_NAMES[true_class]}"
    )
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(
        heatmap,
        cmap="jet"
    )
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(visualization)
    plt.title(
        f"Prediction\n"
        f"{CLASS_NAMES[predicted_class]}"
    )
    plt.axis("off")

    plt.tight_layout()

    output_path = "results/gradcam_sample.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Grad-CAM saved to:")
    print(output_path)


if __name__ == "__main__":
    main()