import torch

from model import build_model


MODEL_PATH = "models/hemavision_efficientnet_b0.pth"


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = build_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

target_layer = model.features[-1][0]

print("Target layer:")
print(target_layer)

captured = {}


def forward_hook(module, inputs, output):
    captured["activation"] = output

    print(
        "FORWARD HOOK:",
        output.shape,
        "requires_grad =",
        output.requires_grad
    )


def backward_hook(module, grad_input, grad_output):
    print(
        "BACKWARD HOOK:",
        grad_output[0]
    )

    if grad_output[0] is not None:
        captured["gradient"] = grad_output[0]


target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


x = torch.randn(
    1,
    3,
    224,
    224,
    device=device,
    requires_grad=True
)

print("\nRunning forward pass...")

output = model(x)

print(
    "Model output:",
    output.shape
)

print(
    "Output requires_grad:",
    output.requires_grad
)

target = output[0, 0]

print(
    "Target requires_grad:",
    target.requires_grad
)

print("\nRunning backward pass...")

model.zero_grad()

target.backward()

print("\n===== RESULT =====")

print(
    "Activation captured:",
    "activation" in captured
)

print(
    "Gradient captured:",
    "gradient" in captured
)