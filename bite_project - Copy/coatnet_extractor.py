"""
coatnet_extractor.py
CoAtNet-based deep feature extraction for bite-mark images.
"""

import torch
import timm
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[CoAtNet] Using device: {DEVICE}")


# ---------------------------------------------------------
# Load pretrained CoAtNet
# ---------------------------------------------------------

MODEL_NAME = "coatnet_0_rw_224.sw_in1k"

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=0
)

model = model.to(DEVICE)
model.eval()


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

def extract_coatnet_features(image_path):
    """
    Extract a deep feature vector from an image using CoAtNet.

    Returns:
        numpy array containing the feature vector.
    """

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model(tensor)

    features = features.squeeze(0)

    # L2 normalization
    features = features / (torch.norm(features) + 1e-8)

    return features.cpu().numpy()


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python coatnet_extractor.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    features = extract_coatnet_features(image_path)

    print()
    print("=== CoAtNet Feature Extraction Test ===")
    print(f"Image: {image_path}")
    print(f"Device: {DEVICE}")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector dtype: {features.dtype}")
    print(f"First 10 values:")
    print(features[:10])