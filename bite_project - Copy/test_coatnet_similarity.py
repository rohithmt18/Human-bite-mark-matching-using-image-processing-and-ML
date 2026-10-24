import numpy as np
from coatnet_extractor import extract_coatnet_features


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two feature vectors.
    """
    similarity = np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    return float(similarity)


# ---------------------------------------------------------
# Images to compare
# ---------------------------------------------------------

image1 = r".\datasets\P010\vishmaya lower.jpg"

image2 = r".\datasets\P010\vishmaya upper.jpg"

image3 = r".\datasets\P009\Rishi lower.jpg"


# ---------------------------------------------------------
# Extract features
# ---------------------------------------------------------

print("\nExtracting features...")

features1 = extract_coatnet_features(image1)
features2 = extract_coatnet_features(image2)
features3 = extract_coatnet_features(image3)


# ---------------------------------------------------------
# Calculate similarities
# ---------------------------------------------------------

same_person = cosine_similarity(features1, features2)

different_person = cosine_similarity(features1, features3)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n========================================")
print("       CoAtNet Similarity Test")
print("========================================")

print(f"\nVishmaya lower ↔ Vishmaya upper")
print(f"Cosine similarity: {same_person:.4f}")
print(f"Percentage: {same_person * 100:.2f}%")

print(f"\nVishmaya lower ↔ Rishi lower")
print(f"Cosine similarity: {different_person:.4f}")
print(f"Percentage: {different_person * 100:.2f}%")

print("\n========================================")