import numpy as np
import random

def create_pairs(images, labels):
    """
    Generate positive and negative pairs for contrastive loss training.

    Returns:
        pair_images: list of image pairs [img1, img2]
        pair_labels: 1 if same class, 0 if different class
    """
    pair_images = []
    pair_labels = []
    label_to_indices = {}

    # Build a map from label to indices
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    labels_set = list(label_to_indices.keys())
    n_classes = len(labels_set)

    for label in labels_set:
        indices = label_to_indices[label]
        for anchor_idx in indices:
            # Positive pair
            positive_idx = anchor_idx
            while positive_idx == anchor_idx:
                positive_idx = random.choice(indices)
            pair_images.append([images[anchor_idx], images[positive_idx]])
            pair_labels.append(1)

            # Negative pair
            negative_label = label
            while negative_label == label:
                negative_label = random.choice(labels_set)
            negative_idx = random.choice(label_to_indices[negative_label])
            pair_images.append([images[anchor_idx], images[negative_idx]])
            pair_labels.append(0)

    return np.array(pair_images), np.array(pair_labels)


def create_triplets(images, labels):
    """
    Generate triplets (anchor, positive, negative) for triplet loss training.

    Returns:
        triplets: list of (anchor, positive, negative)
    """
    triplets = []
    label_to_indices = {}

    # Build a map from label to indices
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    labels_set = list(label_to_indices.keys())

    for label in labels_set:
        indices = label_to_indices[label]
        for anchor_idx in indices:
            # Positive sample (same class)
            positive_idx = anchor_idx
            while positive_idx == anchor_idx:
                positive_idx = random.choice(indices)

            # Negative sample (different class)
            negative_label = label
            while negative_label == label:
                negative_label = random.choice(labels_set)
            negative_idx = random.choice(label_to_indices[negative_label])

            triplets.append([images[anchor_idx], images[positive_idx], images[negative_idx]])

    return np.array(triplets)
