from preprocess import load_and_preprocess_lfw, describe_dataset
from embedding_model import FaceEmbeddingCNN
from sampling import create_pairs, create_triplets
from losses import ContrastiveLoss, TripletLoss

import torch

X_train, X_test, y_train, y_test, target_names = load_and_preprocess_lfw(image_shape=(64, 64))
describe_dataset(X_train, y_train, target_names)

model = FaceEmbeddingCNN(embedding_dim=128)
dummy_input = torch.randn(4, 1, 64, 64)  # batch of 4 grayscale faces
embeddings = model(dummy_input)
print("Output shape:", embeddings.shape)  # (4, 128)

# Pairs
pairs, pair_labels = create_pairs(X_train, y_train)
print("Pairs shape:", pairs.shape)
print("Pair labels shape:", pair_labels.shape)

# Convert to torch tensors
pairs = torch.tensor(pairs[:8], dtype=torch.float32)  # First 8 pairs
pair_labels = torch.tensor(pair_labels[:8], dtype=torch.float32)

# Split into two inputs
img1 = pairs[:, 0]
img2 = pairs[:, 1]

# Get embeddings
embed1 = model(img1)
embed2 = model(img2)

# Compute contrastive loss
contrastive_loss_fn = ContrastiveLoss(margin=1.0)
contrastive_loss = contrastive_loss_fn(embed1, embed2, pair_labels)
print("Contrastive loss:", contrastive_loss.item())

# Triplets
triplets = torch.tensor(create_triplets(X_train, y_train)[:8], dtype=torch.float32)
print("Triplets shape:", triplets.shape)

anchor = triplets[:, 0]
positive = triplets[:, 1]
negative = triplets[:, 2]

# Get embeddings
anchor_out = model(anchor)
positive_out = model(positive)
negative_out = model(negative)

# Compute triplet loss
triplet_loss_fn = TripletLoss(margin=1.0)
triplet_loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
print("Triplet loss:", triplet_loss.item())
