# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from embedding_model import FaceEmbeddingCNN
from losses import ContrastiveLoss, TripletLoss
from sampling import create_pairs, create_triplets
from preprocess import load_and_preprocess_lfw
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# ---------------- Dataset Definitions ----------------
class PairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1 = torch.tensor(self.pairs[idx][0], dtype=torch.float32)
        img2 = torch.tensor(self.pairs[idx][1], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img1, img2, label

class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor = torch.tensor(self.triplets[idx][0], dtype=torch.float32)
        positive = torch.tensor(self.triplets[idx][1], dtype=torch.float32)
        negative = torch.tensor(self.triplets[idx][2], dtype=torch.float32)
        return anchor, positive, negative

# ---------------- Training Function ----------------
def train(model, dataloader, loss_fn, optimizer, loss_type="contrastive", epochs=10, save_path=None):
    model.train()
    losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            if loss_type == "contrastive":
                img1, img2, labels = batch
                embed1 = model(img1)
                embed2 = model(img2)
                loss = loss_fn(embed1, embed2, labels)
            else:
                anchor, positive, negative = batch
                embed_a = model(anchor)
                embed_p = model(positive)
                embed_n = model(negative)
                loss = loss_fn(embed_a, embed_p, embed_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"[{loss_type.capitalize()}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save checkpoint only if model improved
        if avg_loss < best_loss and save_path:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model improved. Saved to {save_path}")

    return losses

# ---------------- Embedding Evaluation ----------------

def find_best_threshold(distances, labels, thresholds=np.linspace(0.1, 2.0, 100)):
    best_acc = 0.0
    best_thresh = 0.0
    for thresh in thresholds:
        preds = (distances < thresh).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def evaluate_embeddings(model, X_test, y_test, method="pairs"):
    model.eval()

    if method == "pairs":
        # Create test pairs (actual images)
        pairs, labels = create_pairs(X_test, y_test)
        img1 = torch.tensor(np.array([p[0] for p in pairs]), dtype=torch.float32)
        img2 = torch.tensor(np.array([p[1] for p in pairs]), dtype=torch.float32)


        with torch.no_grad():
            emb1 = model(img1)
            emb2 = model(img2)
            distances = torch.norm(emb1 - emb2, dim=1).numpy()

        labels_np = np.array(labels)
        best_thresh, best_acc = find_best_threshold(distances, labels_np)
        preds = (distances < best_thresh).astype(int)
        accuracy = np.mean(preds == labels_np)

        print(f"[Contrastive] Best threshold: {best_thresh:.4f}")
        print(f"[Contrastive] Accuracy on test pairs: {accuracy:.4f}")
        print(classification_report(labels, preds))
        print(confusion_matrix(labels, preds))

    elif method == "triplets":
        # Basic triplet evaluation (sanity check – not accuracy-based)
        triplets = create_triplets(X_test, y_test)
        anchor_np = np.array([t[0] for t in triplets])
        positive_np = np.array([t[1] for t in triplets])
        negative_np = np.array([t[2] for t in triplets])

        anchor = torch.from_numpy(anchor_np).float().view(-1, 1, 64, 64)
        positive = torch.from_numpy(positive_np).float().view(-1, 1, 64, 64)
        negative = torch.from_numpy(negative_np).float().view(-1, 1, 64, 64)

        with torch.no_grad():
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            pos_dist = torch.norm(emb_a - emb_p, dim=1)
            neg_dist = torch.norm(emb_a - emb_n, dim=1)

            triplet_margin = 1.0
            violations = (pos_dist + triplet_margin > neg_dist).sum().item()
            total = len(triplets)
            print(f"[Triplet] Triplet violations: {violations}/{total} ({violations/total:.2%})")


# ---------------- Embedding Visualization ----------------
def visualize_embeddings(model, X, y, method="tsne", title="Embeddings"):
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.tensor(X, dtype=torch.float32))
    embeddings = embeddings.numpy()

    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Only t-SNE supported for now.")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=y, cmap="tab10", s=10)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

# ---------------- Embeddings Entry ----------------

def save_embeddings(model, X, y, output_path="embeddings.npz"):
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.tensor(X, dtype=torch.float32))
    embeddings = embeddings.numpy()
    np.savez(output_path, embeddings=embeddings, labels=y)
    print(f"Saved embeddings to {output_path}")


# ---------------- Main Entry ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # Load data once
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_lfw(image_shape=(64, 64))
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    for loss_type in ["triplet"]:
        print(f"\n--- Running training for {loss_type} loss ---")

        model = FaceEmbeddingCNN(embedding_dim=128)

        if loss_type == "contrastive":
            pairs, labels = create_pairs(X_train, y_train)
            dataset = PairDataset(pairs, labels)
            loss_fn = ContrastiveLoss(margin=1.0)
        else:
            triplets = create_triplets(X_train, y_train)
            dataset = TripletDataset(triplets)
            loss_fn = TripletLoss(margin=1.0)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        save_path = f"checkpoints/model_{loss_type}.pth"

        # Train model
        losses = train(model, dataloader, loss_fn, optimizer,
                       loss_type=loss_type, epochs=args.epochs,
                       save_path=save_path)

        # Save loss curve
        plt.plot(losses)
        plt.title(f"{loss_type.capitalize()} Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"logs/{loss_type}_loss_curve.png")
        plt.close()

        # Evaluate embeddings (define this function!)
        evaluate_embeddings(model, X_test, y_test,
                            method="pairs" if loss_type == "contrastive" else "triplets")

        # Visualize embeddings
        visualize_embeddings(model, X_train[:1000], y_train[:1000], title=f"{loss_type.capitalize()} Embeddings")

        save_embeddings(model, X_test[:1000], y_test[:1000],
                        output_path=f"logs/embeddings_{loss_type}.npz")

        # Optionally save results to a file
        with open(f"logs/{loss_type}_summary.txt", "w") as f:
            f.write(f"{loss_type.capitalize()} final loss: {losses[-1]:.4f}\n")
            f.write(f"Checkpoint saved to: {save_path}\n")

            

