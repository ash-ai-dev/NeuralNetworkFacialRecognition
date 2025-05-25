# visualize_embeddings.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os

def plot_embeddings(embeddings, labels, title="Embeddings", save_path="embedding_plot.png"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
    plt.legend(*scatter.legend_elements(), title="Classes", loc="best", fontsize=8)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="logs/embeddings_contrastive.npz", help="Path to .npz embedding file")
    parser.add_argument("--title", type=str, default="Embedding Visualization", help="Plot title")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        return

    data = np.load(args.file)
    embeddings = data["embeddings"]
    labels = data["labels"]

    base_name = os.path.splitext(os.path.basename(args.file))[0]
    save_path = f"logs/{base_name}_tsne.png"

    plot_embeddings(embeddings, labels, title=args.title, save_path=save_path)

if __name__ == "__main__":
    main()
