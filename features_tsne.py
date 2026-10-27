import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def load_feature_dict(feature_path, epoch):
    """Load a features_{epoch}.json file produced by the training pipeline."""
    json_path = os.path.join(feature_path, f"features_{epoch}.json")
    with open(json_path, "r") as f:
        return json.load(f)


def load_label_map(label_map_path):
    with open(label_map_path, "r") as f:
        return json.load(f)


def plot_tsne_by_class(
    feature_path,
    label_map_path,
    epoch=0,
    perplexity=30,
    random_state=42,
    save_path=None,
    figsize=(12, 10),
    point_size=14,
):
    """
    Create a t-SNE plot of features for a given epoch, coloring every ESB
    class (as defined in label_map.json) with its own shade.
    """
    feature_dict = load_feature_dict(feature_path, epoch)
    label_map = load_label_map(label_map_path)

    names = list(feature_dict.keys())
    features = np.vstack([feature_dict[n]["feature"] for n in names])
    categories = [feature_dict[n]["category"] for n in names]

    # Map category string -> integer label using label_map.json
    labels = np.array([label_map[c] for c in categories])
    n_classes = len(label_map)

    perplexity = min(perplexity, max(2, len(names) - 1))
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)

    # Distinct shade per class using a large qualitative/continuous colormap
    cmap = plt.get_cmap("gist_ncar", n_classes)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap=cmap,
        vmin=0,
        vmax=n_classes - 1,
        s=point_size,
        alpha=0.85,
    )

    # Colorbar with class names as ticks (may be crowded with 45 classes)
    inv_label_map = {v: k for k, v in label_map.items()}
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(n_classes), pad=0.01)
    cbar.ax.set_yticklabels([inv_label_map[i] for i in range(n_classes)], fontsize=6)

    ax.set_title(f"t-SNE feature space (epoch {epoch}) — colored by ESB class")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()

    return features_2d, labels


if __name__ == "__main__":
    FEATURES_PATH = "experiments/RI_MAE/SSL_models/pretrain_shapenet/features"
    LABEL_MAP_PATH = "data/ESB/label_map.json"

    plot_tsne_by_class(
        feature_path=FEATURES_PATH,
        label_map_path=LABEL_MAP_PATH,
        epoch=0,
        save_path="tsne_by_class.png",
        point_size=36
    )