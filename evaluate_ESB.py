import os
from utils.config import *
from utils import misc, parser
from tools import builder
import time
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    args.distributed = False
    
    # config
    config = get_config(args)

    # batch size
    config.dataset.train.others.bs = 1
        
    # load dataset
    (_, eval_dataloader) =  builder.dataset_builder(args, config.dataset.train)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    PICKLE_PATH = "feature_dict.pkl"

    if os.path.exists(PICKLE_PATH):
        print(f"Loading existing feature dict from {PICKLE_PATH}")
        with open(PICKLE_PATH, "rb") as f:
            feature_dict = pickle.load(f)
    else:
        feature_dict = {}

        # load ckpt
        builder.load_model(base_model, args.start_ckpts)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for _, (category, object_name, data) in enumerate(eval_dataloader, 0):
            points = data.to(device)
            npoints = config.dataset.train.others.npoints
            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True).squeeze(0).cpu().detach().numpy()
            assert feature.shape == (2*config.model.transformer_config.trans_dim,), f"Unexpected feature shape: {feature.shape}"
            feature_dict[object_name[0]] = {
                'category': category[0],
                'feature': feature
            }

        print(f"Saving feature dict to {PICKLE_PATH}")
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(feature_dict, f)

    NN_precision(feature_dict)
    visualize_features(feature_dict, highlight_category="Gear like Parts")



def NN_precision(feature_dict):
    """Compute nearest neighbor precision for each category."""
    names = list(feature_dict.keys())
    features = np.vstack([feature_dict[n]["feature"] for n in names])
    categories = [feature_dict[n]["category"] for n in names]

    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(features)
    distances, indices = nbrs.kneighbors(features)

    correct = 0
    total = len(names)

    for i in range(total):
        if categories[i] == categories[indices[i,1]]:  # indices[i][0] is itself
            correct += 1

    precision = correct / total
    print(f"Nearest neighbor precision: {precision:.4f}")


def visualize_features(feature_dict, highlight_category="Gear like Parts"):
    """Reduce features to 2D with t-SNE and plot, highlighting one category in red."""
    names = list(feature_dict.keys())
    features = np.vstack([feature_dict[n]["feature"] for n in names])
    categories = [feature_dict[n]["category"] for n in names]

    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(names) - 1))
    features_2d = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # All other categories in grey
    other_mask = [c != highlight_category for c in categories]
    ax.scatter(
        features_2d[other_mask, 0],
        features_2d[other_mask, 1],
        c="lightgrey",
        s=10,
        label="Other",
        alpha=0.6,
    )

    # Highlighted category in red
    highlight_mask = [c == highlight_category for c in categories]
    ax.scatter(
        features_2d[highlight_mask, 0],
        features_2d[highlight_mask, 1],
        c="red",
        s=20,
        label=highlight_category,
        alpha=0.9,
    )

    ax.legend()
    ax.set_title(f"t-SNE feature space — '{highlight_category}' highlighted")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig("features_tsne.png", dpi=150)
    print("Saved plot to features_tsne.png")
    plt.show()



if __name__ == '__main__':

    main()

