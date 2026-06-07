import argparse
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA


def load_point_clouds(h5_path, dataset_key="point_clouds"):
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        print("Dataset keys:", list(f.keys()))

        if dataset_key not in f:
            raise KeyError(
                f"Dataset '{dataset_key}' not found. Available keys: {list(f.keys())}"
            )

        point_clouds = np.asarray(f[dataset_key][:], dtype=np.float64)

        mesh_names = None
        if "mesh_names" in f:
            mesh_names = f["mesh_names"][:]
            mesh_names = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in mesh_names
            ]

    return point_clouds, mesh_names


def center_and_scale(point_clouds):
    """
    Normalize each shape
    """
    centered = point_clouds - point_clouds.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(centered.reshape(centered.shape[0], -1), axis=1)
    scale = np.maximum(scale, 1e-12)
    normalized = centered / scale[:, None, None]
    return normalized


def fit_pca(point_clouds, n_components):
    """
    Create the matrix X of shape (n_shapes, n_points * 3) and fit PCA
    """
    n_shapes, n_points, _ = point_clouds.shape
    X = point_clouds.reshape(n_shapes, n_points * 3)

    max_components = min(n_components, n_shapes)
    pca = PCA(n_components=max_components)
    scores = pca.fit_transform(X)

    return pca, scores

def components_for_explained_variance(pca, threshold=0.95):
    """
    Search for the number of components k needed to reach a threshold of explained variance
    """
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k = np.searchsorted(cumulative, threshold) + 1
    return k, cumulative


def reconstruct_shape(pca, scores, shape_index, n_points, n_components):
    """
    Reconstruct a shape using the first n_components from PCA
    The reconstruction iis given by: mean_shape + coefficients @ eigenvectors
    """
    coeffs_k = scores[shape_index, :n_components]
    components_k = pca.components_[:n_components, :]
    reconstructed = pca.mean_ + coeffs_k @ components_k

    return reconstructed.reshape(n_points, 3)


def pca_mode_shape(pca, n_points, component_index, sigma):
    """
    Visualize how the mean shape changes along a PCA direction of U_k :
    Create a shape by moving along a PCA mode by sigma and standard deviation related to the corresponding eigenvalue.
    """
    eigenvalue = pca.explained_variance_[component_index]
    std = np.sqrt(eigenvalue)
    vector = pca.mean_ + sigma * std * pca.components_[component_index]
    return vector.reshape(n_points, 3)


def save_point_cloud(path, points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points, fmt="%.8f")


def save_outputs(point_clouds, pca, scores, output_dir, k95 = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_shapes, n_points, _ = point_clouds.shape

    np.save(output_dir / "pca_mean.npy", pca.mean_.reshape(n_points, 3))
    np.save(output_dir / "eigenvectors.npy", pca.components_)
    np.save(output_dir / "pca_scores.npy", scores)
    np.save(output_dir / "explained_variance_ratio.npy", pca.explained_variance_ratio_)
    np.save(output_dir / "explained_variance.npy", pca.explained_variance_)

    save_point_cloud(output_dir / "mean_point_cloud.xyz", pca.mean_.reshape(n_points, 3))

    for component_index in range(min(3, len(pca.explained_variance_))):
        for sigma in [-2.0, 0.0, 2.0]:
            points = pca_mode_shape(pca, n_points, component_index, sigma)
            name = f"pc{component_index + 1}_sigma_{sigma:+.1f}.xyz"
            save_point_cloud(output_dir / name, points)

    reconstruction_components = [1, 3, 5, 10, 20]

    if k95 is not None:
        reconstruction_components.append(k95)
    reconstruction_components = sorted(set([
        k for k in reconstruction_components
        if k <= len(pca.explained_variance_)
]))
    reconstruction_components = [
        k for k in reconstruction_components if k <= len(pca.explained_variance_)
    ]

    for k in reconstruction_components:
        rec = reconstruct_shape(pca, scores, shape_index=0, n_points=n_points, n_components=k)
        save_point_cloud(output_dir / f"shape0_reconstruction_{k}_components.xyz", rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", help="Path to CPDataset .h5 file")
    parser.add_argument("--key", default="point_clouds", help="H5 dataset key")
    parser.add_argument("--components", type=int, default=100)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--output-dir", default="outputs/pca_point_clouds")
    args = parser.parse_args()

    point_clouds, mesh_names = load_point_clouds(args.h5_path, args.key)

    print("point_clouds shape:", point_clouds.shape)
    if mesh_names is not None:
        print("first mesh names:", mesh_names[:5])

    if args.no_normalize:
        pca_input = point_clouds
    else:
        pca_input = center_and_scale(point_clouds)

    pca, scores = fit_pca(pca_input, args.components)

    k95, cumulative = components_for_explained_variance(pca, threshold=0.95)
    print(f"Number of components for 95% explained variance: {k95}")

    print("PCA components:", len(pca.explained_variance_ratio_))
    print("explained variance ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {ratio:.4f}")

    print("cumulative explained variance:")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for i, ratio in enumerate(cumulative, start=1):
        print(f"  PC1-{i}: {ratio:.4f}")

    save_outputs(pca_input, pca, scores, args.output_dir, k95=k95)
    print("Saved outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
