import csv
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import Pool, get_context
from tqdm import tqdm
import numpy as np
from scipy import linalg
import torch
from joblib import Parallel, delayed



def compute_statistics_of_features(path, agg_operation):
    data = torch.load(path)
    if isinstance(data, list):      
        data = [torch.tensor(item['node_features']) for item in data]
        # Concatenate all node features across graphs into a single tensor
        if agg_operation == "flat":
            agg_features = torch.cat(data, dim=0)  # Concatenate along the first dimension
        elif agg_operation == "mean":
            mean_features = [d.mean(dim=0).unsqueeze(0) for d in data]
            agg_features = torch.cat(mean_features)
        elif agg_operation == "sum":
            sum_features = [d.sum(dim=0).unsqueeze(0) for d in data]
            agg_features = torch.cat(sum_features)
        elif agg_operation == "max":
            max_features = [d.max(dim=0)[0].unsqueeze(0) for d in data]
            agg_features = torch.cat(max_features)

        features = agg_features.detach().cpu().numpy()
        print(features.shape)
    elif torch.is_tensor(data):
        features = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        features = data
    elif isinstance(data, dict):
        # Assume features are stored under the key 'features'
        if 'features' in data:
            features = data['features']
            if torch.is_tensor(features):
                features = features.numpy()
        else:
            raise ValueError(f"Cannot find 'features' key in the loaded data from {path}")
    else:
        raise ValueError(f"Unsupported data type in {path}")

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    # print("Mean:", np.mean(mu))
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "FID calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical errors might give slight imaginary components
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}, consider downgrading scipy by running 'pip install scipy==1.11.1'")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_csi(m1, s1, m2, s2):
    csi_score = calculate_frechet_distance(m1, s1, m2, s2)
    return csi_score


def extract_metadata(file_path):
    """
    Extract the sampling type and seed number from the file path.
    
    Args:
        file_path (str): Path to the input file.
    
    Returns:
        tuple: Extracted sampling type and seed number.
    """
    base_name = os.path.basename(file_path)
    sampling_keyword = "Sampling"
    seed_keyword = "Seed"
    sampling_type = None
    seed_number = None

    if sampling_keyword in base_name and seed_keyword in base_name:
        parts = base_name.split("_")
        for part in parts:
            if part.startswith(sampling_keyword):
                sampling_type = part[len(sampling_keyword):]
            elif part.startswith(seed_keyword):
                seed_number = part[len(seed_keyword):]
    sampling_type = sampling_type.replace(".pt", "")
    return sampling_type or "unknown", seed_number or "unknown"

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=2, help="Paths to upstream and downstream datasets")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of processes to use")
    parser.add_argument("--model_name", type=str, default="gemnet", help="Model name")
    parser.add_argument("--agg_operation", type=str, default="flat", help="Aggregation operation")
    parser.add_argument("--checkpoint_tag", type=str, default="OC20", help="checkpoint tag")
    args = parser.parse_args()

    # Load upstream dataset features
    upstream_features_path = f"dataset_features_{args.model_name}_{args.checkpoint_tag}/{args.path[0]}"
    upstream_mu, upstream_sigma = compute_statistics_of_features(upstream_features_path, args.agg_operation)

    # Load downstream dataset features
    downstream_features_path = f"dataset_features_{args.model_name}_{args.checkpoint_tag}/{args.path[1]}"
    downstream_mu, downstream_sigma = compute_statistics_of_features(downstream_features_path, args.agg_operation)

    # Extract metadata
    upstream = upstream_features_path.split("/")[-1].split("_")[0]
    downstream = downstream_features_path.split("/")[-1].split("_")[0]
    if downstream == "matbench":
        downstream = "matbench_" + downstream_features_path.split("/")[-1].split("_")[1]  
    sampling, seed = extract_metadata(upstream_features_path)
    
    print(f"Upstream: {upstream}, Downstream: {downstream}, Sampling: {sampling}, Seed: {seed}")


    csi_score = compute_csi(upstream_mu, upstream_sigma, downstream_mu, downstream_sigma)
    print(f"CSI Score: {csi_score:.2f}")
    print()
    print("=============================================================")

    # Save the CSI score
    save_filename = f"csi_scores_{args.agg_operation}_{args.model_name}_{args.checkpoint_tag}.csv"
    file_exists = os.path.isfile(save_filename)
    with open(save_filename, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Upstream", "Downstream", "Sampling", "Seed", "CSI Score"])
        writer.writerow([upstream, downstream, sampling, seed, csi_score])


if __name__ == "__main__":
    main()