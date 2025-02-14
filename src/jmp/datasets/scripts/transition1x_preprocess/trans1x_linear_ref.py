"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pickle
from functools import cache
from pathlib import Path

import multiprocess as mp
import numpy as np
import torch
from jmp.datasets.pretrain_lmdb import PretrainDatasetConfig, PretrainLmdbDataset
from torch_scatter import scatter
from tqdm import tqdm


def _compute_mean_std(args: argparse.Namespace):
    @cache
    def dataset():
        return PretrainLmdbDataset(
            PretrainDatasetConfig(src=args.src, lin_ref=args.linref_path)
        )

    def extract_data(idx):
        data = dataset()[idx]
        y = data.y
        f = data.force.cpu().numpy()
        return (y, f)

    pool = mp.Pool(args.num_workers)
    indices = range(len(dataset()))

    outputs = list(tqdm(pool.imap(extract_data, indices), total=len(indices)))

    energies = [y for y, forces in outputs]
    forces = np.array([force for y, forces in outputs for force in forces])

    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    force_rms = np.sqrt(np.mean(np.square(forces)))
    force_md = np.mean(np.linalg.norm(forces, axis=-1))

    print(
        f"energy_mean: {energy_mean}, energy_std: {energy_std}, force_rms: {force_rms}, force_md: {force_md}"
    )

    with open(args.out_path, "wb") as f:
        pickle.dump(
            {
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "force_rms": force_rms,
                "force_md": force_md,
            },
            f,
        )


def _linref(args: argparse.Namespace):
    @cache
    def dataset():
        return PretrainLmdbDataset(PretrainDatasetConfig(src=args.src))

    def extract_data(idx):
        data = dataset()[idx]
        x = (
            scatter(
                torch.ones(data.atomic_numbers.shape[0]),
                data.atomic_numbers.long(),
                dim_size=10,
            )
            .long()
            .numpy()
        )
        y = data.y
        f = data.force.cpu().numpy()
        return (x, y, f)

    pool = mp.Pool(args.num_workers)
    indices = range(len(dataset()))

    outputs = list(tqdm(pool.imap(extract_data, indices), total=len(indices)))

    features = [x[0] for x in outputs]
    targets = [x[1] for x in outputs]

    X = np.vstack(features)
    y = targets

    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    np.savez_compressed(args.out_path, coeff=coeff)
    print(f"Saved linear reference coefficients to {args.out_path}")


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcommand")

    compute_mean_std_parser = subparsers.add_parser("compute_mean_std")
    compute_mean_std_parser.add_argument("--src", type=Path)
    compute_mean_std_parser.add_argument("--out_path", type=Path)
    compute_mean_std_parser.add_argument("--linref_path", type=Path)
    compute_mean_std_parser.add_argument("--num_workers", type=int, default=32)
    compute_mean_std_parser.set_defaults(fn=_compute_mean_std)

    linref_parser = subparsers.add_parser("linref")
    linref_parser.add_argument("--src", type=Path)
    linref_parser.add_argument("--out_path", type=Path)
    linref_parser.add_argument("--num_workers", type=int, default=32)
    linref_parser.set_defaults(fn=_linref)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
