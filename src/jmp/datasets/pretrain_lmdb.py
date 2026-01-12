"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import pickle
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any
from argparse import Namespace

import lmdb
import numpy as np
import torch
from jmp.lightning import TypedConfig
from torch.utils.data import Dataset
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ..utils.ocp import pyg2_data_transform
from .sampling_utils import (
    get_molecule_df, 
    apply_random_sampling,
    apply_difficulty_sampling,
    apply_class_balanced_sampling,
    apply_stratified_sampling,
    apply_naive_uniform_sampling,
)

class PretrainDatasetConfig(TypedConfig):
    src: Path
    """Path to the LMDB file or directory containing LMDB files."""

    metadata_path: Path | None = None
    """Path to the metadata npz file containing the number of atoms in each structure."""

    total_energy: bool | None = None
    """Whether to train on total energies."""
    oc20_ref: Path | None = None
    """Path to the OC20 reference energies file."""
    lin_ref: Path | None = None
    """Path to the linear reference energies file."""
    max_samples: int | None = None
    """Maximum number of samples"""
    is_train: bool | None = None
    """Whether config is for train or val."""
    args: Namespace = Namespace()
    """Additional arguments"""

    def __post_init__(self):
        super().__post_init__()

        # If metadata_path is not provided, assume it is src/metadata.npz
        if self.metadata_path is None:
            self.metadata_path = self.src / "metadata.npz"


class PretrainLmdbDataset(Dataset[BaseData]):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def data_sizes(self, indices: list[int]) -> np.ndarray:
        return self.atoms_metadata[indices]

    @property
    def atoms_metadata(self) -> np.ndarray:
        if (
            metadata := next(
                (
                    self.metadata[k]
                    for k in ["natoms", "num_nodes"]
                    if k in self.metadata
                ),
                None,
            )
        ) is None:
            raise ValueError(
                f"Could not find atoms metadata key in loaded metadata.\n"
                f"Available keys: {list(self.metadata.keys())}"
            )
        return metadata

    @property
    @cache
    def metadata(self) -> Mapping[str, np.ndarray]:
        metadata_path = getattr(self, "metadata_path", None)
        if not metadata_path or not metadata_path.is_file():
            metadata_path = self.config.metadata_path

        if metadata_path and metadata_path.is_file():
            return np.load(metadata_path, allow_pickle=True)

        raise ValueError(f"Could not find atoms metadata in {metadata_path=}.")

    def __init__(
        self,
        config: PretrainDatasetConfig,
        use_referenced_energies: bool = True,
        transform: Callable[[BaseData], Any] | None = None,
    ):
        super(PretrainLmdbDataset, self).__init__()
        self.config = config

        self.path = Path(self.config.src)
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, (
                f"No LMDB files found in '{self.path}'. "
                f"Please check if the root path '{self.config.args.root_path}' contains the correct dataset."
            )
            # self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                try:
                    length = pickle.loads(
                        self.envs[-1].begin().get("length".encode("ascii"))
                    )
                except TypeError:
                    length = self.envs[-1].stat()["entries"]
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)

        else:
            # self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)

        # if self.config.args.sampling_strategy not in ["random", "low_fid", "high_fid", "random_fid", "mixed_fid"]:
        #     # Assert that extract_features is set for non-random strategies
        #     assert hasattr(self.config.args, 'extract_features') and self.config.args.extract_features, (
        #         "For non-random and non-FID sampling strategies, 'extract_features' must be set in self.config.args."
        #     )


        # Extract molecule names
        self.molecule_df = None
        if self.config.is_train:
            # Load molecule data when extracting features
            if (hasattr(self.config.args, 'extract_features') and self.config.args.extract_features):
                self.molecule_df = get_molecule_df(self.path)

        max_samples = self.config.max_samples
        seed = self.config.args.seed
        if self.config.is_train and max_samples is not None:
            # "balanced", "balancedNoRep", "stratified" and "uniform" are only supported with self.config.args.extract_features
            if "difficulty" in self.config.args.sampling_strategy:
                self.molecule_df = get_molecule_df(self.path)
                self.shuffled_indices = apply_difficulty_sampling(self.molecule_df, max_samples, self.config.args.sampling_strategy, seed)
            elif self.config.args.sampling_strategy == "balanced":
                self.shuffled_indices = apply_class_balanced_sampling(self.molecule_df, max_samples, seed, allow_repetition=True)
            elif self.config.args.sampling_strategy == "balancedNoRep":
                self.shuffled_indices = apply_class_balanced_sampling(self.molecule_df, max_samples, seed, allow_repetition=False)
            elif self.config.args.sampling_strategy == "stratified":
                self.shuffled_indices = apply_stratified_sampling(self.molecule_df, max_samples, seed)
            elif self.config.args.sampling_strategy == "uniform":
                self.shuffled_indices = apply_naive_uniform_sampling(self.molecule_df, max_samples, seed)
            else:
                self.shuffled_indices = apply_random_sampling(self.num_samples, max_samples, seed)
        else:
            self.shuffled_indices = list(range(min(self.num_samples, max_samples)))


        self.transform = transform
        self.lin_ref = self.oc20_ref = None
        self.train_total = self.config.total_energy or False
        # only needed for oc20 datasets, p is total by default
        if self.train_total:
            oc20_ref = self.config.oc20_ref
            if not oc20_ref:
                raise ValueError("oc20_ref must be provided for oc20 datasets")
            self.oc20_ref = pickle.load(open(oc20_ref, "rb"))

        if (lin_ref := self.config.lin_ref) is not None and use_referenced_energies:
            coeff = np.load(lin_ref, allow_pickle=True)["coeff"]
            try:
                self.lin_ref = torch.nn.Parameter(
                    torch.tensor(coeff), requires_grad=False
                )
            except BaseException:
                self.lin_ref = torch.nn.Parameter(
                    torch.tensor(coeff[0]), requires_grad=False
                )

    def __len__(self):
        return len(self.shuffled_indices)

    @override
    def __getitem__(self, idx):
        # Use the selected subset of indices
        lmdb_idx = self.shuffled_indices[idx]

        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, lmdb_idx)
            # Extract index of element within that db.
            el_idx = lmdb_idx
            if db_idx != 0:
                el_idx = lmdb_idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            with self.envs[db_idx].begin(write=False) as txn:
                datapoint_pickled = txn.get(
                    f"{self._keys[db_idx][el_idx]}".encode("ascii")
                )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            with self.env.begin(write=False) as txn:
                datapoint_pickled = txn.get(self._keys[lmdb_idx])
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))


        # Rename data.energy to data.y exists (special case for OC20)
        if getattr(data_object, 'y', None) is None and hasattr(data_object, 'energy'):
            data_object.y = data_object.energy
            del data_object.energy

        # Rename data.forces to data.force exists (special case for OC20)
        if getattr(data_object, 'force', None) is None and hasattr(data_object, 'forces'):
            data_object.force = data_object.forces
            del data_object.forces


        if self.transform is not None:
            data_object = self.transform(data_object)
        # make types consistent
        sid = data_object.sid
        if isinstance(sid, torch.Tensor):
            sid = sid.item()
            data_object.sid = sid
        if "fid" in data_object:
            fid = data_object.fid
            if isinstance(fid, torch.Tensor):
                fid = fid.item()
                data_object.fid = fid
        if "bulk" in data_object:
            del data_object.bulk
        if hasattr(data_object, "y_relaxed"):
            attr = "y_relaxed"
        elif hasattr(data_object, "y"):
            attr = "y"
        # if targets are not available, test data is being used
        else:
            return data_object

        # Lookup molecule name based on sid and fid
        if self.molecule_df is not None:
            data_object.lmdb_idx = lmdb_idx
            row = self.molecule_df[(self.molecule_df['sid'] == sid) & (self.molecule_df['fid'] == fid)]
            if not row.empty:
                molecule_name = row.iloc[0]['Molecule']
                # print("lmdb_idx:", lmdb_idx, "idx:", idx, "molecule_name:", molecule_name, "sid:", sid, "fid:", fid)
                data_object.molecule_name = molecule_name  # Add molecule name to the data object
            else:
                print("sid =", sid, " and fid =", fid, "was not found")

        # convert s2ef energies to raw energies
        if attr == "y":
            # OC20 data
            if "p" not in data_object and self.train_total:
                assert self.oc20_ref is not None

                randomid = f"random{sid}"
                if hasattr(data_object, "task_mask"):
                    data_object[attr][data_object.task_mask] += self.oc20_ref[randomid]
                else:
                    data_object[attr] += self.oc20_ref[randomid]
                data_object.nads = 1
                data_object.p = 0

        # convert is2re energies to raw energies
        else:
            if "p" not in data_object and self.train_total:
                assert self.oc20_ref is not None

                randomid = f"random{sid}"
                data_object[attr] += self.oc20_ref[randomid]
                del data_object.force
                del data_object.y_init
                data_object.nads = 1
                data_object.p = 0

        if self.lin_ref is not None:
            lin_energy = sum(self.lin_ref[data_object.atomic_numbers.long()])
            if hasattr(data_object, "task_mask"):
                data_object[attr][data_object.task_mask] -= lin_energy
            else:
                data_object[attr] -= lin_energy
        if "nads" in data_object:
            del data_object.nads
        if "p" in data_object:
            del data_object.p
        # to jointly train on p+oc20, need to delete these oc20-only attributes
        # ensure otf_graph=1 in your model configuration
        # if "edge_index" in data_object:
        #    del data_object.edge_index
        # if "cell_offsets" in data_object:
        #    del data_object.cell_offsets
        # if "distances" in data_object:
        #    del data_object.distances

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()
