import lmdb
import pandas as pd
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import cast
from multiprocessing import Pool, cpu_count

import torch_geometric
from torch_geometric.data.data import BaseData, Data

# Function to generate molecule ID
def generate_molecule_id(metadata):
    bulk_symbols = metadata.get("bulk_symbols", "Unknown")
    # ads_symbols = metadata.get("ads_symbols", "None")
    return bulk_symbols

# Function to transform PyG data
def pyg2_data_transform(data: BaseData):
    if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
        data = Data(**{k: v for k, v in data.__dict__.items() if v is not None})
        data = cast(BaseData, data)
    return data

# Function to process a single LMDB file
def process_lmdb(args):
    lmdb_file, output_file, position = args
    env = lmdb.open(str(lmdb_file), readonly=True, lock=False, max_readers=128, subdir=False)
    data = []

    with env.begin() as txn:
        # Check if "length" metadata exists, fallback to stat if not
        try:
            length = pickle.loads(txn.get("length".encode("ascii")))
        except TypeError:
            length = env.stat()["entries"]

        # Process each entry in the LMDB with its own progress bar
        for idx in tqdm(range(length), desc=f"Processing {lmdb_file.name}", unit="record", position=position, leave=False):
            key = f"{idx}".encode("ascii")
            value = txn.get(key)
            try:
                # Apply PyG transformation
                record = pyg2_data_transform(pickle.loads(value))
                sid = record.get("sid", "Unknown")
                fid = record.get("fid", "Unknown")
                metadata = dataset_metadata.get(sid, {})
                molecule_symbol = generate_molecule_id(metadata)
                data.append([molecule_symbol, fid, sid])
            except Exception as e:
                print(f"Error processing key {key.decode('utf-8')} in {lmdb_file.name}: {e}")

    # Save the processed data to a text file
    df = pd.DataFrame(data, columns=["Molecule", "fid", "sid"])
    df.to_csv(output_file, index=False, sep=",", header=True)
    print(f"Saved: {output_file}")

# Function to process all LMDB files
def process_all_lmdbs(lmdb_dir, output_dir, num_workers=10):
    lmdb_files = sorted(
        [f for f in Path(lmdb_dir).glob("*.lmdb") if not f.name.endswith("-lock")],
        key=lambda x: int(x.stem.split('.')[1])  # Sort by numeric part of filename
    )
    output_files = [Path(output_dir) / f.name.replace(".lmdb", ".txt").replace("data", "data_log") for f in lmdb_files]
    
    # Use multiprocessing Pool with tqdm for each worker
    with Pool(num_workers) as pool:
        pool.map(
            process_lmdb,
            [(lmdb_file, output_file, i) for i, (lmdb_file, output_file) in enumerate(zip(lmdb_files, output_files))]
        )


if __name__ == "__main__":
    # Path to the OC20/OC22 metadata file
    # TODO: Update this path
    metadata_file = "path/to/oc22_metadata.pkl"
    
    # Download the metadata file from the Open Catalyst Project website:
    # OC20: https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl
    # Found in: https://fair-chem.github.io/core/datasets/oc20.html#oc20-mappings

    # OC22: https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_metadata.pkl
    # Found in: https://fair-chem.github.io/core/datasets/oc22.html#oc22-mappings

    # Path to the OC20/OC22 lmdb training files
    # TODO: Update this path
    lmdb_dir = "path/to/lmdb/files"

    # Load metadata
    with open(metadata_file, "rb") as f:
        dataset_metadata = pickle.load(f)

    # Run the processing
    output_dir = lmdb_dir
    process_all_lmdbs(lmdb_dir, output_dir, num_workers=1)