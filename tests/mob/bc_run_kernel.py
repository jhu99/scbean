import warnings
warnings.filterwarnings("ignore")

import argparse
import anndata as ad
import pandas as pd
import multiprocessing as mp
from scbean.model import visgp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        choices=["rbf", "matern", "periodic", "anisotropic", "multi"]
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None
    )
    args = parser.parse_args()

    kernel = args.kernel

    filepath = '/home/wangz/VISGP/somematerials/Rep9_MOB_count_matrix-1.tsv'
    data = pd.read_csv(filepath, sep='\t')

    # =========================================================
    # 1. Extract spatial coordinates
    # =========================================================
    location = pd.DataFrame(index=data.index)
    location['x'] = data['Unnamed: 0'].str.split('x').str.get(0).map(float)
    location['y'] = data['Unnamed: 0'].str.split('x').str.get(1).map(float)

    data.drop('Unnamed: 0', axis=1, inplace=True)

    # =========================================================
    # 2. Filter practically unobserved genes
    # data: spots x genes
    # =========================================================
    data = data.T[data.sum(0) >= 10].T

    # =========================================================
    # 3. Convert to genes x spots
    # =========================================================
    data = data.T

    # =========================================================
    # 4. Filter practically empty spots
    # data: genes x spots
    # =========================================================
    spot_keep = data.sum(0) >= 10
    location = location.loc[spot_keep, :]
    data = data.loc[:, spot_keep]

    # =========================================================
    # 5. Standardize spatial coordinates
    # =========================================================
    location[['x', 'y']] = (
        location[['x', 'y']] - location[['x', 'y']].mean()
    ) / location[['x', 'y']].std()

    # =========================================================
    # 6. Build AnnData
    # =========================================================
    obs = pd.DataFrame(index=data.index)
    obs['gene_name'] = data.index.values

    adata = ad.AnnData(
        data.values,
        obs=obs,
        var=location,
        dtype='float64'
    )

    print("========================================")
    print(f"Running VISGP with kernel: {kernel}")
    print("AnnData:", adata)
    print("Genes:", adata.n_obs)
    print("Spots:", adata.n_vars)
    print("========================================")

    if args.processes is None:
        n_processes = max(1, mp.cpu_count() - 2)
    else:
        n_processes = args.processes

    print("Using processes:", n_processes)

    obj = visgp.VISGP(
        adata,
        processes=n_processes,
        kernel_type=kernel,
        nu=1.5
    )

    results = obj.run()

    out_file = f"mob9_results_{kernel}.csv"
    results.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")
    print(
        "Number of SVGs with q_value < 0.05:",
        (results['q_value'].astype(float) < 0.05).sum()
    )
    print("Finished.")


if __name__ == "__main__":
    main()