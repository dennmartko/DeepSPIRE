from itertools import combinations
import os
import gc
import argparse
import pandas as pd
from tqdm import tqdm
from astropy.table import Table, vstack
from astropy.io import fits
import numpy as np

# These columns are required to generate the datamaps and for evaluation
# pySIDES expects flux column names following the convention 'S' + FILTER NAME
FLUX_COLS = ['SMIPS24', 'SSPIRE250', 'SSPIRE350', 'SSPIRE500']

EXTRA_COLS = ['ra', 'dec', 'redshift']
max_i, max_j = 6, 8  # max_j is used only for j=8 section in SIDES tiles

def load_catalog(tile_i, tile_j):
    filename = f'/mnt/g/data/PhD Projects/SR/pysides_from_uchuu_catalogs/pySIDES_from_uchuu_tile_{tile_i}_{tile_j}.fits'
    return Table.read(filename)

def process_tiles(output_dir, N=30, sub_area_sqdeg=2):
    step = 2 if sub_area_sqdeg == 2 else 1
    """Process Uchuu SIDES dataset catalogs from tile pairs and save at most N catalogs.
    Note, the complexity comes from the fact that tiles need to be adjacent!"""
    os.makedirs(output_dir, exist_ok=True)
    sides_count = 0

    if step == 1:
        points = [(i, j) for i in range(max_i + 1) for j in range(max_j + 1)]
        # Singletons + all 2-combinations
        all_pairs = [[p] for p in points] + [list(c) for c in combinations(points, 2)]

    elif step == 2:
        all_pairs = (
            [[(i, j), (i, j + 1)] for i in range(max_i + 1) for j in range(0, max_j, step)]
            + [[(i, max_j), (i + 1, max_j)] for i in range(0, max_i, 2)]
        )
        print(f"Total SIDES tile pairs to process: {all_pairs}, len={len(all_pairs)}")
    elif step > 2:
        raise ValueError("sub_area_sqdeg must be either 1 or 2.")
    
    for pair in tqdm(all_pairs[:N], desc='Processing SIDES tile pairs'):
        if sides_count >= N:
            break
        tables = []
        for (i, j) in pair:
            try:
                cat = load_catalog(i, j)
                tables.append(cat[FLUX_COLS + EXTRA_COLS])
            except Exception as e:
                print(f"Skipping tiles {pair}: {e}")
                break;
        if len(tables) == len(pair):
            merged = vstack(tables)
            sides_count += 1
            fname = os.path.join(output_dir, f'SIDES_{sides_count}_cat.fits')
            merged.write(fname, overwrite=True)
            del merged, tables
            gc.collect()

def load_shark_catalog():
    """Load and merge the SHARK catalogs from chunks."""
    file_path_fluxes = '/mnt/g/data/PhD Projects/SR/Shark-deep-opticalLightcone-AtLAST-FIR.txt'
    file_path_coords = '/mnt/g/data/PhD Projects/SR/Shark-deep-opticalLightcone-AtLAST.txt'
    flux_iter = pd.read_csv(
        file_path_fluxes,
        sep=r'\s+|\t',
        header=None,
        engine='python',
        skiprows=8,
        names=FLUX_COLS,
        usecols= [0, 4, 5, 7], # Must be manually set!!! CAREFUL: [1, 0] = [0, 1] hence it needs to be sorted!
        chunksize=int(6e6),
        dtype={f"{col}" : np.float64 for col in FLUX_COLS},
    )
    pos_iter = pd.read_csv(
        file_path_coords,
        sep=r'\s+|\t',
        header=None,
        engine='python',
        skiprows=12,
        names=["dec", "ra", "redshift"],
        usecols=[0, 1, 2],
        chunksize=int(6e6),
        dtype={"dec": np.float64, "ra": np.float64, "redshift": np.float64}
    )

    merged_chunks = []
    chunk_counter = 0
    for pos_chunk, flux_chunk in zip(pos_iter, flux_iter):
        chunk_counter += 1
        print(f"Processing SHARK chunk {chunk_counter}")
        flux_chunk = flux_chunk / 1000  # Convert mJy to Jy
        pos_chunk = pos_chunk.reset_index(drop=True)
        flux_chunk = flux_chunk.reset_index(drop=True)
        merged_chunks.append(pd.concat([pos_chunk, flux_chunk], axis=1))

    print(f"Loaded {chunk_counter} chunks. Concatenating SHARK catalog...")
    merged_cat = pd.concat(merged_chunks, ignore_index=True)
    print("Finished loading SHARK catalog.")
    return merged_cat

def process_shark_catalog(output_dir, N, sub_area_sqdeg=2):
    """Cut the SHARK catalog into subregions and write at most N catalogs."""
    print("Starting SHARK catalog processing...")
    shark_cat = load_shark_catalog()

    # Create output_dir if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Define region bounds and cut size (delta)
    ra_start, ra_end = 211.5, 223.5
    dec_start, dec_end = -4.5, 4.5
    delta_ra, delta_dec = (2, 1) if sub_area_sqdeg == 2 else (1, 1)  # Adjust delta_ra for 2 sq.deg. subregions

    shark_count = 0
    for dec_min in np.arange(dec_start, dec_end, delta_dec):
        dec_max = dec_min + delta_dec
        for ra_min in np.arange(ra_start, ra_end, delta_ra):
            if shark_count >= N:
                break
            ra_max = ra_min + delta_ra
            print(f"Processing SHARK subregion {shark_count+1}: RA {ra_min}-{ra_max}, Dec {dec_min}-{dec_max}")
            subset = shark_cat[(shark_cat['ra'] >= ra_min) & (shark_cat['ra'] < ra_max) &
                               (shark_cat['dec'] >= dec_min) & (shark_cat['dec'] < dec_max)]
            if not subset.empty:
                table = Table.from_pandas(subset)
                shark_count += 1
                fname = os.path.join(output_dir, f'SHARK_{shark_count}_cat.fits')
                table.write(fname, overwrite=True)
                print(f"Saved SHARK subregion {shark_count} to {fname}")
        if shark_count >= N:
            break

    print("Finished processing SHARK catalog.")

if __name__ == '__main__':
    output_dir = '/mnt/g/data/PhD Projects/SR/sim_catalogs' # Set your desired output directory here
    # To process SIDES catalogs, uncomment the next line:
    process_tiles(output_dir, N=30, sub_area_sqdeg=2)  # Process SIDES catalogs
    gc.collect()
    process_shark_catalog(output_dir, N=30, sub_area_sqdeg=2) # Process SHARK catalogs