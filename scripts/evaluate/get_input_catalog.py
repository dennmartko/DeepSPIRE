import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

progress = Progress(
    TextColumn("[bold blue]Status:[/bold blue] [medium_purple]{task.description}"),
    BarColumn(
        bar_width=60,
        complete_style="bold green",
        finished_style="green",
        pulse_style="bright_blue"
    ),
    TextColumn("[bold cyan]{task.percentage:>3.0f}% Complete[bold cyan]"),
    TimeElapsedColumn(),
    refresh_per_second=10
)

def load_cutout_wcs(path):
    """Return dict filename → WCS object."""
    fns = [f for f in os.listdir(path) if f.lower().endswith('.fits')]
    wcs_dict = {}

    with progress:
        task = progress.add_task("Loading cutout WCS objects...", total=len(fns))
        for fn in fns:
            hdr = fits.open(os.path.join(path, fn))[0].header
            match = re.search(r'_(\d+)\.fits$', fn)
            if match:
                file_id = int(match.group(1))
            wcs_dict[file_id] = WCS(hdr)
            progress.update(task, advance=1)
    print(f"Loaded WCS for {len(wcs_dict)} cutouts.")
    return wcs_dict

def normalize_ra(ra):
    """Normalize RA to [0, 360) degrees."""
    return np.mod(ra, 360)

def get_catalog_bounds(ra, dec):
    """Return RA/Dec bounds for a catalog, correctly handling RA wraparound."""
    ra = normalize_ra(np.array(ra))
    dec = np.array(dec)

    # Sort RA and find minimal interval that includes all points
    ra_sorted = np.sort(ra)
    ra_diff = np.diff(np.concatenate([ra_sorted, [ra_sorted[0] + 360]]))
    max_gap_index = np.argmax(ra_diff)

    # Exclude the largest gap, get the smallest bounding RA interval
    ra_min = ra_sorted[(max_gap_index + 1) % len(ra_sorted)]
    ra_max = ra_sorted[max_gap_index]

    ra_bounds = (ra_min, ra_max + 360) if ra_min > ra_max else (ra_min, ra_max)
    dec_bounds = (np.min(dec), np.max(dec))

    return ra_bounds, dec_bounds


def get_image_bounds_from_wcs(wcs, shape):
    """Given a WCS and image shape (ny, nx), return RA/Dec bounds."""
    ny, nx = shape

    # Define corner pixel coordinates
    corners_pix = np.array([
        [0, 0],           # bottom-left
        [0, nx - 1],      # bottom-right
        [ny - 1, 0],      # top-left
        [ny - 1, nx - 1], # top-right
    ])

    # Convert pixel to sky coordinates
    sky_coords = wcs.all_pix2world(corners_pix[:, 1], corners_pix[:, 0], 0)
    # print(sky_coords)
    ra = normalize_ra(sky_coords[0])
    dec = sky_coords[1]

    return get_catalog_bounds(ra, dec)


def ra_interval_overlap(ra1, ra2):
    """Check for overlap between two RA intervals, accounting for wraparound."""
    # Normalize to [0, 360)
    ra1_min, ra1_max = np.mod(ra1[0], 360), np.mod(ra1[1], 360)
    ra2_min, ra2_max = np.mod(ra2[0], 360), np.mod(ra2[1], 360)

    def expand_interval(rmin, rmax):
        if rmin <= rmax:
            return [(rmin, rmax)]
        else:
            # Wraps around 0: e.g., 350° to 10° becomes two segments
            return [(rmin, 360), (0, rmax)]

    seg1 = expand_interval(ra1_min, ra1_max)
    seg2 = expand_interval(ra2_min, ra2_max)

    # Check all combinations of segments for overlap
    for s1_min, s1_max in seg1:
        for s2_min, s2_max in seg2:
            if s1_max >= s2_min and s2_max >= s1_min:
                return True
    return False

def dec_interval_overlap(dec1, dec2):
    """Simple 1D overlap for Dec."""
    return not (dec1[1] < dec2[0] or dec2[1] < dec1[0])

def bounds_overlap(cat_bounds, img_bounds):
    """Check for RA/Dec overlap with proper RA wraparound."""
    ra_cat, dec_cat = cat_bounds
    ra_img, dec_img = img_bounds
    return ra_interval_overlap(ra_cat, ra_img) and dec_interval_overlap(dec_cat, dec_img)

def cutout_overlaps_catalog(wcs, shape, cat_bounds):
    """Check if an image defined by WCS and shape overlaps a catalog."""
    img_bounds = get_image_bounds_from_wcs(wcs, shape)
    return bounds_overlap(img_bounds, cat_bounds)

def process_sim_file(fp, wcs_dict):
    """Return list of DataFrames for all cutouts overlapping this sim file."""
    sim_cat = Table.read(fp)

    # Calculate the catalog boundary once. Speeds up ~x100.
    cat_bounds = get_catalog_bounds(sim_cat['ra'], sim_cat['dec'])

    results = []
    # Iterate over the cutouts and find sources from the catalog belonging whose coordinates lie within these cutouts
    for file_id, w_cut in wcs_dict.items():
        # First, check whether the cutout overlaps with the catalog
        is_overlap = cutout_overlaps_catalog(w_cut, CUTOUT_SHAPE, cat_bounds)

        if is_overlap:
            # Convert all catalog coordinates to x, y coordinates using the cutout WCS
            x, y = w_cut.all_world2pix(sim_cat['ra'], sim_cat['dec'], 0, ra_dec_order=True)
            # Mask the sources that within the cutout region
            mask = (0 <= x) & (x < CUTOUT_SHAPE[1]) & (0 <= y) & (y < CUTOUT_SHAPE[0])
            if not mask.any():
                continue

            df_sub = sim_cat[mask].to_pandas()

            # The input catalog is directly written to disk, but will be used in other scripts
            # It is therefore desired to keep the potential memory usage low
            # It is mostly the SIDES simulation that is responsible for huge data sizes due to the inclusion of incredibly faint sources - low mass galaxies.
            # 1. We only keep the specified flux columns (Only the target flux bands are really necessary)
            # 2. We only remove sources that are below all the specified flux thresholds in their corresponding bands
            # To Do: Convert to float32.
            
            df_sub = df_sub[['ra', 'dec'] + flux_columns_to_use + extra_columns_to_use]

            mask = np.zeros(len(df_sub), dtype=bool)
            for col, thresh in zip(flux_columns_to_use, flux_threshold):
                mask |= (df_sub[col] >= thresh)
            df_sub = df_sub[mask]
            df_sub['file_id'] = file_id
            results.append(df_sub)
    return results

if __name__ == '__main__':
    # --- Config        ---
    # --- GLOBAL params ---
    # Paths containing data
    sim_catalogs_dirs    = ["/mnt/g/data/PhD Projects/SR/sim_catalogs"]                                    
    test_cutouts_path    = "/mnt/g/data/PhD Projects/SR/shark_sides_mips_spire_smoothed_120sqdeg/Test/500SR"
    catalog_output_dir   = "/mnt/g/data/PhD Projects/SR/evaluation"
    catalog_file_name    = "spire_smoothed_test_input_catalog.csv"
    flux_columns_to_use  = ['SMIPS24', 'SSPIRE500']                                                        # Flux columns to include in the resulting input catalog; Will include 'file_id', 'ra', 'dec' by default
    extra_columns_to_use = ['redshift']                                                                    # Additional columns to include in the resulting input catalog
    flux_threshold       = [.1e-6, .01e-3]                                                                 # Minimum flux to include source in input catalog corresponding to flux_columns_to_use; in Jy; Use 0 if unsure, keep a tab at file size.
    CUTOUT_SHAPE         = (256, 256)                                                                      # (ny, nx)
    max_workers          = 12                                                                              # Number of processes to use. Each process requires at least the equivalent of individual catalogs in memory.
      # Load all cutout WCS
    wcs_dict = load_cutout_wcs(test_cutouts_path)

    # Process each sim catalog in parallel
    sim_cat_files = [os.path.join(d, f) for d in sim_catalogs_dirs for f in os.listdir(d) if f.lower().endswith('.fits')]
    cutouts_ids = []

    with progress as prog, ProcessPoolExecutor(max_workers=max_workers) as executor:
        task = progress.add_task(f"Processing simulated catalogs...", total=len(sim_cat_files))
        futures = {executor.submit(process_sim_file, fn, wcs_dict): fn for fn in sim_cat_files}
        out_csv_path = os.path.join(catalog_output_dir, catalog_file_name)

        # Check if catalog already exists; if so, remove it to start fresh
        print(f"Saving input test catalog to: {out_csv_path}")
        if os.path.exists(out_csv_path):
            os.remove(out_csv_path)

        for future in as_completed(futures):
            dfs_list = future.result()
            prog.update(task, advance=1)

            if not dfs_list:
                continue

            # Merge DataFrames from this sim file
            df = pd.concat(dfs_list, ignore_index=True)

            if df.empty:
                continue

            # Store which cutouts have been extracted from the input test catalog
            cutouts_ids.extend(df['file_id'].unique())

            # Append to CSV incrementally
            df.to_csv(out_csv_path, mode='a', header=not os.path.exists(out_csv_path), index=False)

    # assemble the input catalog for the test set
    n_unique   = len(set(cutouts_ids))
    print(f"Recovered {n_unique} / {len(wcs_dict)} cutouts")
    print(f"Input test catalog saved to: {out_csv_path}")