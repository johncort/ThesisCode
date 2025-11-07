# Imports
import re
import itertools
from pathlib import Path
import concurrent.futures as cf

import numpy as np
import xarray as xr
import polars as pl

def process_glacier(args):

    """
    Helper function for GCM_run that processes a single glacier fl_diagnostics NetCDF file for hypsometry plotting
    Note: These two functions are by far the most complex I had to create. To overcome kernel crashes and to significantly speed up processing I moved out of a notebook to work directly 
    in the terminal and use multiprocessing. I therefore had to split up the function and to pass the arguments through to this function as a tuple, which did the job but I fear is not optimal. 
    I have tried to provide comments in greater depth to show the rationale behind their design. 

    Arguments:
        args (tuple): A tuple containing:
            filepath (Path): Path to fl_diagnostics NetCDF file
            pattern (re.Pattern): Pattern for RGI ID extraction from filename
            processed_glaciers (set): Set containing processed glacier IDs as strings (I pass this to enable skipping of processed glaciers)
            elev_bins (np.ndarray): Array of elevation bin edges
            bin_centres (np.ndarray): Array of elevation bin centres
            time_initial (int): Index of start year
            time_final (int): Index of final year
            t_change (int): Difference between initial and final time step
            GCM (str): Name of GCM
            ssp (str): Name of SSP

    Returns: 
        bin_summary (list): List of dictionaries, each dictionary contains data for an elevation bin

    Raises:
        None

    """

    # Extracts the argument tuple for processing
    (filepath, pattern, processed_glaciers, elev_bins, bin_centres, time_initial, time_final,
     t_change, GCM, ssp) = args

    # Locates glacier ID via filepath and regex pattern and skips glaciers that don't match the expected format or have already been processed 
    filename = filepath.name
    match = pattern.match(filename)
    if not match:
        return []
    glacier_id = match.group(1)
    if glacier_id in processed_glaciers:
        print(f'Already processed glacier {glacier_id} for {ssp}, skipping')
        return []

    # Opens glacier data and selects the primary flowline (note: OGGM sets this to be the last flowline)
    try:
        with xr.open_dataset(filepath) as fl_ds:
            fl_ids = fl_ds.flowlines.data
        main_fl_id = fl_ids[-1]

        # Loads bed elevation, glacier thickness, and glacier area along the primary flowline 
        # This code block is adapted from OGGM's 'Plotting the OGGM surface mass-balance, the ELA and AAR' tutorial
        # available at: https://tutorials.oggm.org/stable/notebooks/tutorials/plot_mass_balance.html
        with xr.open_dataset(filepath, group=f'fl_{main_fl_id}') as ds:
            ds.load()
            bed_elev = ds.bed_h.values
            thickness = ds.thickness_m.values
            area = ds.area_m2.values

            # Calculates initial and final surface elevations and areas (OGGM provides bed elevation and ice thickness so it is just a simple addition at each time slice)
            surface_initial = bed_elev + thickness[time_initial, :]
            surface_final = bed_elev + thickness[time_final, :]
            area_initial = area[time_initial, :]
            area_final = area[time_final, :]

            # Assigns data to elevation bins derived from surface elevation
            bins_initial = np.digitize(surface_initial, elev_bins) - 1
            bins_final = np.digitize(surface_final, elev_bins) - 1

            # Creates arrays for data storage
            area_by_elev_initial = np.zeros(len(bin_centres))
            area_by_elev_final = np.zeros(len(bin_centres))
            thickness_change_initial = np.full(len(bin_centres), np.nan)
            thickness_change_final = np.full(len(bin_centres), np.nan)

            # Loops through elevation bins and stores any data that falls within each bin for the initial and final time steps
            for i in range(len(bin_centres)):
                in_bin_initial = bins_initial == i
                if np.any(in_bin_initial):
                    area_by_elev_initial[i] = area_initial[in_bin_initial].sum()
                    thickness_change_initial[i] = thickness[time_initial, in_bin_initial].mean()
            for i in range(len(bin_centres)):
                in_bin_final = bins_final == i
                if np.any(in_bin_final):
                    area_by_elev_final[i] = area_final[in_bin_final].sum()
                    thickness_change_final[i] = thickness[time_final, in_bin_final].mean()

            # Calculates annual thickness change
            thickness_change_per_year = (thickness_change_final - thickness_change_initial) / t_change

            # Appends data to a dictionary per elevation bin, ensuring any cases of no data are dealt with safely for later when plotting, and stores all in list
            # I then return this list to the GCM_run function to append to a parquet file
            bin_summary = []
            for i, centre in enumerate(bin_centres):
                bin_summary.append({
                    'glacier_id': glacier_id,
                    'GCM': GCM,
                    'ssp': ssp,
                    'bin_centre': float(centre),
                    'area_initial': float(area_by_elev_initial[i]),
                    'area_final': float(area_by_elev_final[i]),
                    'thickness_change_per_year': (
                        float(thickness_change_per_year[i]) if not np.isnan(thickness_change_per_year[i]) else None
                    ),
                })

        print(f'Processed glacier {glacier_id} for {ssp}')
        return bin_summary

    # Returns an empty list in cases where the processing fails and prints error message but allows looping to continue
    except Exception as e:
        print(f'Error processing {filename}: {e}')
        return []

def GCM_run(GCM):

    """
    Processes all fl_diagnostics files for a specified GCM across SSPs

    Arguments:
        GCM (str): GCM name (note: this needs to match that within the filepath exactly as I pass it when locating files)

    Returns:
        None

    Raises:
        None

    """

    # Defines parameters for use in process_glaciers function 
    ssp_list = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    bin_size = 100
    elev_min = 0
    elev_max = 8000
    t_change = 75
    time_initial = 5
    time_final = -1

    # Creates fixed elevation bins for use across all glaciers (note that elev_max is set far above expected max elevation)
    elev_bins = np.arange(elev_min, elev_max + bin_size, bin_size)
    bin_centres = elev_bins[:-1] + bin_size / 2

    # Defines input and output filepaths
    data_dir = Path(f'custom_filepath/{GCM}') 
    output_dir = data_dir.parent

    print(f'Starting processing for GCM: {GCM}')

    for ssp in ssp_list:

        # Defines patterns for file matching for each GCM and SSP (Note: decided to automatically build filenames as regex from available RGI IDs rather than pass these as a list 
        # to avoid needing to account for missing files; took a while to realise that I needed to use a raw f string for this)
        pattern = re.compile(
            rf'fl_diagnostics_(RGI60-\d+\.\d+)_CMIP6_{GCM}_{ssp}\.nc'
        )
        args_list = [
            (filepath, pattern, processed_glaciers, elev_bins, bin_centres,
            time_initial, time_final, t_change, GCM, ssp)
            for filepath in list(data_dir.glob(f'fl_diagnostics_*_{ssp}.nc'))
        ]

        # Edits the GCM naming structure then forms path to parquet file for export (note: had to rename the GCM due to the fact I was inconsistent with my use of - and _ in filenames in other code)
        GCM_export = GCM.replace("-", "_")
        parquet_path = output_dir / 'Processed' / f'{GCM_export}_export_{ssp}.parquet'

        # Checks for existing output files to be skipped, ultimately adding to a set that gets passed to helper function and enables quick skipping
        if parquet_path.exists():
            existing_df = pl.read_parquet(parquet_path)
            print(f'Columns in existing_df: {existing_df.columns}')
            processed_glaciers = set(existing_df['glacier_id'].unique().to_list())
        else:
            existing_df = None
            processed_glaciers = set()

        # Formulates args list for input into process_glaciers function (list contains a tuple of params for each filepath)
        # Note: I spent ages trying to figure out how to pass multiple arguments through through concurrent futures, trying the approaches suggested in
        # here: https://stackoverflow.com/questions/6785226/pass-multiple-parameters-to-concurrent-futures-executor-map , however I couldn't get it to work. 
        # I ultimately decided to try to pass a list containing a tuple of parameters for each file and it worked, looping through each tuple in the list - however I'm still unsure how ideal this method is. 
        args_list = [
            (filepath, pattern, processed_glaciers, elev_bins, bin_centres,
            time_initial, time_final, t_change, GCM, ssp)
            for filepath in list(data_dir.glob(f'fl_diagnostics_*_{ssp}.nc'))
        ]

        # Runs process_glacier in parallel (Note: max_workers should be adjusted based on computer specs)
        # The use of concurrent.futures and the division of my original function into this and the helper function was prompted by Copilot (GPT-4.1) (https://code.visualstudio.com/docs/copilot/overview) after prompting where to integrate multiprocessing
        # in the previous iteration of this function to help speed up the issues I was having with processing time and kernel crashes (when previously running in a notebook)
        with cf.ProcessPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_glacier, args_list))

        # Flattens results, converts to a polars DF, then saves to a parquet file (Note: I also integrated a counter to track its progress)
        # Note: Parquet files were used due to storage constraints so this could easily be changed to another file format
        flat_results = [i for sublist in results for i in sublist if i]
        processed_glaciers = sum(1 for sublist in results if sublist)
        print(f'Finished processing {processed_glaciers} glaciers for {ssp}')
        new_df = pl.DataFrame(flat_results)
        combined_df = pl.concat([existing_df, new_df]) if existing_df is not None else new_df
        combined_df.write_parquet(parquet_path)

        # Prints number of rows in parquet (note: used this for a sanity check to ensure data was saving correctly)
        print(f'Updated {parquet_path} with {len(new_df)} new rows for {ssp}')

# Ensures script can be run directly through terminal - found implementing the parallelism through the terminal drastically sped things up when running across the GCM-ensemble and prevented kernel crashes occuring in the notebook
if __name__ == "__main__":
    GCMs = ['BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2', 'EC-Earth3-Veg', 'FGOALS-f3-L', 'GFDL-ESM4', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    for GCM in GCMs:
        GCM_run(GCM)