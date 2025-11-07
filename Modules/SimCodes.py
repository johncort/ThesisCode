# Imports
import os
import re
import time
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import rioxarray as rxr
from tqdm import tqdm
import xarray as xr

import oggm.cfg as cfg
from oggm import utils, workflow, tasks, DEFAULT_BASE_URL
from oggm.shop import gcm_climate
from oggm.sandbox import distribute_2d


def chunk_sim(GCM, valid=None, process_chunk_size=50, clear_previous=False):
    """
    Runs glacier simulations on a chunk of glaciers and append results to CSV.
    Note: OGGM's tutorial was helpful for formatting the OGGM simulation functions used:
    https://tutorials.oggm.org/stable/notebooks/10minutes/run_with_gcm.html 
    All OGGM functions can be viewed at: https://docs.oggm.org/en/latest/api.html 

    Arguments:
        GCM (str): GCM name 
        valid (pd.Series): the subset of glacier IDs to process in this chunk
        process_chunk_size (int): number of glaciers per process batch
        batch_size (int): number of glaciers per batch
        clear_previous (bool): clears previous CSV if True

    Returns: 
        None

    Raises: 
        ValueError: If GCM not supported
        FileNotFoundError: If precipitation data not found
        FileNotFoundError: If temperature data not found

    """

    # Lists supported GCM ensemble (note: this is just set based upon all the GCMs used in the study but could be expanded to others)
    supported_GCMs = [
        'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2',
        'EC-Earth3-Veg', 'FGOALS-f3-L', 'GFDL-ESM4',
        'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM'
    ]
    if GCM not in supported_GCMs:
        raise ValueError(f"GCM '{GCM}' is not supported. Choose from: {supported_GCMs}")

    # Defines filepaths to read RGI IDs and to save volume and climate outputs (note: had to deal with inconsistent use of - and _ in filenames)
    id_path = 'regional_rgi_ids.csv'
    output_csv = f'DemoExport{GCM.replace("-", "_")}_Output.csv'
    climate_csv = f'DemoExport{GCM.replace("-", "_")}_ClimateData.csv'  

    # If valid is None loads full list of IDs, otherwise proceeds with the passed subset of IDs (note: I just added this as a fallback in case no IDs passed but this shouldn't really happen)
    if valid is None:
        valid = (
            pd.read_csv(id_path)
            .stack()
            .reset_index(drop=True)
        )

    # Wipes output_csv and climate_csv if clear_previous is set to True (note: should be set to false once runs are underway but was helpful if having to restart)
    if clear_previous:
        if os.path.exists(output_csv):
            os.remove(output_csv)
            print(f'Existing {output_csv} cleared')
        if os.path.exists(climate_csv):
            os.remove(climate_csv)
            print(f'Existing {climate_csv} cleared')

    print(f'Processing {len(valid)} glaciers')

    # Creates empty list for all processed glaciers and dictionary to store climate data for methods section
    all_gdirs = [] 
    climate_dict = {}  

    # Divides passed RGI IDs according to size of processing chunks
    total_chunks = (len(valid) + process_chunk_size - 1) // process_chunk_size

    # Initiates simulation loop for specified number of glaciers 
    # Note: I played around with the size of chunks and number of glaciers processed at once, 50 was deemed efficient whilst preventing slowdown but this probably varies with computers
    for i, idx in enumerate(tqdm(range(0, len(valid), process_chunk_size),
                                 desc=f'Glacier processing'),
                          start=1):
        valid_chunk = valid[idx:idx + process_chunk_size]

        try:
            # Initiates glacier directories, uses level 5 preprocessed directories, and reset set to False to prevent redownloads
            gdirs = workflow.init_glacier_directories(
                valid_chunk,
                from_prepro_level=5,
                prepro_base_url=DEFAULT_BASE_URL,
                reset=False
            )

            # Path to folder containing all GCM files downloaded from OGGM's server locally (by default OGGM keeps these within its working directory structure, but I had to store on SSD for storage)
            climate_files = 'custom_filepath'

            # Accesses GCM files (note: by default GCM output would instead be accessed via OGGM's server)  
            for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:

                # Defines filepaths to precipitation and temperature data
                fp = f'{climate_files}/{GCM}/{GCM}_{ssp}_r1i1p1f1_pr.nc'
                ft = f'{climate_files}/{GCM}/{GCM}_{ssp}_r1i1p1f1_tas.nc'

                # Ensures files exist before processing
                if not os.path.exists(fp):
                    raise FileNotFoundError(f'Precipitation file not found: {fp}')
                if not os.path.exists(ft):
                    raise FileNotFoundError(f'Temperature file not found: {ft}')

                # Note: the following code largely follows the common OGGM workflow outlined in: https://tutorials.oggm.org/stable/notebooks/10minutes/run_with_gcm.html 
                # Processes climate data, including bias correction (note: y0 and y1 are used to define the period over which data should be processed, OGGM reccommends a buffer either side)
                workflow.execute_entity_task(
                    gcm_climate.process_cmip_data,
                    gdirs,
                    filesuffix=f'_CMIP6_{GCM}_{ssp}',
                    fpath_temp=ft,
                    fpath_precip=fp,
                    y0=1946,
                    y1=2115
                )

                # Reads climate data and stores in dict for separate analysis conducted when plotting in methods section
                all_data = []
                for gdir in gdirs:
                    try:
                        fpath = gdir.get_filepath('gcm_data', filesuffix=f'_CMIP6_{GCM}_{ssp}')
                        ds = xr.open_dataset(fpath)
                        df = ds.to_dataframe().reset_index()
                        df['RGI_ID'] = gdir.rgi_id
                        df['Scenario'] = ssp
                        all_data.append(df)
                    
                    # Prints failure but allows loop to continue (note: I had issues at the start where I'd get far into a chunk before 1 glacier would break the loop 
                    # so this method seemed to prevent this recurring)
                    except Exception as e:
                        tqdm.write(f'Failed to load for {gdir.rgi_id} ({ssp}): {e}')
                        continue
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    climate_dict[f'{idx}_{ssp}'] = combined_df

            # Runs glacier simulations - again ys and ye used to specify the period over which the sim is run
            for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
                rid = f'_CMIP6_{GCM}_{ssp}'
                workflow.execute_entity_task(
                    tasks.run_from_climate_data,
                    gdirs,
                    ys=2020,
                    ye=2100,
                    climate_filename='gcm_data',
                    climate_input_filesuffix=rid,
                    init_model_filesuffix='_historical',
                    output_filesuffix=rid,
                    store_model_geometry=False,
                    store_fl_diagnostics=True,
                )

            all_gdirs.extend(gdirs)

        # Again prints failure but allows progress to continue
        except Exception as e:
            print(f'Error processing glaciers {idx} to {idx + len(valid_chunk)}: {e}')
            continue

    # Compiles simulation outputs for each SSP (Note: had to set batch_size below 1000 because >999 would crash the notebook's kernel - unsure why)
    batch_size = 999
    ssp_dfs = []
    for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        rid = f'_CMIP6_{GCM}_{ssp}'
        batch_dfs = []

        for i in range(0, len(all_gdirs), batch_size):
            batch = all_gdirs[i:i + batch_size]
            try:
                # Uses OGGM's compile_run_output function to compile simluation outputs and then stores in DataFrame
                # Note: At the time of simulating was not familiar with Polars, however it was more efficient for plotting and likely the same if used here
                ds_batch = utils.compile_run_output(batch, input_filesuffix=rid)
                df_batch = ds_batch.to_dataframe().reset_index()
                keep_cols = ['time', 'rgi_id', 'calendar_year', 'volume', 'area', 'length']
                existing_cols = [col for col in keep_cols if col in df_batch.columns]
                df_batch = df_batch[existing_cols]
                batch_dfs.append(df_batch)
            # Note: if a batch fails, whilst the loop shouldn't break it needs attention as a lot of data will thus be missing
            except Exception as e:
                print(f"Error compiling batch {i} for SSP {ssp}: {e}")
                continue

        # Aggregates outputs into single DataFrame for each SSP 
        if batch_dfs:
            ssp_df = pd.concat(batch_dfs, ignore_index=True)
            rename_dict = {col: f'{col}_{ssp}' for col in ['volume', 'area', 'length'] if col in ssp_df.columns}
            ssp_df = ssp_df.rename(columns=rename_dict)
            ssp_df = ssp_df.drop_duplicates(subset=['time', 'rgi_id', 'calendar_year'])
            ssp_dfs.append(ssp_df)

    if not ssp_dfs:
        print('No SSP dataframes created, skipping save')
        return

    # Merges SSP DataFrames for final output, adding data under all SSPs to a single row per glacier and time step
    combined_df = reduce(lambda left, right: pd.merge(left, right,
                                                      on=['time', 'rgi_id', 'calendar_year'],
                                                      how='outer'), ssp_dfs)

    # Exports data, either via appending to existing file or if clear_previous is True/output file doesn't exist, rewrites the file
    if clear_previous or not os.path.exists(output_csv):
        combined_df.to_csv(output_csv, index=False)
    else:
        existing_df = pd.read_csv(output_csv)
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['time', 'rgi_id', 'calendar_year'], inplace=True)
        combined_df.to_csv(output_csv, index=False)

    # Applies same logic in order to export climate data (note: added an additional check to deal with missing columns as this was an issue in early versions,
    # however is likely not needed now)
    combined_climate_df = pd.DataFrame()
    climate_combined_df = pd.concat(climate_dict.values(), ignore_index=True)
    cols = ['time', 'RGI_ID', 'Scenario', 'dayofyear']
    existing_cols = [col for col in cols if col in climate_combined_df.columns]
    if clear_previous or not os.path.exists(climate_csv):
        climate_combined_df.to_csv(climate_csv, index=False)
        combined_climate_df = climate_combined_df
    else:
        existing_df = pd.read_csv(climate_csv)
        combined_climate_df = pd.concat([existing_df, climate_combined_df], ignore_index=True)
        combined_climate_df.drop_duplicates(subset=existing_cols, inplace=True)
        combined_climate_df.to_csv(climate_csv, index=False)

    print(f"\nChunk complete, glacier results saved to {output_csv} and climate data saved to {climate_csv}")

def multi_sim(GCM, num_chunks=10, process_chunk_size = 50, clear_previous=False):
    """
    Splits glacier IDs into argued number of chunks and runs glacier simulation on each chunk.

    Arguments:
        GCM (str): GCM name
        num_chunks (int): Number of chunks to split glacier IDs into
        process_chunk_size (int): number of glaciers per process batch (passed to chunk_sim)
        batch_size (int): number of glaciers per batch (passed to chunk_sim)
        clear_previous (bool): clears previous CSV data if True (only for first chunk)

    Returns:
        None

    Raises:
        None

    """
    # Determines level of log printing 
    cfg.initialize(logging_level='WARNING')
    # Sets path to working directory
    cfg.PATHS['working_dir'] = 'custom_filepath'

    # Optional: Determines whether multiprocessing is used (note: this is faster for sims with many glaciers so is recommended)
    cfg.PARAMS['use_multiprocessing'] = True

    # Optional: Chooses whether to store additional model output (note: needed fl_diagnostics for the hypsometry and ice thickness sections)
    cfg.PARAMS['store_fl_diagnostics'] = True
    cfg.PARAMS['store_model_geometry'] = False

    # Defines filepath to RGI IDs csv (this is a csv containing all glaciers filtered for the sample)
    id_path = 'regional_rgi_ids.csv'

    # Reads all glacier IDs (note: for demo, I capped the number of glacier IDs being read - for normal runs remove this indexing)
    all_ids = (pd.read_csv(id_path).stack(future_stack=True).dropna().reset_index(drop=True))[:4] 

    # Calculates chunk size 
    total_len = len(all_ids)
    chunk_size = (total_len + num_chunks - 1) // num_chunks  

    # Loops through number of chunks, running chunk_sim function
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_len)
        valid_chunk = all_ids[start_idx:end_idx].reset_index(drop=True)

        print(f'Starting chunk {i + 1}/{num_chunks} with {len(valid_chunk)} glaciers')

        chunk_sim(
            GCM=GCM,
            valid=valid_chunk,
            process_chunk_size=process_chunk_size,
            clear_previous=clear_previous
        )


def thickness_sim(rgi_ids):
    """
    Runs OGGM's glacier volume simulation and ice thickness redistribution 
    All OGGM functions can be viewed at: https://docs.oggm.org/en/latest/api.html 

    Arguments:
        rgi_ids (list): List of RGI IDs
    
    Returns: 
        None

    Raises: 
        None
    
    """

    # Defines GCMs and SSPs
    GCMs = [
        'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2',
        'EC-Earth3-Veg','FGOALS-f3-L', 'GFDL-ESM4',
        'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM'
    ]
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    # Initialises OGGM and sets up working directory 
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = 'custom_filepath'
    cfg.PARAMS['use_multiprocessing'] = True

    # Initialises pre-processed glacier directories (Note: Need to be Level 4 for ice distribution sims)
    gdirs = workflow.init_glacier_directories(
        rgi_ids,
        from_prepro_level=4,
        prepro_border=80,
        prepro_base_url=DEFAULT_BASE_URL, 
        reset = False
    )

    # Loops through GCMs
    for GCM in GCMs:

        # Defines path to folder containing GCM outputs (Note: If this doesn't exist, OGGM should just automatically download them)
        climate_files = 'custom_filepath'

        # Loops through glaciers in glacier directories (gdirs)
        for gdir in gdirs:
            rgi_id = gdir.rgi_id
            print(f'Processing {rgi_id} for {GCM}')

            # Loops through SSPs
            for ssp in ssps:
        
                rid = f'_CMIP6_{GCM}_{ssp}'

                # Defines path to GCM's precipitation and temperature files
                fp = f'{climate_files}/{GCM}/{GCM}_{ssp}_r1i1p1f1_pr.nc'
                ft = f'{climate_files}/{GCM}/{GCM}_{ssp}_r1i1p1f1_tas.nc'

                # Processes climate data for each glacier 
                workflow.execute_entity_task(
                    gcm_climate.process_cmip_data,
                    gdir,
                    filesuffix=rid,
                    fpath_temp=ft,
                    fpath_precip=fp,
                    y0=1946,
                    y1=2115
                )

            # Loops through SSPs
            for ssp in ssps:

                rid = f'_CMIP6_{GCM}_{ssp}'

                # Runs OGGM glacier sim (Note: This is the same as in the main simulation function)
                workflow.execute_entity_task(
                    tasks.run_from_climate_data,
                    gdir,
                    ys=2020,
                    climate_filename='gcm_data',
                    climate_input_filesuffix=rid,
                    init_model_filesuffix='_historical',
                    output_filesuffix=rid,
                    ye=2100,
                    store_model_geometry=False,
                    store_fl_diagnostics=True,
                )

                # Runs OGGM's ice thickness redistribution
                # Note: This section is adapted from OGGM's 'Display glacier area and thickness changes on a grid' tutorial
                # Available at: https://tutorials.oggm.org/stable/notebooks/tutorials/distribute_flowline.html (Accessed 10/08/25)
                distribute_2d.add_smoothed_glacier_topo(gdir)
                tasks.distribute_thickness_per_altitude(gdir)
                distribute_2d.assign_points_to_band(gdir)
                ds = distribute_2d.distribute_thickness_from_simulation(
                    gdir,
                    input_filesuffix=rid,
                    concat_input_filesuffix='_historical',
                )

                # Exports NetCDF file to output path (Note: Played around with encoding because storage was becoming an issue but this is completely optional)
                fname = f'custom_filepath/{rgi_id}_{GCM}_{ssp}.nc'
                ds['simulated_thickness'].to_netcdf(
                    fname,
                    encoding={'simulated_thickness': {'zlib': True, 'complevel': 6}}
                )
                print(f'Saved for {ssp}')


def thickness_ensemble(rgi_ids):
    """
    Loads redistributed ice thickness output, calculates GCM-ensemble mean change, exports to GeoTiffs

    Arguments:
        rgi_ids (list): List of RGI IDs (strings)

    Returns:
        None

    Raises: 
        None

    """

    # Initialises OGGM
    cfg.initialize(logging_level='WARNING') 
    cfg.PATHS['working_dir'] = 'custom_filepath'
    cfg.PARAMS['use_multiprocessing'] = True

    # Defines path to folder containing redistributed ice thickness outputs
    data_dir = 'custom_filepath'

    # Defines SSPs and GCMs
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    GCMs = [
        'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2',
        'EC-Earth3-Veg', 'FGOALS-f3-L', 'GFDL-ESM4',
        'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM'
    ]

    # Loops through glaciers
    for rgi_id in rgi_ids:
        print(f'Processing {rgi_id}')

        # Initialises OGGM glacier directory (this is an easy way to get the coords for exporting to GeoTiff)
        gdir = workflow.init_glacier_directories(
            rgi_id, from_prepro_level=4, prepro_border=80,
            prepro_base_url=utils.DEFAULT_BASE_URL, reset=False
        )

        # Loops through SSPs
        for ssp in ssps:
            GCM_datasets = []

            # Loops through GCMs
            for GCM in GCMs:

                # Forms filepath and opens
                fname = os.path.join(data_dir, f'{rgi_id}_{GCM}_{ssp}.nc')
                if not os.path.exists(fname):
                    print(f'File not found {fname}, skipping {GCM}.')
                    continue
                ds = xr.open_dataset(fname, engine='netcdf4')
                da = ds['simulated_thickness'].load()

                # Computes ice thickness change between 2025 and last available year (which is 2100 in my simulations) 
                da_change = da.isel(time=-1).fillna(0) - da.sel(time=2025).fillna(0)
                GCM_datasets.append(da_change)
                ds.close()

            if not GCM_datasets:
                print(f'No valid model data found for SSP {ssp}')
                continue

            # Computes GCM-ensemble mean
            combined = xr.concat(GCM_datasets, dim='GCM')
            ensemble_mean = combined.fillna(0).mean(dim='GCM')

            # Exports as GeoTIFF using the OGGM DEM for transformation (note: I decided to plot in QGIS to have more freedom with the formatting)
            dem = rxr.open_rasterio(gdir[0].get_filepath('dem'), masked=True).squeeze()
            ensemble_mean = ensemble_mean.rio.write_crs(dem.rio.crs)
            ensemble_mean.rio.write_transform(dem.rio.transform(), inplace=True)
            out_dir = os.path.join(data_dir, 'Demo')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'{rgi_id}_ensemble_{ssp}.tif')
            ensemble_mean.rio.to_raster(out_path)
            print(f'Exported GeoTIFF for {ssp}')

            # Adds a small pause because the kernel occasionally crashed during the loop and this seemed to avoid it 
            time.sleep(2)
