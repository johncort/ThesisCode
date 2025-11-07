# Imports
import polars as pl
import os

def read_climate_data(GCMs):

    """
    Reads in csv files containing climate data for all glaciers and converts to parquet files
    Note: These files are very large (~ 5 GB each) so this significantly speeds up the runtime for the processing function 
    This function thus only needs to be run once as all subsequent functions use the parquet files

    Arguments:
        GCMs (list): List of GCMs for processing (strings)

    Returns:
        None

    Raises
        None

    """
    
    # Loops through GCMs
    for GCM in GCMs:

        # Defines filepath and attempts to read file
        csv_path = f'custom_filepath/{GCM}_ClimateData.csv'

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            continue

        # Reads CSV with polars (note: played around with changing 'Scenario' to categorical to try to speed up read time for these big files, think this works?)
        df = (
            pl.read_csv(csv_path)
            .with_columns(pl.col('Scenario').cast(pl.Categorical))
        )

        # Saves CSV data to parquet file
        parquet_path = f'{GCM}_ClimateData.parquet'
        df.write_parquet(parquet_path)
        print(f'Saved {parquet_path}')

        # Adds a pause to prevent kernel from crashing (note: noticed this happened sometimes after 3-4 GCMs)
        time.sleep(2)

    print('All valid files converted')

def process_climate_data(GCMs):

    """
    Reads and processes climate data from parquet files, calculating anomalies 

    Arguments:
        GCMs (list): List of GCMs for processing (strings)

    Returns: 
        ensemble_dict (dict): Dictionary containing subregional temperature and precipitation anomalies across the GCM ensemble

    Raises:
        None
    """

    # Reads CSV containing RGI IDs pre-filtered into their subregions
    subregional_ids = pl.read_csv('regional_rgi_ids.csv')

    # Creates empty dictionary to store ensemble output 
    ensemble_dict = {}

    # Loops through GCMs
    for GCM in GCMs:
        print(f'Processing {GCM}')

        # Reads each GCM's parquet file
        df = pl.read_parquet(f'{GCM}_ClimateData.parquet')

        # Creates empty dictionary to store subregional data
        subregional_dict = {}

        # Loops through subregions
        for subregion in range(1, 7):

            # Performs filtering and takes mean temperature and precipitation, grouping by year
            rgi_list = subregional_ids[str(subregion)].drop_nulls().to_list()
            df_subregion = (
                df.filter(pl.col('RGI_ID').is_in(rgi_list))
                .with_columns(pl.col('time').dt.year().alias('year'))
                .group_by(['year', 'RGI_ID', 'Scenario'])
                .agg([
                    pl.col('prcp').mean().alias('prcp'),
                    pl.col('temp').mean().alias('temp')
                ])
                .sort(['year', 'RGI_ID', 'Scenario'])
            )

            # Computes the baseline climatology (1961–1990)
            climatology = (
                df_subregion.filter(pl.col('year').is_between(1961, 1990))
                .group_by(['RGI_ID', 'Scenario'])
                .agg([
                    pl.col('prcp').mean().alias('prcp_mean'),
                    pl.col('temp').mean().alias('temp_mean')
                ])
            )

            # Joins and computes anomalies relative to the 1961-90 baseline, dropping the original mean and temp columns
            df_subregion = (
                df_subregion.join(climatology, on=['RGI_ID', 'Scenario'], how='left')
                .with_columns([
                    ((pl.col('prcp') - pl.col('prcp_mean')) / pl.col('prcp_mean') * 100).alias('prcp_anom_1961_1990'),
                    (pl.col('temp') - pl.col('temp_mean')).alias('temp_anom_1961_1990')
                ])
                .drop(['prcp_mean', 'temp_mean'])
            )

            # Stores df in dictionary
            subregional_dict[subregion] = df_subregion

        # Stores all subregional data for GCM in dictionary that is then used in plotting 
        ensemble_dict[GCM] = subregional_dict

        print(f'Processed {GCM}')

    return ensemble_dict