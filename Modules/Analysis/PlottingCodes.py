#Imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner as gridliner
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl
import scikit_posthocs as sp
import xarray as xr
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import kruskal, shapiro

import oggm
from oggm import cfg, graphics, tasks, utils, workflow
from oggm.core import massbalance

# Sets font to Arial to match my thesis  
plt.rcParams['font.family'] = 'Arial'

def plot_ERA5(yr0 = 1981, yr1 = 2010):

    """
    Loads, processes, and plots ERA5 T2m anomalies (used for figure in introduction)

    Arguments:
        yr0 (int): Start year for baseline climatology (default = 1981)
        yr1 (int): End year for baseline climatology (default = 2010)

    Returns:
        None 

    Raises:

    """
    
    # Loads NetCDF file and extracts temperature variable (note: I was using the T2 monthly averaged data)
    filepath = 'tempERA5.nc'
    ds = xr.open_dataset(filepath)
    temp = ds['t2m'].rename({'valid_time': 'time'}).sortby('time')

    # Calculates seasonal means (note the handling of DJF overlapping 2 years necessitates the use of .resample, however I could only get this
    # to work if I subsequently sliced by time rather than year as in JJA)
    JJA = temp.sel(time=temp['time.month'].isin([6, 7, 8])).groupby('time.year').mean('time')
    DJF = (
        temp
        .sel(time=temp['time.month'].isin([12, 1, 2]))
        .resample(time='QS-DEC')                        
        .mean('time')                                   
    )

    # Calculates annual mean
    annual = temp.groupby('time.year').mean('time')

    # Calculates baseline 1981-2010 climatologies (note the aforementioned careful handling of DJF using time slicing)
    JJA_clim = JJA.sel(year=slice(yr0, yr1)).mean('year')
    DJF_clim = DJF.sel(time=slice(f'{yr0}-01-01', f'{yr1}-12-31')).mean('time')
    annual_clim = annual.sel(year=slice(yr0, yr1)).mean('year')

    # Calculates anomalies from baseline climataology 
    JJA_anom = JJA - JJA_clim
    DJF_anom = DJF - DJF_clim
    annual_anom = annual - annual_clim

    # Creates dictionary containing anomaly climatologies
    anomaly_dict = {
        'Annual': annual_anom.sel(year=slice(1991, 2020)).mean(dim='year'),
        'DJF': DJF_anom.sel(time=slice('1991-01-01', '2020-12-31')).mean('time'),
        'JJA': JJA_anom.sel(year=slice(1991, 2020)).mean(dim='year'),
    }

    # Sets projection manually (Note this would obviously need to be changed if another region is used instead)
    # Played around with standard_parallels until I deemed the plot looked reasonable
    proj = ccrs.AlbersEqualArea(
        central_longitude=-150,
        central_latitude=63,
        standard_parallels=(55, 65)
    )

    # Creates figure
    fig = plt.figure(figsize=(12, 10))

    # Positions axes manually (this took a while to finetune)
    ax_pos = {
        'Annual': [0.08, 0.58, 0.84, 0.37], 
        'DJF':    [0.08, 0.26, 0.465, 0.25], 
        'JJA':    [0.44, 0.26, 0.465, 0.25], 
        'cbar':   [0.25, 0.18, 0.5, 0.025],   
    }

    # Creates axes for plotting then loops through dict and plots on each axis
    axes = {}
    for label in ['Annual', 'DJF', 'JJA']:
        axes[label] = fig.add_axes(ax_pos[label], projection=proj)

    for label, anom in anomaly_dict.items():
        ax = axes[label]

        # Conducts interpolation to make plot smoother (4x resolution)
        lat_new = np.linspace(anom.latitude.min().item(), anom.latitude.max().item(), anom.latitude.size * 4)
        lon_new = np.linspace(anom.longitude.min().item(), anom.longitude.max().item(), anom.longitude.size * 4)
        anom_interp = anom.interp(latitude=lat_new, longitude=lon_new)

        # Sets vmin/vmax (if using the default 1981-2010 baseline like in my introduction vmin/vmax are predefined)
        if yr0 == 1981 and yr1 == 2010:
            vmin, vmax = -2, 2
        else:
            vmin = anom_interp.min().item()
            vmax = anom_interp.max().item()
        
        # Plots interpolated anomalies (again the extent is hardcoded here to suit Alaska so would need altering if another region used)
        ax.set_extent([-175, -120, 50, 72], crs=ccrs.PlateCarree())
        im = anom_interp.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            add_colorbar=False
        )
        ax.set_title(f'{label}', fontsize=16)

        # Adds coastlines and Alaska-Canada border
        ax.coastlines(linewidth = 0.5)
        ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth = 0.5)

        # Adds gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='black', alpha=0.5,
                        xlocs=[], ylocs=range(50, 75, 15))
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True
        gl.bottom_labels = False
        gl.ylabel_style = {'size': 12}
        ax.gridlines(draw_labels=False, linewidth=0.5, color='black', alpha=0.2,
                    xlocs=range(-180, -100, 15), ylocs=[])

        # Adds manual longitude tick labels since the default ones weren't suitable
        # (Note: I managed this by performing some transformations between coordinate systems to place the text but I am sure there is a more pythonic method)
        for lon in range(-165, -120, 15):
            x_disp, y_disp = proj.transform_point(lon, 50, ccrs.PlateCarree())
            x_axes, _ = ax.transData.transform((x_disp, y_disp))
            x_axes = ax.transAxes.inverted().transform((x_axes, 0))[0]
            ax.text(x_axes, -0.02, f'{abs(lon)}°W',
                    ha='center', va='top', fontsize=12, transform=ax.transAxes)

    # Adds a shared colourbar 
    cbar_ax = fig.add_axes(ax_pos['cbar'])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Temperature Anomaly (°C)', fontsize=16, labelpad = 10)

    # Uses preselected tick range if using 1981-2010 (note: found that -2 to 2 was the best for the default baseline, but again this is data dependent)
    if yr0 == 1981 and yr1 == 2010:
        ticks = np.arange(-2, 2 + 0.1, 1)
        cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=13)  

    # Adds subplot labels
    fig.text(0.08, 0.97, 'a', fontsize=16, fontweight='bold', va='top', ha='left') 
    fig.text(0.08, 0.54, 'b', fontsize=16, fontweight='bold', va='top', ha='left')  

    plt.savefig('custom_filepath/ERA5TempAnomsFig.png', dpi = 300)
    plt.show()

def climate_plots(results, variable='temp', clim_start=2071, clim_end=2100):
    """
    Plots a timeseries of GCM-ensemble anomalies relative to 1961-90 for temp/prcp under each SSP within each subregion (with shaded uncertainty)

    Arguments:
        results (dict): Dictionary containing processed climate data (obtained by running process_climate_data.py)
        variable (str): Climate data for plotting (temp/prcp)
        clim_start (int): Start year for projection climatology 
        cim_end (int): End year for projection climatology 
    
    Returns:
        None

    Raises:
        ValueError: If variabe is not 'temp' or 'prcp'

    """

    # Define labels and colours
    ssp_colours = {
        'ssp126': '#059105',
        'ssp245': '#115b8f',
        'ssp370': '#ff9d00',
        'ssp585': '#ff0000'
    }
    ssp_labels = {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5'
    }

    # Defines conditional labelling for plotting and saving and ensures a valid variable is passed
    if variable == 'temp':
        variable_label = 'Temperature Anomaly (°C)' 
        save_variable = 'Temperature'
    elif variable == 'prcp':
        variable_label = 'Precipitation Anomaly (%)'
        save_variable = 'Precipitation'
    else:
        raise ValueError(f'{variable} is not a valid variable (choose temp or prcp)')

    # Creates list of all GCMs within processed data
    GCMs = list(results.keys())

    # Defines SSPs
    ssp_list = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    # Creates subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    # Loops through subregions
    for i, subregion in enumerate(range(1, 7)):
        ax = axes[i]
        stats_texts = []

        # Loops through SSPs and GCMs
        for ssp in ssp_list:
            GCM_means = []
            for GCM in GCMs:

                # Accesses data for each SSP, calculates the subregional mean for each year, and converts to pandas for plotting
                df = results[GCM][subregion]
                df_ssp = (
                    df.filter(pl.col('Scenario') == ssp)
                    .group_by('year')
                    .agg(pl.col(f'{variable}_anom_1961_1990').mean().alias('mean'))
                    .to_pandas()
                )
                GCM_means.append(df_ssp.set_index('year')['mean'])
            
            # Takes the GCM ensemble mean and SD
            combined = pd.concat(GCM_means, axis=1)
            avg_series = combined.mean(axis=1)
            sd_series = combined.std(axis=1)

            # Masks the data within the argued projection climatology and calculates the mean and SD
            clim_mask = (avg_series.index >= clim_start) & (avg_series.index <= clim_end)
            clim_mean = avg_series[clim_mask].mean()
            clim_sd = sd_series[clim_mask].mean()

            # Plots GCM-ensemble, subregional mean anomalies with shaded SD
            colour = ssp_colours.get(ssp, None)
            ssp_label = ssp_labels.get(ssp, ssp)
            ax.plot(avg_series.index, avg_series.values, label=ssp_label if i==0 else None, color=colour, linewidth=2)
            ax.fill_between(avg_series.index, avg_series - sd_series, avg_series + sd_series, alpha=0.1, color=colour)

            # Appends climatological data for printing summary stats later
            stats_texts.append((ssp_label, clim_mean, clim_sd, colour))

        # Plot formatting: includes conditional labelling of axis and handles y lims (this was done manually based on an initial plot)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f'Subregion 0{subregion}', fontsize=26)

        if variable == 'temp':
            ax.set_ylim(-3, 11)
        else:
            ax.set_ylim(-45, 80)
        ax.set_xlim(2020, 2100)
        ax.grid(False)
        ax.tick_params(labelsize=22)

        if i >= 3:
            ax.set_xlabel('Year', fontsize=30, labelpad=12)
        if i % 3 == 0:
            ax.set_ylabel(variable_label, fontsize=30, labelpad=12)
        if i == 0:
            ax.legend(fontsize=19.5, loc='upper left', bbox_to_anchor=(0, 1.02), frameon=False)

        ax.set_xticks(range(2020, 2101, 20))

        # Prints summary statistics in each subplot (note: I had to customise this based upon whether the variable is temp or prcp)  
        y_start = 0.015
        y_spacing = 0.045
        if variable =='temp':
            for idx, (label_text, mean_val, sd_val, col) in enumerate(stats_texts):
                text_str = f'{clim_start}-{clim_end}: Mean = {mean_val:.2g} ± {sd_val:.2g} °C'
                ax.text(0.98, y_start + idx * y_spacing, text_str,
                        transform=ax.transAxes,
                        ha='right', va='bottom',
                        fontsize=18.5,
                        color=col,
                        alpha=1,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.0, edgecolor='none'))
        else:
            for idx, (label_text, mean_val, sd_val, col) in enumerate(stats_texts):
                text_str = f"{clim_start}-{clim_end}: Mean = {mean_val:.2g} ± {sd_val:.2g}%"
                ax.text(0.98, y_start + idx * y_spacing, text_str,
                        transform=ax.transAxes,
                        ha='right', va='bottom',
                        fontsize=18.5,
                        color=col,
                        alpha=1,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.0, edgecolor='none'))

    # Adds a/b label to figure
    subplot_label = 'a' if variable == 'temp' else 'b'
    fig.text(0.0, 0.999, subplot_label, fontsize=27, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig(f'custom_filepath/{save_variable}.png', dpi=300)
    plt.show()

def mass_balance_demo(rgi_ids):

    """
    Runs OGGM mass balance calibration and plots comparisons for visualisations
    All OGGM functions can be viewed at: https://docs.oggm.org/en/latest/api.html 

    Arguments: 
        rgi_ids(list): List of RGI IDs (note: will need to check for glaciers that have in-situ measurements on OGGM's server)

    Returns:
        None
        
    Raises:
        None
    """

    # Initialises OGGM and creates a temporary working directories for selected glaciers - note: need to set border to 10 
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = utils.gettempdir(dirname='mass_balance_demo', reset=True)
    cfg.PARAMS['border'] = 10
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/'
    gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_base_url=base_url)

    # Creates subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # Loops through glaciers
    for i, gdir in enumerate(gdirs):

        ax = axes[i]
        # Reads the calibrated and reference mass balance between 2000-2020 from the glacier directory
        # Note: The following 9 lines are adapted from OGGM (no date) 'A look into the new mass balance calibration in OGGM v1.6' 
        # Available at: https://tutorials.oggm.org/stable/notebooks/tutorials/massbalance_calibration.html
        height, width = gdir.get_inversion_flowline_hw()
        mb_model = massbalance.MonthlyTIModel(gdir)
        years = np.arange(2000, 2020)
        mb_df = pd.DataFrame(index=years)
        mb_df['mod_mb'] = mb_model.get_specific_mb(height, width, year=mb_df.index)
        ref_mb = gdir.get_ref_mb_data().loc[2000:2020]
        mb_df['in_situ_mb'] = ref_mb['ANNUAL_BALANCE']
        avg_model = mb_df['mod_mb'].mean()
        avg_obs = mb_df['in_situ_mb'].mean()

        # Adds horizontal line to plot
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=2.5)

        # Plots modelled and reference mass balance timeseries (converted to m w.e.)
        ax.plot(
            mb_df.index,
            mb_df['mod_mb'] / 1000,
            label=f"Modelled (Average: {avg_model / 1000:.2f} m w.e.)",
            color="#005DAF",
            lw=2.5,
            ls='-'
        )
        ax.plot(
            mb_df.index,
            mb_df['in_situ_mb'] / 1000,
            label=f"In-situ (Average: {avg_obs / 1000:.2f} m w.e.)",
            color="#000000",
            lw=2.5,
            ls='--'
        )

        # Plot formatting
        ax.set_title(f"{gdir.rgi_id}", fontsize=21, pad = 60)
        ax.grid(axis = 'both', linestyle = '--', alpha = 0.4)
        ax.set_xlabel('Year', fontsize=26, labelpad = 12)
        ax.set_xlim(2000, 2019)
        ax.set_xticks(np.arange(2000, 2021, 5))
        
        # Conditional labelling for subplots
        if i == 0:
            ax.set_ylabel(r'Mass balance (m.w.e.)', fontsize=26, labelpad = 12)
        else:
            ax.set_ylabel('') 

        if i == 0:
            ax.tick_params(axis='y')
        else:
            ax.tick_params(axis='y', labelleft=False)

        ax.tick_params(axis='both', labelsize=20)

        # Reads calibration parameters (melt factor, precipitation factor, temperature bias, and adds them as annotations to bottom of subplots
        calib = gdir.read_json("mb_calib")
        melt_fac = calib['melt_f']
        prcp_fac = calib['prcp_fac']
        temp_bias = calib['temp_bias']

        param_text = (
            f"Melt factor: {melt_fac:.2f}\n"
            f"Precipitation bias: {prcp_fac:.2f}\n"
            f"Temperature bias: {temp_bias:.2f}"
        )

        ax.text(
            0.02, 0.02, param_text,
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='bottom',
            horizontalalignment='left'
        )

        # Adds legend above subplots to improve figure clarity
        ax.legend(
            fontsize=18,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15),
            frameon=False
            )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.175) 
    plt.savefig('custom_filepath/MassBalanceCalib.png', dpi = 300)
    plt.show()

def process_plot(GCMs, totalregion=False, statistics = False, stat_sub = False):

    """
    Loads, processes, and plots simulated glacier volume change. 
    Note: This function evolved from simply creating a subregional multipanel plot to additionally plotting for the region and conducting
    statistical analysis as the thesis evolved. I have tried to keep the code concise but by bringing it all into one function there are
    a few areas that are slightly repetitive. Ideally I would've kept it in separate functions but time constraints necessitated this approach instead. 

    Arguments:
        GCMs (list): List of GCM names (strings) within output filenames
        totalregion (bool): Determines if regional plot is created
        statistics (bool): Determines if general stats are printed
        stat_sub (bool): Determines if subregional analysis is conducted

    Returns:
        None 

    Raises:
        None

    """

    # Defines SSPs and their labels for plotting 
    ssp_list = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    ssp_labels = {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5'
    }

    # Loads predefined RGI IDs sorted into subregions (note this is a csv that contains the study's sample glaciers separated by subregion) and organises in dictionary via comprehension
    valids = pl.read_csv('regional_rgi_ids.csv')
    subregional_ids = {subregion_col: valids[str(subregion_col)].drop_nulls().to_list()for subregion_col in range(1, 7)}

    # Reads output data for all argued GCMs and adds to dictionary
    GCM_dfs = {}
    for GCM in GCMs:
        filepath = f'custom_filepath/{GCM}_Output.csv'
        GCM_dfs[GCM] = pl.read_csv(filepath)

    # Optional: If statistics, computes and prints the initial raw volume in cubed km (used for write up) - note: had to rename from stats because of scipy import
    if statistics:
        ref_ssp = ssp_list[0]
        ref_GCM = GCMs[0]

        df = GCM_dfs[ref_GCM]
        initial_vol_m3 = (
            df.filter(pl.col('calendar_year') == 2020)
            .select(f'volume_{ref_ssp}')
            .sum()
            .item()
        )
        initial_vol_km3 = initial_vol_m3 / 1e9 
        print(f'\nInitial total glacier volume (from {ref_GCM}, {ref_ssp}): {initial_vol_km3:.2f} km³')

    # Filters by subregion for each GCM, then ensures dictionary is sorted numerically for plotting (note: used setdefault to help form nested dictionary)
    filtered_dfs = {}
    for GCM, df in GCM_dfs.items():
        for subregion_num, rgi_list in subregional_ids.items():
            filtered_df = df.filter(pl.col('rgi_id').is_in(rgi_list))
            filtered_dfs.setdefault(subregion_num, {})[GCM] = filtered_df
    filtered_dfs = dict(sorted(filtered_dfs.items()))

    # Creates subplots
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    fig.text(0.01, 0.98, 'b', fontsize=21.5, fontweight='bold', va='top', ha='left')

    # Creates dictionary to store total regional outputs and subregional outputs for optional stat testing
    regional_GCM_vols = {ssp: {} for ssp in ssp_list}
    subregion_means = []

    # Loops through each subregion to process volume under each SSP
    for i, (subregion, GCM_dict) in enumerate(filtered_dfs.items()):
        ax = axs[i]
        processed_vol = {ssp: {} for ssp in ssp_list}
        for ssp in ssp_list:
            GCM_series = []

            # Pivots data for calculations 
            # (Note: Once data is read by Polars, I ended up converting back to pandas quite early on here because I'm familiar with it when using matplotlib subsequently)
            for GCM, df in GCM_dict.items():
                data = df.select(['rgi_id', 'calendar_year', f'volume_{ssp}']).to_pandas()
                pivot_data = data.pivot(index='calendar_year', columns='rgi_id', values=f'volume_{ssp}')

                # Conducts normalisation using 2025 as the baseline year 
                total_vol = pivot_data.sum(axis=1)
                baseline = total_vol.loc[2025]
                relative_vol = (total_vol / baseline) * 100
                GCM_series.append(relative_vol)

                # Appends subregion_means which is used later for stat testing between subregions
                subregion_means.append({
                    'subregion': f'S0{subregion}',
                    'GCM': GCM,
                    'SSP': ssp,
                    'mean_norm_vol': relative_vol.loc[2100] 
                })
            
            # Calculates GCM ensemble mean and standard deviation for the subregion 
            combined = np.stack([s.values for s in GCM_series])
            years = GCM_series[0].index.to_numpy()
            processed_vol[ssp]['years'] = years
            processed_vol[ssp]['mean'] = np.nanmean(combined, axis=0)
            processed_vol[ssp]['std'] = np.nanstd(combined, axis=0)

        # Accumulates subregion's data and appends to the regional dictionary for an optional regional plot later (this code began as just subregional analysis but ended up adding regional)
        for ssp in ssp_list:
            for GCM, df in GCM_dict.items():
                data = df.select(['rgi_id', 'calendar_year', f'volume_{ssp}']).to_pandas()
                pivot_data = data.pivot(index='calendar_year', columns='rgi_id', values=f'volume_{ssp}')
                total_vol = pivot_data.sum(axis=1)
                regional_GCM_vols[ssp][GCM] = regional_GCM_vols[ssp].get(GCM, 0) + total_vol

        # Calculates and prints stats for each subregion if True (note: this is entirely optional and should play no role in the plotting but was vital for my writeup)
        if statistics:
            for ssp in ssp_list:
                years = processed_vol[ssp]['years']
                mean = processed_vol[ssp]['mean']
                std = processed_vol[ssp]['std']

                index_2100 = np.where(years == 2100)[0]
                if len(index_2100) == 0:
                    vol_2100, std_2100 = np.nan, np.nan
                else:
                    index_2100 = index_2100[0]
                    vol_2100, std_2100 = mean[index_2100], std[index_2100]

                clim_mask = (years >= 2071) & (years <= 2100)
                clim_mean = np.nanmean(mean[clim_mask])
                clim_std = np.nanmean(std[clim_mask])

                print(f'\nRegion {subregion}, {ssp_labels[ssp]}:')
                print(f'2100 Volume: {vol_2100:.4f}% (SD = {std_2100:.4f}%)')
                print(f'2071–2100 Climatology: {clim_mean:.4f}% (SD = {clim_std:.4f}%)')   

        # Plots the subregion's mean volume and shaded standard deviation under each SSP (note: ended up using a very low alpha because I wasn't happy with readability)
        for ssp, colour in zip(ssp_list, ['#059105', '#115b8f', '#ff9d00', '#ff0000']):
            years, mean, std = processed_vol[ssp]['years'], processed_vol[ssp]['mean'], processed_vol[ssp]['std']
            ax.plot(years, mean, label=ssp_labels[ssp], color=colour, linewidth = 1.9)
            ax.fill_between(years, mean - std, mean + std, color=colour, alpha=0.1)

        # Subplot formatting, including conditional axis labelling to improve grid clarity
        ax.set_title(f'Subregion 0{subregion}', fontsize=22)
        ax.axhline(100, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        if i > 2:
            ax.set_xlabel('Year', fontsize=23, labelpad = 12)
        ax.set_xlim(2025, 2100)
        ax.set_ylim(0, 110)
        if i % 3 == 0:  
            ax.set_ylabel('Volume (% of 2025)', fontsize=22, labelpad = 13)
        ax.tick_params(axis='both', labelsize=17)
        ax.grid(False)
        if i == 2:
            ax.legend(
                loc='upper right', frameon=True, 
                fontsize=18, facecolor='white', 
                framealpha=0.8, edgecolor = 'None')

    plt.tight_layout()
    plt.savefig('custom_filepath/SubregionalVols.png', dpi = 300)
    plt.show()

    # If conducting subregional analysis, proceeds to carry out pairwise comparisons (used for results table)
    if stat_sub:

        # Loops through SSPs, extracting data for each SSP 
        for ssp in ssp_list:
            print(f'\nSSP: {ssp_labels[ssp]}')

            ssp_data = [i for i in subregion_means if i['SSP'] == ssp]

            subregion_df = pd.DataFrame(ssp_data)

            # Runs Shapiro-Wilk test for each subregion (note: I ended up not relying on this too much since subsequent Q-Q plots better visualise non-normality)
            print('Shapiro-Wilk test:')
            for subregion in subregion_df['subregion'].unique().tolist():
                data = subregion_df.loc[subregion_df['subregion'] == subregion, 'mean_norm_vol'].dropna().to_numpy()
                _, p = shapiro(data)
                result = 'Normal' if p > 0.05 else 'Not normal'
                print(f'{subregion}: p = {p:.4f} -  {result}')

                # Plots Q-Q plot for each subregion (S-W test didn't seem to pick up clear non-normality, likely because of sample size?)
                fig, ax = plt.subplots(figsize=(6, 4))
                stats.probplot(data, dist='norm', plot=ax)
                plt.title(f'Q-Q Plot - {subregion} ({result})')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.show()

            # Conducts Kruskal-Wallis test across subregions using GCM means (decided to use non-parametric after doing Q-Q plots)
            print('\nKruskal-Wallis Test (regional GCM means):')
            subregions = subregion_df['subregion'].unique().tolist()
            subregion_lists = [
                subregion_df.loc[subregion_df['subregion'] == subregion, 'mean_norm_vol'].dropna().tolist()
                for subregion in subregions
            ]
            try:
                stat, p = kruskal(*subregion_lists)
                print(f'Kruskal-Wallis H = {stat:.4f}, p = {p:.4e}')
            except Exception as e:
                print(f'Kruskal-Wallis test failed: {e}')
                p = None
                
            # Conducts Dunn's test using Bonferroni correction for p values (if Kruskal-Wallis test returns a significant p)
            if p is not None and p < 0.05:
                print("Significant differences found so performing post hoc Dunn's test\n")
                dunns_results = sp.posthoc_dunn(subregion_df, val_col='mean_norm_vol', group_col='subregion', p_adjust='bonferroni')
                # Formats p values (note: this was just done to help with write up; had to explicity state object to avoid depreciation warnings)
                formatted_results = dunns_results.copy().astype('object')
                for row in formatted_results.index:
                    for col in formatted_results.columns:
                        val = formatted_results.loc[row, col]
                        if val < 0.001:
                            formatted_results.loc[row, col] = '<0.001'
                        else:
                            formatted_results.loc[row, col] = f'{val:.2g}'
                print("Dunn's Test results (corrected p-values):")
                print(formatted_results)

            else:
                print('No significant differences found between subregions.')

    # Proceeds to plot the total regional volume change if True
    if totalregion:

        fig = plt.figure(figsize=(10, 5))
        fig.text(0.01, 0.98, 'a', fontsize=16.5, fontweight='bold', va='top', ha='left')

        # Loops through SSPs and conducts normalisation 
        for ssp, colour in zip(ssp_list, ['#059105', '#115b8f', '#ff9d00', '#ff0000']):
            GCM_series_r = []
            for GCM, series in regional_GCM_vols[ssp].items():
                series = series.sort_index()
                norm_series = (series / series.loc[2025]) * 100
                GCM_series_r.append(norm_series)

            # Calculates the regional GCM ensemble mean and standard deviation 
            years = GCM_series_r[0].index.to_numpy()
            combined = np.stack([i.values for i in GCM_series_r])
            mean = np.nanmean(combined, axis=0)
            std = np.nanstd(combined, axis=0)

            index_2100 = np.where(years == 2100)[0][0]
            vol_2100 = mean[index_2100]
            std_2100 = std[index_2100]

            # Prints regional stats if True
            if statistics:
                clim_mask = (years >= 2071) & (years <= 2100)
                print(f'\nRegional, {ssp_labels[ssp]}:')
                print(f'2100 Volume: {vol_2100:.4f}% (SD = {std_2100:.4f}%)')
                print(f'2071–2100 Climatology: {np.nanmean(mean[clim_mask]):.4f}% (SD = {np.nanmean(std[clim_mask]):.4f}%)')

            # Plots mean and standard deviation 
            plt.plot(years, mean, label=ssp_labels[ssp], color=colour, linewidth = 1.7)
            plt.fill_between(years, mean - std, mean + std, color=colour, alpha=0.1)

            # Adds annotated volume percentage at 2100 for each SSP
            ax = plt.gca()  
            ax.annotate(
                f"{vol_2100:.0f} ± {std_2100:.0f}%",
                xy=(2100, vol_2100),
                xytext=(8, 0),
                textcoords='offset points',
                color=colour,
                fontsize=14,
                fontweight = 'bold',
                va='center',
                ha='left',
                clip_on=False  
            )

        # Plot formatting
        plt.axhline(100, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.xlabel('Year', fontsize=15.5, labelpad = 12)
        plt.ylabel('Volume (% of 2025)', fontsize=15.5, labelpad = 10)
        plt.tick_params(axis='both', labelsize=13.5)
        plt.xlim(2025, 2100)
        plt.ylim(0, 110)
        plt.grid(False)
        plt.legend(loc='lower left', frameon=True, fontsize=13, facecolor = 'white', edgecolor = 'None', framealpha = 0.7)
        plt.tight_layout()
        plt.savefig('custom_filepath/RegionalVol.png', dpi=300)
        plt.show()

def process_plot_GCMs(GCMs, stats = False):

    """
    Loads, processes, and plots simulated glacier volume change for each GCM

    Arguments:
        GCMs (list): List of GCM names (strings) within output filenames

    Returns:
        None 

    Raises:
        None

    """

    # Defines SSP list and labels for plotting
    ssp_list = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    ssp_labels = {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5'
    }

    # Defines custom colours for plotting
    colours = [
        '#2c7da0', '#f0c808', '#d9b285', '#c00000', '#469374',
        '#e3427d', '#5d4b20', '#9341b3', '#f06900'
    ]
    gcm_colours = {GCM: colours[i % len(colours)] for i, GCM in enumerate(GCMs)}

    # Creates dictionary containing properly formatted GCM names (note: required because of how output files were saved with _ not -)
    legend_labels = {
    'BCC_CSM2_MR': 'BCC-CSM2-MR',
    'CAMS_CSM1_0': 'CAMS-CSM1-0',
    'CESM2': 'CESM2',
    'EC_Earth3_Veg': 'EC-Earth3-Veg',
    'FGOALS_f3_L': 'FGOALS-f3-L',
    'GFDL_ESM4': 'GFDL-ESM4',
    'MPI_ESM1_2_HR': 'MPI-ESM1-2-HR',
    'MRI_ESM2_0': 'MRI-ESM2-0',
    'NorESM2_MM': 'NorESM2_MM'
    }

    # Creates subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    axs = axs.flatten()

    # Loops through SSPs
    for i, ssp in enumerate(ssp_list):
        ax = axs[i]
        combined_rel_vols = []

        # Loops through GCMs, forming filepath to data
        for GCM in GCMs:
            filepath = f'custom_filepath/{GCM}_Output.csv'

            # Reads in GCM output data and pivots for calculations (reads with Polars for speed then converts to pandas for plotting)
            df = pl.read_csv(filepath)
            data = df.select(['rgi_id', 'calendar_year', f'volume_{ssp}']).to_pandas()
            pivot_data = data.pivot(index='calendar_year', columns='rgi_id', values=f'volume_{ssp}')

            # Calculates total volume, defines baseline as 2025, and calculates relative volume change
            total_vol = pivot_data.sum(axis=1)
            baseline = total_vol.loc[2025]
            rel_vol = (total_vol / baseline) * 100
            combined_rel_vols.append(rel_vol)

            # Prints statistics if Stats is true (note: printing gets pretty messy here in the loop but I just used for writeup)
            if stats:
                end_vol = rel_vol.loc[2100]
                print(f'{GCM} - {ssp_labels[ssp]} (2100): {end_vol:.2f}%')

            # Plots individual volume change 
            ax.plot(rel_vol.index, rel_vol.values, color=gcm_colours[GCM], linewidth=2.3, label=legend_labels.get(GCM, GCM), alpha=0.9)

        # Plots the ensemble mean calculated for all processed GCMs
        if combined_rel_vols:
            all_df = np.stack([i.values for i in combined_rel_vols])
            mean_volume = np.nanmean(all_df, axis=0)
            years = combined_rel_vols[0].index
            ax.plot(years, mean_volume, color='black', linewidth=3.4, label='Mean', zorder=10)

        # Plot formatting
        ax.set_title(f'{ssp_labels[ssp]}', fontsize=21.5)
        ax.axhline(100, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xlim(2025, 2100)
        ax.set_ylim(0, 110)
        ax.grid(axis='both', linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', labelsize=18.5)

        # Sets conditional axis labelling and legend
        if i in [2, 3]:
            ax.set_xlabel('Year', fontsize=22, labelpad = 12.5)
        if i in [0, 2]:
            ax.set_ylabel('Volume (% of 2025)', fontsize=22, labelpad = 12.5)
        if i == 0:  
            handles, labels = ax.get_legend_handles_labels()

            # Shifts mean item to top of legend 
            mean_idx = labels.index('Mean')
            handles = [handles[mean_idx]] + handles[:mean_idx] + handles[mean_idx+1:]
            labels = ['Ensemble mean'] + labels[:mean_idx] + labels[mean_idx+1:]

            # Adds legend
            ax.legend(
                handles,
                labels,
                loc='lower left',
                ncol = 2, 
                fontsize=16.5,
                frameon=False,
                handlelength=1.5,
                labelspacing=0.3
            )

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.2)
    plt.savefig('custom_filepath/GCMvols.png', dpi=300)
    plt.show()

def hypsometry(GCMs, regression = False):

    """ 
    Processes fl_diagnostics data stored in parquet files and calculates area and thickness hypsometry

    Arguments:
        GCMs (list): List of GCMs for processing 
        regression (bool): Plots area retention vs elevation with curve fit using logistic function)

    Returns: 
        None

    Raises: 
        None
    """
    
    # Creates dictionary to store SSP statistics and defines SSPs
    all_ssp_stats = {}
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']


    # Defines colour mapping and labelling
    colours = {
        'ssp126': '#115b8f',  
        'ssp245': '#059105',  
        'ssp370': '#ff9d00',  
        'ssp585': '#ff0000'   
    }
    ssp_labels = {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5'
    }

    # Loops through SSPs 
    for ssp in ssps:

        all_GCM_dfs = []

        #  Reads in parquet files (note: these are created by the processing conducted in ProcessHypsometry.py)
        for GCM in GCMs:
            parquet_path = f'custom_filepath/{GCM}_export_{ssp}.parquet'
            df = pl.read_parquet(parquet_path)

            # Cleans data, calculates weighted thickness change, groups data by bins, and then calculates area-weighted change
            df_clean = df.filter(pl.col('thickness_change_per_year').is_not_null())
            df_clean = df_clean.with_columns([
                (pl.col('thickness_change_per_year') * pl.col('area_initial')).alias('weighted_thickness')
            ])
            weighted_by_bin = (
                df_clean.group_by('bin_centre')
                .agg([
                    pl.col('area_initial').sum().alias('area_sum'),
                    pl.col('weighted_thickness').sum().alias('weighted_sum')
                ])
                .with_columns([
                    (pl.col('weighted_sum') / pl.col('area_sum')).alias('area_weighted_thickness')
                ])
            )

            # Calculates total initial and final area per ele bin
            area_df = (
                df.group_by('bin_centre')
                .agg([
                    pl.col('area_initial').sum().alias('area_initial'),
                    pl.col('area_final').sum().alias('area_final')
                ])
            )

            # Merges data and appends to list containing each GCM's data
            GCM_df = weighted_by_bin.join(area_df, on='bin_centre', how='left')
            all_GCM_dfs.append(GCM_df)

        # Stacks GCM dfs
        combined_df = pl.concat(all_GCM_dfs)

        # Calculates summary statistics for each SSP (was going to do some printing but didn't need in the end, just used this to help with formatting the stacking of bars)
        stats_df = (
            combined_df.group_by('bin_centre')
            .agg([
                pl.col('area_weighted_thickness').mean().alias('mean_thickness'),
                pl.col('area_weighted_thickness').std().alias('std_thickness'),
                pl.col('area_initial').mean().alias('mean_area_initial'),
                pl.col('area_final').mean().alias('mean_area_final')
            ])
            .sort('bin_centre')
        )
        all_ssp_stats[ssp] = stats_df

    # Prepares bins for plotting 
    bin_centres = all_ssp_stats[ssps[0]]['bin_centre'].to_numpy()
    bar_width = 100

    # Creates subplots and formats grid plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    # Plots initial  area (converted to square kilometres)
    initial_area = all_ssp_stats[ssps[0]]['mean_area_initial'].to_numpy() / 1e6
    ax1.bar(bin_centres, initial_area, width=bar_width, color= '#b3cde0' , alpha=1, label='Initial Area', edgecolor='black', linewidth=0.3)

    # Sorts by size to ensure that all bars are visible and plots final area under each SSP (converted to square kilometres)
    ordered_ssp_area = sorted(ssps, key=lambda ssp: all_ssp_stats[ssp]['mean_area_final'].sum(), reverse=True)
    for ssp in ordered_ssp_area:
        area_final = all_ssp_stats[ssp]['mean_area_final'].to_numpy() / 1e6
        ax1.bar(bin_centres, area_final, width=bar_width, color=colours[ssp], label=ssp_labels[ssp], edgecolor='black', linewidth=0.3)

    # Plot formatting 
    ax1.set_ylabel('Glacier Area (km²)', fontsize=20, labelpad = 11)
    ax1.legend(ncol=1, frameon=True, fontsize=16, loc='lower right',  edgecolor = 'none', facecolor = 'white')
    ax1.set_xlim(0, 6000)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color = 'black')
    ax1.yaxis.get_offset_text().set_fontsize(16)

    # Sorts by size and plots thickness change under each SSP
    ordered_ssp_thickness = sorted(ssps, key=lambda ssp: all_ssp_stats[ssp]['mean_thickness'].abs().mean(), reverse=True)
    for ssp in ordered_ssp_thickness:
        thickness_mean = all_ssp_stats[ssp]['mean_thickness'].to_numpy()
        ax2.bar(bin_centres, thickness_mean, width=bar_width, color=colours[ssp], alpha=1, label=ssp_labels[ssp], edgecolor='black', linewidth=0.3)

    # Plot formatting
    handles, labels = ax2.get_legend_handles_labels()
    ax2.axhline(0, color='black', lw=0.8, linestyle='-', alpha=0.8)
    ax2.set_ylabel('Area-Weighted Thickness Change (m yr$^{-1}$)', fontsize=20, labelpad = 11)
    ax2.set_xlabel('Elevation (m)', fontsize=20, labelpad = 11)
    ax2.legend(handles[::-1], labels[::-1], frameon=True, fontsize=14, edgecolor = 'none', facecolor = 'white')
    ax2.set_xlim(0, 6000)
    ax2.set_ylim(-2.5, 0.5)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3, color = 'black')
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(500))

    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    fig.text(-0.02, 0.98, 'a', fontsize=17, fontweight='bold', va='top', ha='left')
    fig.text(-0.02, 0.50, 'b', fontsize=17, fontweight='bold', va='top', ha='left')
    plt.savefig('custom_filepath/Hypsometry.png', dpi=300, bbox_inches = 'tight')
    plt.show()

    # Plots regression if True 
    # Note: Took a while to decide between this and my previous linear approach, but the logistic function clearly fits the data better 
    if regression:
        
        # Defines logistic function (with an offset that I set as minimum retention value)
        def logistic(x, L, k, x0, b):
            return b + L / (1 + np.exp(-k * (x - x0)))

        # Creates figure 
        plt.figure(figsize=(12, 8))

        # Loops through SSPs
        for ssp in ssps:
            stats_df = all_ssp_stats[ssp]
            
            # Calculates area retention per bin as a percentage
            stats_df = stats_df.with_columns([
                (pl.col('mean_area_final') / pl.col('mean_area_initial') * 100).alias('area_retention_percent')
            ])
            bin_centres = stats_df['bin_centre'].to_numpy()
            retention = stats_df['area_retention_percent'].to_numpy()
            
            # Plots points as scatter
            plt.scatter(bin_centres, retention, color=colours[ssp], alpha=0.8)
            
            # For curve, makes param guesses based upon data (note: played around with setting manually but auto was best)
            L = max(retention) - min(retention)
            k = 0.01 / np.std(bin_centres)
            x0 = bin_centres[np.argmax(np.gradient(retention))]
            b = min(retention)
            p0 = [L, k, x0, b]

            # Defines bounds derived automatically from data as well as general assumptions
            bounds = (
                [0, 0, bin_centres.min(), 0],
                [200, 1, bin_centres.max(), 100]
            )

            # Tries to use curve_fit to generate a logistic function for fitting the curve
            try:
                params, covariance = curve_fit(logistic, bin_centres, retention, p0=p0, bounds=bounds)
                L, k, x0, b = params
                x_vals = np.linspace(bin_centres.min(), bin_centres.max(), 200)
                y_vals = logistic(x_vals, L, k, x0, b)
                
                # Plots curve
                plt.plot(x_vals, y_vals, color=colours[ssp], linewidth=2,
                        label=f'{ssp_labels[ssp]}')

                # For write-up, identifies the nearest bin which is at the minimum elevation to retain 50% of its area and prints logistic function params
                bin_50 = x0 - (1 / k) * np.log((L / (50 - b)) - 1)
                nearest_bin = bin_centres[np.abs(bin_centres - bin_50).argmin()]
                print(f'{ssp_labels[ssp]}: L = {L:.2f}, k = {k:.4f}, x0 = {x0:.0f}, b = {b:.2f}\nMinimum elevation bin with 50% area retention {nearest_bin} m')

            # Note: only had errors at the start, when manually making param guesses but kept error checking in just in case
            except RuntimeError:
                print(f'Could not fit logistic curve for {ssp}')

        # Plot formatting
        plt.xlabel('Elevation (m)', fontsize=19, labelpad=12)
        plt.ylabel('Area Retention (%)', fontsize=19, labelpad=12)
        plt.xlim(0, 6100)
        plt.ylim(0, 120)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3, color='black')
        plt.legend(fontsize=16, frameon=True, loc='lower right', facecolor='white', edgecolor='none')
        ax = plt.gca()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(500))
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.gca().set_axisbelow(True)

        plt.tight_layout()
        plt.savefig('custom_filepath/HypsometryRegression.png', dpi=300)
        plt.show()


def SLE(GCMs, start_year, end_year, stats = False, shapefile = False):

    """
    Reads volume projection data, converts to SLE, provides stats, plots, and adds SLE contribution to shapefile

    Arguments:
        GCMs (list): List of GCMs (strings) for plotting
        start_year (int): Initial year for SLE period calculation 
        end_year (int): Final year for SLE period calculation 
        stats (bool): Prints summary stats if True
        shapefile (bool): Adds total SLE contribution of each glacier to RGI outline shapefile 

    Returns:
        None

    Raises:
        None


    """

    # Defines ssp columns for data reading and labels for plotting
    ssp_columns = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    colours = ['#059105', '#115b8f', '#ff9d00', '#ff0000']  
    ssp_labels = {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5'
    }

    # Defines constants for SLE conversion and start/end year for calculations (additionally adds starting volume for a subsequent calculation for writeup)
    ice_density = 0.9
    gt_per_mm_SLE = 361.8
    years = list(range(start_year, end_year))
    years_for_plot = [y + 1 for y in years]  
    reference_total = 45.25  

    # Loads csv containing RGI IDs prefiltered by subregion, conducts dictionary comprehension
    valids = pl.read_csv('regional_rgi_ids.csv')
    subregion_rgis = {subregion_col: valids.select(pl.col(subregion_col).drop_nulls().cast(str)).to_series().to_list()for subregion_col in valids.columns}

    # Stacks all RGI IDs for regional plotting
    stacked_ids = [rgi for rgi_list in subregion_rgis.values() for rgi in rgi_list]

    # Loads volume projection data 
    GCM_dfs = {}
    for GCM in GCMs:
        df = pl.read_csv(f'custom_filepath/{GCM}_Output.csv')
        df = df.with_columns(pl.col('rgi_id').cast(pl.Utf8))

        # Renames columns in line with those elsewhere since I was running into issues otherwise
        for ssp_col in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
            df = df.rename({f'volume_{ssp_col}': ssp_col})
        GCM_dfs[GCM] = df.filter(pl.col('rgi_id').is_in(stacked_ids))
        print(f'Loaded {GCM}')

    # Loops through SSPs 
    region_stats = {}
    for ssp_column in ssp_columns:
        print(f'Processing SSP: {ssp_labels[ssp_column]}')

        # Creates empty dictionaries, setting up nested structure (this structure worked when testing so kept as is, but it could probably be simplified)
        GCM_annual_SLEs = {GCM: {rgi: [] for rgi in stacked_ids} for GCM in GCMs}
        GCM_total_SLEs = {rgi: {} for rgi in stacked_ids}

        # Computes annual SLE  
        for GCM in GCMs:
            df = GCM_dfs[GCM]

            # For given GCM, calculates SLE annually (i.e. calculated as the contribution between year a and b based upon vol change, converted to mass change, between a and b)
            yearly_SLE = {}
            for year in years:
                year_a = df.filter(pl.col('calendar_year') == year).select(['rgi_id', ssp_column])
                year_b = df.filter(pl.col('calendar_year') == year + 1).select(['rgi_id', ssp_column])
                change = year_b.join(year_a, on='rgi_id', suffix='_a')
                change = change.with_columns(
                    ((pl.col(f'{ssp_column}_a') - pl.col(ssp_column)) * ice_density / 1e9 / gt_per_mm_SLE).alias('sle_mm')
                )
                yearly_SLE[year] = {rgi: SLE for rgi, SLE in change.select(['rgi_id', 'sle_mm']).rows()}

            # Stacks and saves yearly SLE
            for rgi in stacked_ids:
                annual_contribution = [yearly_SLE.get(year, {}).get(rgi, 0.0) for year in years]
                GCM_annual_SLEs[GCM][rgi] = annual_contribution

            # Sums total SLE per glacier for given GCM for the regional stats
            for rgi in stacked_ids:
                total_SLE = sum(yearly_SLE.get(year, {}).get(rgi, 0.0) for year in years)
                GCM_total_SLEs[rgi][GCM] = total_SLE

        # Calculates GCM-ensemble regional annual mean and SD and aggregates 
        region_mean = []
        region_sd = []
        for year_idx, year in enumerate(years):
            glacier_means = []
            glacier_sds = []
            for rgi in stacked_ids:
                sle_data = [GCM_annual_SLEs[GCM][rgi][year_idx] for GCM in GCMs]
                glacier_means.append(np.mean(sle_data))
                glacier_sds.append(np.std(sle_data))

            region_mean.append(sum(glacier_means))
            region_sd.append(np.sqrt(sum(np.array(glacier_sds) ** 2)))

        # Adds mean and SD to stats dict
        region_stats[ssp_column] = {
            'mean': np.array(region_mean),
            'sd': np.array(region_sd)
        }

        # Calculates mean and SD of total glacier SLE across GCMs for stats
        glacier_stats = {}
        for rgi in stacked_ids:
            sle_data = [GCM_total_SLEs[rgi].get(GCM, 0.0) for GCM in GCMs]
            glacier_stats[rgi] = {
                'mean': np.mean(sle_data),
                'sd': np.std(sle_data)
            }

        # Aggregates data by subregion (ensuring that RGI ID is present in the glacier_stats dict) and calculates stats, including uncertainty propagation
        subregion_SLE_stats = {}
        for subregion in subregion_rgis.keys():
            glacier_list = subregion_rgis[subregion]
            means = [glacier_stats[rgi]['mean'] for rgi in glacier_list if rgi in glacier_stats]
            sds = [glacier_stats[rgi]['sd'] for rgi in glacier_list if rgi in glacier_stats]
            subregion_mean = sum(means)
            subregion_sd = np.sqrt(sum(sd ** 2 for sd in sds))
            subregion_SLE_stats[subregion] = {'mean': subregion_mean, 'sd': subregion_sd}

        # Prints subregional and regional stats if stats
        if stats:
            print(f'\nSubregional total SLE stats for {ssp_column}:')
            for subregion, stats in subregion_SLE_stats.items():
                print(f'Subregion 0{subregion} total = {stats["mean"]:.2f} mm (SD = {stats["sd"]:.2f} mm)')

            total_mean = sum(stats['mean'] for stats in subregion_SLE_stats.values())
            total_sd = np.sqrt(sum(stats['sd'] ** 2 for stats in subregion_SLE_stats.values()))
            print(f'\nAlaska total SLE for {ssp_column}: {total_mean:.2f} mm (SD = {total_sd:.2f} mm)')

            print(f'\nAverage annual SLE - {ssp_column}:')
            for subregion, stats in subregion_SLE_stats.items():
                annual_mean = stats['mean'] / len(years)
                annual_sd = stats['sd'] / len(years)
                print(f'{subregion}: {annual_mean:.12f} mm/year (SD = {annual_sd:.12f} mm/year)')

            region_mean = total_mean / len(years)
            region_sd = total_sd / len(years)
            print(f'Alaska total: {region_mean:.12f} mm/year (SD = {region_sd:.12f} mm/year)')

            #Optional: Compares to reference value (i.e. full potential contribution of Alaska glaciers). Used this for write up
            percent_of_reference = (total_mean / reference_total) * 100
            percent_sd = (total_sd / reference_total) * 100 
            print(f'Alaska contribution is {percent_of_reference:.2f}% (SD = {percent_sd:.2f}% of the 45.25 mm potential)')

            # Optional: Bins glacier SLE (used for write up)
            glacier_totals = np.array([i['mean'] for i in glacier_stats.values()])
            bins = np.arange(0, 1.2, 0.1)
            counts, _ = np.histogram(glacier_totals, bins=bins)
            percentages = 100 * counts / len(glacier_totals)

            print(f'\nPercentage of glaciers by total SLE bins for {ssp_column}:')
            for i in range(len(counts)):
                print(f'{bins[i]:.1f}–{bins[i+1]:.1f} mm: {counts[i]} glaciers ({percentages[i]:.2f}%)')

            # Optional: Identifies the top three biggest single contributors (used for write up)
            single_big = sorted(glacier_stats.items(), key=lambda x: x[1]['mean'], reverse=True)[:3]

            print(f'\nTop three individual glacier SLR contributors for {ssp_column}:')
            for i, (rgi_id, data) in enumerate(single_big, 1):
                print(f'{i}. RGI ID: {rgi_id} — Total SLE = {data["mean"]:.3f} mm (SD = {data["sd"]:.3f} mm)')

        # If shapefile, proceeds to add SLE data to RGI glacier outlines shapefile (note: used this for SLE plots in QGIS)
        if shapefile:

            # Defines shapefile path (note: this is a path to the RGI 6.0 Alaska .shp)
            shapefile_path = 'custom_filepath/glacier_outlines.shp'

            # Shortens the column name to be compatible with shapefile
            col_name = f'{ssp_column}SLE'

            # Converts glacier_stats to DataFrame
            df_result = pd.DataFrame.from_dict(
                {rgi: data['mean'] for rgi, data in glacier_stats.items()},
                orient='index',
                columns=[col_name]  
            )
            df_result.index.name = 'rgi_id'
            df_result = df_result.reset_index()

            # Loads shapefile in the first SSP loop and retains in locals (note: this then allows me to append all four SSPs' data and export outside of the loop)
            if 'gdf' not in locals():
                gdf = gpd.read_file(shapefile_path)
                # Renames RGI ID column for consistency
                if 'rgi_id' not in gdf.columns and 'RGIId' in gdf.columns:
                    gdf['rgi_id'] = gdf['RGIId']

            # Removes existing column to enable overwriting and merges new SLE data (note: I did not do this at first and when testing ended up adding multiple extra duplicated columns)
            if col_name in gdf.columns:
                gdf.drop(columns=[col_name], inplace=True)
            gdf = gdf.merge(df_result, on='rgi_id', how='left')

    # Exports shapefile once looped through all SSPs
    if shapefile:
        output_file = shapefile_path.replace('.shp', '_SLE.shp')
        gdf.to_file(output_file)
        print('Shapefile updated with SLE')

    # Creates subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharey=True, sharex='col')
    axs = axs.flatten()

    # Loops through SSPs for plotting 
    for i, ssp_column in enumerate(ssp_columns):
        ax = axs[i]
        legend_lines = []

        # Loops through all four SSPs, making the current SSP bold but also plotting the other SSPs faded to improve the plot clarity
        # Note: It took a while to decide upon this format. Plotting all SSPs on a single plot was too messy, but plotting each SSP individually made comparison difficult.
        for j, focus_ssp in enumerate(ssp_columns):
            mean = np.array(region_stats[focus_ssp]['mean'])
            sd = np.array(region_stats[focus_ssp]['sd'])

            # Defines alphas and linewidth based upon whether the SSP is the focus of the subplot
            is_main = focus_ssp == ssp_column
            alpha_line = 1.0 if is_main else 0.1
            alpha_fill = 0.4 if is_main else 0.1
            line_width = 2.5 if is_main else 1.0

            # Plots
            ax.fill_between(years_for_plot, mean - sd, mean + sd, color=colours[j], alpha=alpha_fill)
            ax.plot(years_for_plot, mean, color=colours[j], linewidth=line_width, alpha=alpha_line)

            # Formats legend conditionally (note: based upon the SSP that is the focus of the subplot, I make all other SSPs more transparent)
            legend_alpha = 1.0 if is_main else 0.3
            legend_line, = ax.plot([], [], color=colours[j], linewidth=2.5, alpha=legend_alpha, label = ssp_labels[focus_ssp])
            legend_lines.append(legend_line)

        # Plot formatting
        ax.tick_params(axis='both', labelsize=16)
        ax.set_title(f'{ssp_labels[ssp_column]}', fontsize=19)
        if i % 2 == 0:
            ax.set_ylabel('SLE (mm/year)', fontsize=18, labelpad=12)
        ax.set_xlim(start_year, end_year)
        ax.set_ylim(0, 0.3)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        if i < 2:
            ax.tick_params(labelbottom=False) 
        else:
            ax.set_xlabel('Year', fontsize=18, labelpad=12)

        ax.legend(
            handles=legend_lines,
            loc='lower left',
            frameon=False,
            fontsize=15
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig('custom_filepath/SLE.png', dpi=300)
    plt.show()

def triple_individual_plot(GCMs, rgi_ids, stats = False):

    """
    Creates a 1x3 plot of individual glaciers - used for SLR results section
    Note: The use case of this is highly limited since it essentially replicates the plotting in process_plot but just for three glaciers, however I decided to 
    include it regardless for full transparency of the study's workflow

    Arguments:
        GCMs (list): List of GCM names (strings) within output filenames
        rgi_ids (list): List of RGI IDs (string) for plotting
        stats (Bool): If True prints summary statistics 

    Returns:
        None 

    Raises:
        ValueError: If too many RGI IDs are passed

    """

    # Ensures only three RGI IDs are passed 
    if len(rgi_ids) != 3:
        raise ValueError('Requires three RGI IDs')

    # Defines SSPs, their labels, and their colours for plotting (note: I also define the plot labels a-c, this is pretty crude but works)
    ssp_list = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    ssp_labels = {
        'ssp126': 'SSP1-2.6',
        'ssp245': 'SSP2-4.5',
        'ssp370': 'SSP3-7.0',
        'ssp585': 'SSP5-8.5'
    }
    colours = ['#059105', '#115b8f', '#ff9d00', '#ff0000']
    plot_labels = ['a','b','c']

    # Reads in output data for argued GCMs
    GCM_dfs = {
        GCM: pl.read_csv(f'custom_filepath/{GCM}_Output.csv').filter(pl.col('rgi_id').is_in(rgi_ids))
        for GCM in GCMs
    }
    # Creates subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 9), sharey=True)

    # Loops through RGI IDs
    for i, (ax, rgi_id) in enumerate(zip(axs, rgi_ids)):
        ax2 = ax.twinx()
        vol_norm_baselines = []

        # Loops through each SSP 
        for ssp, colour in zip(ssp_list, colours):
            vol_norm_tseries = []
            vol_raw_tseries = []
            vol_2100_list = []
            rel_2100_list = []

            # Loops through each GCM, filters data for each RGI ID, and converts to pandas
            for GCM in GCMs:
                df = GCM_dfs[GCM].filter(pl.col('rgi_id') == rgi_id)
                df_pd = df.select(['calendar_year', f'volume_{ssp}']).to_pandas()
                df_pd = df_pd.sort_values('calendar_year').set_index('calendar_year')

                # Converts volume from cubed metres to cubed kilometres and normalises it against 2025 baseline
                # Note this is done for both the normalised volume and the raw volume. I decided that the latter would provide additional context
                # to the size of Malaspina-Seward glacier. 
                vol_raw = df_pd[f'volume_{ssp}'] / 1e9
                vol_norm = (vol_raw / vol_raw.loc[2025]) * 100 
                vol_norm_tseries.append(vol_norm)
                vol_raw_tseries.append(vol_raw)
                vol_norm_baselines.append(vol_raw.loc[2025])
                vol_2100_list.append(vol_raw.loc[2100])
                rel_2100_list.append(vol_norm.loc[2100])

            # If stats is True, prints summary statistics
            if stats:
                mean_2100_km3 = np.mean(vol_2100_list)
                sd_2100_km3 = np.std(vol_2100_list)
                mean_2100_rel = np.mean(rel_2100_list)
                rel_change_2100 = 100-mean_2100_rel
                sd_2100_rel = np.std(rel_2100_list)

                print(f'{rgi_id} - {ssp_labels[ssp]} (2100):')
                print(f'Raw Volume:\nMean = {mean_2100_km3:.2f} km³ (SD = {sd_2100_km3:.2f} km³)')
                print(f'Relative Volume Change:\nMean = {rel_change_2100:.2f}% (SD = {sd_2100_rel:.2f}%)\n')

            # Calculates GCM-ensemble mean and SD
            years = vol_norm_tseries[0].index.to_numpy()
            rel_combined = np.stack([i.values for i in vol_norm_tseries])
            raw_combined = np.stack([i.values for i in vol_raw_tseries])
            rel_mean = np.nanmean(rel_combined, axis=0)
            rel_sd = np.nanstd(rel_combined, axis=0)

            # Plots relative volume change on primary axis with shaded SD
            ax.plot(years, rel_mean, label=ssp_labels[ssp], color=colour, linewidth=2.7)
            ax.fill_between(years, rel_mean - rel_sd, rel_mean + rel_sd, color=colour, alpha=0.12)

        # Plot formatting
        ax.set_title(f'{plot_labels[i]}) {rgi_id}', fontsize=23)
        ax.axhline(100, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlim(2025, 2100)
        ax.set_ylim(0, 120)
        ax.set_xlabel('Year', fontsize=25, labelpad=12)
        ax.tick_params(axis='both', labelsize=17.5)
        ax.tick_params(labelleft=True, left=True) 
        ax.xaxis.set_tick_params(pad=8)

        # Creates ticks for raw volume on the secondary y-axis to correspond with plotted relative volume change
        # Note: Had to play around with the formatting for the 3 glaciers used in the results section so the formatting is very
        # specific and would almost certaintly need adjusting for other glaciers. I tried auto scaling but I wasn't happy with the output
        if vol_norm_baselines:
            avg_2025 = np.mean(vol_norm_baselines)
            rel_ymin, rel_ymax = ax.get_ylim()
            km3_ymin = avg_2025 * (rel_ymin / 100)
            km3_ymax = avg_2025 * (rel_ymax / 100)
            tick_start = int(np.floor(km3_ymin / 100) * 100)
            tick_end = int(np.ceil(km3_ymax / 100) * 100)
            tick_step = 500 if i == 0 else 100
            raw_ticks = np.arange(tick_start, tick_end + tick_step, tick_step)
            ax2.set_ylim(km3_ymin, km3_ymax)
            ax2.set_yticks(raw_ticks)
            ax2.tick_params(axis='y', labelsize=17.5)
            if i == len(rgi_ids) - 1:
                ax2.set_ylabel('Raw Volume (km³)', fontsize=25, labelpad=14)

    axs[0].set_ylabel('Volume (% relative to 2025)', fontsize=25, labelpad=14)
    axs[0].legend(loc='lower left', frameon=False, fontsize=19)

    plt.tight_layout()
    plt.savefig('custom_filepath/TripleVols.png', dpi=300)
    plt.show()
