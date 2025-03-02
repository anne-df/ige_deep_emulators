# Useful libraries
import os
import xarray as xr
import numpy as np
from tunax import Grid, Trajectory

# preprocessing functions
def preprocess_and_generate_labels(file_paths, norm_factor=25.0):
    """
    Preprocess NetCDF files to create normalized input data and labels.
    
    Parameters:
        file_paths: list of str
            List of paths to NetCDF files.

    Returns:
        combined_data: np.ndarray
            Data frame of stacked normalized data, dimensions: (total_time-len(file_paths), 50, 5).
        labels: np.ndarray
            Data frame of labels, dimensions: (total_time-len(file_paths), 50).
        min_vals: np.array
            Vector of size 5 (minimum of each channel over time and space).
        max_vals: np.array
            Vector of size 5 (maximum of each channel over time and space).
    """
    all_data = []  # To store preprocessed data from all files
    all_labels = []  # To store labels from all files

    for file_path in file_paths:

        # Load NetCDF file
        data = xr.open_dataset(file_path)

        # Extract time dimension
        time_dim = data.dims['time']
        time_dim = time_dim - 1  # Remove last time step - no prediction available for the next state

        # Extract variables
        temperature = data['t'].values/norm_factor
        salinity = data['s'].values/norm_factor/2. # normalize salinity
        u_velocity = data['u'].values
        v_velocity = data['v'].values
        # we only normalize temperature and salinity as the other variables are already in the unit circle

        # Create a 5th channel containing the varying parameters (strat_t, u, c1) in the first positions
        custom_channel = np.zeros((time_dim,))
        custom_channel[0] = data.attrs.get('strat_t', 0.0)
        custom_channel[1] = data.attrs.get('u', 0.0)
        custom_channel[2] = data.attrs.get('c1', 0.0)/norm_factor # only parameter that needs to be normalized
        

        # Merge as a full matrix (time_dim, 5, depth)
        processed_data = np.stack([
            temperature[:time_dim, :],
            salinity[:time_dim, :],
            u_velocity[:time_dim, :],
            v_velocity[:time_dim, :],
            np.tile(custom_channel, (len(temperature[0]), 1)).T
        ], axis=1)
        

        # Compute labels (temperature delta between t and t+1)
        labels = temperature[1:] - temperature[:-1]
        
        # Check that all values in processed_data and labels are in the unit circle
        assert np.all(np.abs(processed_data) <= 1.0), "All values in processed_data should be in the unit circle"
        assert np.all(np.abs(labels) <= 1.0), "All label values should be in the unit circle"
        
        # Store results
        all_data.append(processed_data)
        all_labels.append(labels)

    # Concatenate all matrices
    combined_data = np.vstack(all_data)
    combined_labels = np.concatenate(all_labels)

    return combined_data, combined_labels



# postprocessing functions
def traj_from_ds(ds: xr.Dataset, grid: Grid, different_temperature=None) -> Trajectory:
    """
    Creates a Trajectory from an xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the trajectory data. Must have variables `u`, `v`, `t`, `s`,
        and coordinates `time`, `zr`, `zw`.
    grid : Grid
        The Grid object used to define the geometry of the trajectory.

    Returns
    -------
    trajectory : Trajectory
        The reconstructed Trajectory object.
    """
    # Ensure the dataset contains all required variables and coordinates
    required_variables = ['u', 'v', 't', 's']
    required_coords = ['time', 'zr', 'zw']

    for var in required_variables:
        if var not in ds:
            raise ValueError(f"Dataset is missing required variable '{var}'.")
    for coord in required_coords:
        if coord not in ds.coords:
            raise ValueError(f"Dataset is missing required coordinate '{coord}'.")

    # Extract data from the dataset
    time = ds['time'].values
    u = ds['u'].values  # Shape: (nt, nz) i.e. (360, 50)
    v = ds['v'].values  # Shape: (nt, nz)
    t = ds['t'].values  # Shape: (nt, nz)
    s = ds['s'].values  # Shape: (nt, nz)
    
    if different_temperature is not None:
        t = different_temperature

    # Construct and return the Trajectory object
    return Trajectory(
        grid=grid,
        time=time,
        u=u,
        v=v,
        t=t,
        s=s,
    )