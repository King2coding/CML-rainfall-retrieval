#!/usr/bin/env python3
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import requests
import xarray as xr


TOKEN = os.environ.get("CHECK_RAINFALL_ARTIFACT_TOKEN")
DOWNLOAD_URL = "https://cml.tahmo.org/api/v1/rainfall-maps/188e7f4a-2feb-430c-9e27-734f4bb77b93/download/"
OUTPUT_PATH = Path("/home/kkumah/Projects/cml-stuff/plots/rainfall_artifact_202603300530.nc")
IMAGE_PATH = Path("/home/kkumah/Projects/cml-stuff/plots/rainfall_artifact_preview_202603300530.png")


def download_file():
    headers = {
        "Authorization": f"Bearer {TOKEN}",
    }

    with requests.get(DOWNLOAD_URL, headers=headers, stream=True, timeout=300) as response:
        response.raise_for_status()
        with OUTPUT_PATH.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)

    print(f"Downloaded: {OUTPUT_PATH.resolve()}")
    print(f"Size: {OUTPUT_PATH.stat().st_size} bytes")


def inspect_netcdf():
    with xr.open_dataset(OUTPUT_PATH) as dataset:
        print("\nNetCDF opened successfully.")
        print(f"Variables: {list(dataset.data_vars)}")
        print(f"Coordinates: {list(dataset.coords)}")
        print(f"Dimensions: {dict(dataset.sizes)}")

        if dataset.attrs:
            print("\nGlobal attributes:")
            for key, value in dataset.attrs.items():
                print(f"  {key}: {value}")

        print("\nDataset summary:")
        print(dataset)


def pick_rainfall_variable(dataset):
    preferred_names = [
        "rainfall",
        "rain_rate",
        "R",
        "R_da",
        "R_mm_per_h",
        "rainfall_rate",
    ]

    for name in preferred_names:
        if name in dataset.data_vars:
            return name

    for name in dataset.data_vars:
        lowered = name.lower()
        if "rain" in lowered or lowered.startswith("r"):
            return name

    return next(iter(dataset.data_vars))


def generate_image():
    with xr.open_dataset(OUTPUT_PATH) as dataset:
        if not dataset.data_vars:
            raise ValueError("No data variables were found in the NetCDF file.")

        variable_name = pick_rainfall_variable(dataset)
        data_array = dataset[variable_name]

        if "time" in data_array.dims:
            selected_time = data_array["time"].isel(time=0).item()
            data_array = data_array.isel(time=0)
        else:
            selected_time = None

        data_array = data_array.squeeze(drop=True)

        if data_array.ndim != 2:
            raise ValueError(
                f"Cannot generate a 2D image from variable '{variable_name}' with shape {data_array.shape}."
            )

        figure, axis = plt.subplots(figsize=(10, 6))
        data_array.plot(ax=axis, cmap="Spectral_r", vmax=15, add_colorbar=True)
        title = f"{variable_name} rainfall map"
        if selected_time is not None:
            title = f"{title} | {selected_time}"
        axis.set_title(title)
        axis.set_xlabel("lon" if "lon" in data_array.coords else data_array.dims[-1])
        axis.set_ylabel("lat" if "lat" in data_array.coords else data_array.dims[0])
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(IMAGE_PATH, dpi=200, bbox_inches="tight")
        plt.close(figure)

    print(f"\nImage saved: {IMAGE_PATH.resolve()}")


def main():
    if "PASTE_BEARER_TOKEN_HERE" in TOKEN or "YOUR_DOMAIN" in DOWNLOAD_URL:
        raise ValueError("Update TOKEN and DOWNLOAD_URL at the top of the script before running it.")

    download_file()
    inspect_netcdf()
    generate_image()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)



#%%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
rd_dat = xr.open_dataset('/home/kkumah/Projects/cml-stuff/plots/rainfall_artifact_202603300530.nc')


# Create figure with Cartopy projection
fig, ax = plt.subplots(
    1, 1,
    figsize=(10, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# Plot rainfall (make sure data has lon/lat coords)
rd_dat['R_mm_per_h'].plot(
    ax=ax,
    transform=ccrs.PlateCarree(),   # IMPORTANT
    cmap='Spectral_r',
    vmax=15,
    cbar_kwargs={'shrink': 0.7}
)

# Plot link points
ax.scatter(
    rd_dat['link_lon'],
    rd_dat['link_lat'],
    color='black',
    s=10,
    transform=ccrs.PlateCarree()
)

# Add country borders and coastlines
ax.add_feature(cfeature.BORDERS, linewidth=1)
ax.add_feature(cfeature.COASTLINE, linewidth=1)

# Optional: add land and ocean for context
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='white')

# Optional: gridlines
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

plt.title(f"Rainfall Map with CML Links_{rd_dat['time'].values if 'time' in rd_dat.coords else ''}")
plt.show()
