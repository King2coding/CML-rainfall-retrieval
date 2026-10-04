import os
import cdsapi

# ----------------------------
# Output directory
# ----------------------------
out_dir = "/home/kkumah/Projects/cml-stuff/satellite_data/era5"
os.makedirs(out_dir, exist_ok=True)

# Output filename
out_file = os.path.join(
    out_dir,
    "ERA5_total_precipitation_2025_09_12_Ghana.nc"
)

# ----------------------------
# CDS request
# ----------------------------
dataset = "reanalysis-era5-single-levels"

request = {
    "product_type": "reanalysis",
    "variable": "total_precipitation",
    "year": "2025",
    "month": ["09", "10", "11", "12"],
    "day": [
        "01","02","03","04","05","06","07","08","09","10",
        "11","12","13","14","15","16","17","18","19","20",
        "21","22","23","24","25","26","27","28","29","30","31"
    ],
    "time": [
        "00:00","01:00","02:00","03:00","04:00","05:00",
        "06:00","07:00","08:00","09:00","10:00","11:00",
        "12:00","13:00","14:00","15:00","16:00","17:00",
        "18:00","19:00","20:00","21:00","22:00","23:00"
    ],
    "data_format": "netcdf",
    "area": [11.5, -4, 4.5, 1.5]  # [N, W, S, E]
}

# ----------------------------
# Download
# ----------------------------
client = cdsapi.Client()

client.retrieve(
    dataset,
    request,
    out_file
)

print(f"Download completed:\n{out_file}")
