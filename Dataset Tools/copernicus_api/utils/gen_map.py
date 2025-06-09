import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

total_list = os.listdir(f"../../../Dataset")
csv_list = [x for x in total_list if x.endswith(".csv")]

lon_list = []
lat_list = []

for csv_file in tqdm(csv_list) :
    df = pd.read_csv(f"../../../Dataset/{csv_file}")

    if df.shape[0] == 0 :
        continue

    coords = df.iloc[0]["coordinates"][1:-1].split(",")
    lon = float(coords[0])
    lat = float(coords[1])

    lon_list.append(lon)
    lat_list.append(lat)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.scatter(lon_list, lat_list, s=10, c='red', alpha=0.7, transform=ccrs.PlateCarree())
plt.title("Points on World Map")
plt.show()