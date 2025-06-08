import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/global_points.csv") 

import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS)
ax.scatter(df["Longitude"], df["Latitude"], s=10, c='red', alpha=0.7, transform=ccrs.PlateCarree())
plt.title("Points on World Map")
plt.show()