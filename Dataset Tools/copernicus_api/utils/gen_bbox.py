import pandas as pd

from utils.calc_coords import getCoords
from tqdm import tqdm, trange

global_df = pd.read_csv("../data/global_points.csv")

for row in tqdm(global_df.itertuples(), total=global_df.shape[0]) :

    idx: int = row.Index

    file_name = f'r_{idx:03}'

    START_LATITUDE: float = row.Latitude
    START_LONGITUDE: float = row.Longitude

    current_lat = START_LATITUDE
    current_long = START_LONGITUDE

    major_points_df = {"Longitude": [current_long], "Latitude": [current_lat]}

    # Generate major coords, ie. coords for larger images
    for i in trange(4, leave=False) :
        for j in trange(3, leave=False) :

            if i == 0 and j == 0:
                continue

            next_set, _ = getCoords(current_long, current_lat, 23.04)
            next_long = next_set[2][0]

            major_points_df["Longitude"].append(next_long)
            major_points_df["Latitude"].append(current_lat)

            current_long = next_long

        current_long = START_LONGITUDE
        next_set, _ = getCoords(current_long, current_lat, 23.04)
        current_lat = next_set[0][1]

    points_df = pd.DataFrame(major_points_df)
    points_df.to_csv(f"../data/regions/{file_name}.csv", index=False)

print("\nDone generating coords list")