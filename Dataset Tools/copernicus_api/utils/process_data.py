from pathlib import Path
from tqdm import tqdm

import pandas as pd
import cv2 as cv

import os

total_list = os.listdir("../Images")
filtered_files = [x for x in total_list if x.endswith(".csv")]

for csv_file in tqdm(filtered_files) :

    df = pd.read_csv(f"../Images/{csv_file}")
    data_list = []

    outputFolder_s1 = Path(df.iloc[0].s1_fileName).parent.__str__()
    Path(f"../../Dataset/{outputFolder_s1}").mkdir(parents=True, exist_ok=True)
    outputFolder_s2 = Path(df.iloc[0].s2_fileName).parent.__str__()
    Path(f"../../Dataset/{outputFolder_s2}").mkdir(parents=True, exist_ok=True)

    k = 0
    for row in tqdm(df.itertuples(), total=df.shape[0], leave=False) :

        idx = row.Index

        s1_image = cv.imread(f"Images/{row.s1_fileName}")
        s2_image = cv.imread(f"Images/{row.s2_fileName}")

        for i in tqdm(range(81), leave=False) :

            k += 1

            x = i % 9   # Move columns
            y = i // 9   # Move rows

            crop_s1 = s1_image[256*x : 256*(x+1), 256*y : 256*(y+1)]
            crop_s2 = s2_image[256*x : 256*(x+1), 256*y : 256*(y+1)]

            cv.imwrite(f"../../Dataset/{outputFolder_s1}/img_p{k}.png", crop_s1)
            cv.imwrite(f"../../Dataset/{outputFolder_s2}/img_p{k}.png", crop_s2)

            data_list.append([
                f"{outputFolder_s1}/img_p{k}.png",
                f"{outputFolder_s2}/img_p{k}.png",
                row["coordinates"],
                row["country"],
                row["date-time"],
                row["scale"],
                row["region"],
                row["season"],
                row["operational-mode"],
                row["polarisation"],
                row["bands"]
                ])
            
    data_df = pd.DataFrame(data_list, columns=["s1_fileName", "s2_fileName", 'coordinates', 'country', 'date-time', 'scale', 'region', 'season', 'operational-mode', 'polarisation', 'bands'])
    data_df.to_csv(f"../../Dataset/{csv_file}", index=False)

print("DONE")