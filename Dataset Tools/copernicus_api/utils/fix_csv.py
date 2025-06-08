import pandas as pd

from tqdm import tqdm
from pathlib import Path

import os

total_list = os.listdir("../Images")
filtered_files = [x for x in total_list if x.endswith(".csv")]

# Remove indexing
for csv_file in tqdm(filtered_files) :

    df = pd.read_csv(f"../Images/{csv_file}")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df.to_csv(f"../Images/{csv_file}", index=False)

# Fix files
for csv_file in tqdm(filtered_files) :

    df = pd.read_csv(f"../Images/{csv_file}")

    if df.shape[0] == 0 :
        continue

    outputFolder_s1 = Path(df.iloc[0].s1_fileName).parent.__str__()

    image_list = os.listdir(f"../Images/{outputFolder_s1}")
    full_image_list = [f"{csv_file[5:-4]}/s1_{csv_file[7:10]}/{x}" for x in image_list]

    df = df[df["s1_fileName"].isin(full_image_list)]
    df.to_csv(f"../Images/{csv_file}", index=False)