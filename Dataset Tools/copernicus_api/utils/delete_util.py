import pandas as pd
import os

from tqdm import tqdm

folder_name = "r_008"
df = pd.read_csv(f"../Images/data_{folder_name}.csv")

# To be edited
delete_list = [1, 7, 9, 10, 11]

delete_file_names = [f"{folder_name}/s1_{folder_name[2:]}/img_p{mark}.png" for mark in delete_list]
df = df[~df["s1_fileName"].isin(delete_file_names)]
df.to_csv(f"../Images/data_{folder_name}.csv", index=False)

for mark in tqdm(delete_list) :
    s1_file = f"../Images/{folder_name}/s1_{folder_name[2:]}/img_p{mark}.png"
    s2_file = f"../Images/{folder_name}/s2_{folder_name[2:]}/img_p{mark}.png"


    if os.path.exists(s1_file) :
        os.remove(s1_file)
    if os.path.exists(s2_file) :
        os.remove(s2_file)

print("Finish processing")