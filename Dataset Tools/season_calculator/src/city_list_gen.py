# Imports
import polars as pl
import json

from tqdm import tqdm

# Read the csv
df = pl.read_csv("../data/worldcities.csv", schema_overrides= { "population": pl.Float64 })

# Dictionary to hold cities from a country
res_dict = {}

# Iterate through countries in the dataset
for country, cities in tqdm(df.group_by("country"), total=242) :
    country_name = country[0]
    if country_name not in res_dict :
        res_dict[country_name] = {}

    for city_name, city_details in tqdm(cities.group_by("city_ascii"), leave=False) :
        city_ascii = city_name[0]
        if len(res_dict[country_name]) > 10 :
            continue

        res_dict[country_name][city_ascii] = [city_details.item(0, "lat"), city_details.item(0, "lng")]

print("Finished processing dataset")
print(f"Data count: {len(res_dict)}")

with open("../data/city_coords.json", "w") as json_file :
    json.dump(res_dict, json_file, indent=4)

print("Done saving result")