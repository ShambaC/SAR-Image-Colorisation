#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Value

# --- Hardcoded Configuration ---
# The name of the dataset repository on the Hugging Face Hub.
DS_NAME = "Uniform-Sentinel-1-2-Dataset"
# The local directory containing the dataset files (CSVs and images).
DATA_ROOT = "./Dataset"
# ---

def generate_examples(df: pd.DataFrame):
    """Generator function that yields examples one by one from the DataFrame."""
    def fn():
        # Iterate over each row in the DataFrame
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="Generating examples"):
            # Clean the relative paths by replacing backslashes with forward slashes
            s1_relative_path = row.s1_fileName.replace('\\', '/')
            s2_relative_path = row.s2_fileName.replace('\\', '/')
            
            # Construct the full paths for the input and output images
            s1_path = os.path.join(DATA_ROOT, s1_relative_path)
            s2_path = os.path.join(DATA_ROOT, s2_relative_path)

            # Create the formatted prompt string from region and season
            prompt = f"Colorize the image, Region: {row.region}, Season: {row.season}"

            # Yield the example as a dictionary
            yield {
                "input_image": {"path": s1_path},
                "prompt": prompt,
                "output_image": {"path": s2_path},
            }
    return fn

def main():
    """Main function to load data, create, and push the dataset."""
    # Find all CSV files matching the pattern 'data_r_XXX.csv'
    csv_files = glob.glob(os.path.join(DATA_ROOT, "data_r_*.csv"))

    if not csv_files:
        print(f"❌ Error: No CSV files found in '{DATA_ROOT}' directory with the pattern 'data_r_*.csv'.")
        return

    # Read and concatenate all found CSV files into a single pandas DataFrame
    print(f"Found {len(csv_files)} CSV files. Loading data...")
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # Check for required columns
    required_columns = ['s1_fileName', 's2_fileName', 'region', 'season']
    if not all(col in df.columns for col in required_columns):
        print(f"❌ Error: CSV files must contain the following columns: {required_columns}")
        return

    # Create the generator function with the loaded data
    generation_fn = generate_examples(df)

    print("Creating Hugging Face dataset...")
    # Create the dataset from the generator
    ds = Dataset.from_generator(
        generation_fn,
        features=Features(
            input_image=ImageFeature(),
            prompt=Value("string"),
            output_image=ImageFeature(),
        ),
    )

    print(f"Pushing dataset '{DS_NAME}' to the Hub... 🚀")
    # Push the created dataset to the Hugging Face Hub
    ds.push_to_hub(DS_NAME)
    print("✅ Done!")

if __name__ == "__main__":
    main()
