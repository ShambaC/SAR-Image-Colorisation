import pandas as pd
import pycountry
from tqdm import tqdm

def get_iso_code_fuzzy(country_name: str, iso_format: str = "alpha_3") -> str:
    try:
        match = pycountry.countries.search_fuzzy(country_name)[0]
        return getattr(match, iso_format)
    except (LookupError, AttributeError, IndexError):
        return None

def replace_country_names_with_iso(input_path: str, output_path: str, iso_format: str = "alpha_3"):
    df = pd.read_parquet(input_path)

    tqdm.pandas(desc="Converting country names to ISO codes (pycountry fuzzy)")
    # df["country_iso"] = df["country"].progress_apply(lambda x: get_iso_code_fuzzy(x, iso_format))
    df["country_iso"] = df["country"].progress_map(lambda x: get_iso_code_fuzzy(x, iso_format))

    # Show unresolved names
    missing = df[df["country_iso"].isnull()]["country"].unique()
    if len(missing) > 0:
        print("Could not resolve ISO code for the following countries:")
        for m in missing:
            print(f"- {m}")

    # Drop or keep missing rows based on your needs
    df_clean = df.dropna(subset=["country_iso"]).drop(columns=["country"])
    df_clean.rename(columns={"country_iso": "country"}, inplace=True)

    df_clean.to_parquet(output_path, index=False)
    print(f"✅ Saved file with ISO codes to: {output_path}")

if __name__ == "__main__":
    replace_country_names_with_iso(
        input_path="../data/country_daily_avg.parquet",
        output_path="../data/country_daily_avg_iso.parquet",
        iso_format="alpha_2"
    )