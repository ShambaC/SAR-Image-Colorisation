from utils.authclient import getOAuth
from utils.calc_coords import getCoords
from tqdm import tqdm
from io import TextIOWrapper
from requests_oauthlib import OAuth2Session

import pandas as pd

SEN2_COLLECTION_ID = 'byoc-5460de54-082e-473a-b6ea-d5cbe3c17cca'

def saveImage(oauth: OAuth2Session, long: float, lat: float, idx: int, log_file: TextIOWrapper, fromDateTime: str, toDateTime: str, folder: str) -> tuple[str] :
    INITIAL_LONGITUDE = long
    INITIAL_LATITUDE = lat

    coords_data = getCoords(INITIAL_LONGITUDE, INITIAL_LATITUDE, 23.04)

    # Earth engine format
    # Coords are in (longitude, latitude) format
    boxCoords = coords_data[0]

    ########################################################################
    #                       IMAGE PROPERTIES                               #
    ########################################################################

    img_height = 2304
    img_width = 2304

    bbox = coords_data[1]

    evalScript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B02", "B03", "B04"],
            output: { bands: 3 }
        };
    }

    function evaluatePixel(sample) {
        return [2.5 * sample.B04/10000, 2.5 * sample.B03/10000, 2.5 * sample.B02/10000];
    }

    """

    ########################################################################
    #                    LOCATION AND SEASON DATA                          #
    ########################################################################

    from pathlib import Path

    region = ""

    Path(f"../Images/{folder}/s2_{folder[2:]}").mkdir(parents=True, exist_ok=True)

    if INITIAL_LATITUDE <= 23.5 and INITIAL_LATITUDE >= -23.5 :
        region = "tropical"
    elif (INITIAL_LATITUDE > 23.5 and INITIAL_LATITUDE < 66.5) or (INITIAL_LATITUDE < -23.5 and INITIAL_LATITUDE > -66.5) :
        region = "temperate"
    else :
        region = "arctic"

    fileName = f"../Images/{folder}/s2_{folder[2:]}/img_p{idx}.png"

    ########################################################################
    #                              REQUEST                                 #
    ########################################################################

    request = {
        "input": {
            "bounds": {
                "bbox": bbox
            },
            "data": [
                {
                    "dataFilter": {
                        "timerange": {
                            "from": fromDateTime,
                            "to": toDateTime
                        }
                    },
                    "type": SEN2_COLLECTION_ID
                }
            ]
        },
        "output": {
            "width": img_width,
            "height": img_height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/png"
                    }
                }
            ]
        },
        "evalscript": evalScript
    }

    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    response = oauth.post(url, json=request)

    if response.ok :
        log_file.write("Response OK\n")
        with open(fileName, 'wb') as fp :
            fp.write(response.content)

        log_file.write("Done saving file\n\n")
        return (fileName, region)
    else :
        log_file.write(f"Response code: {response.status_code}\n")
        log_file.write(f"{response.content}\n\n")
        return ("error", "error")


if __name__ == "__main__" :
    import datetime
    import random
    from time import time

    csv_file_name = "r_000"

    # Iterate through rows and download images
    df = pd.read_csv(f"../data/regions/{csv_file_name}.csv")

    log_file = open("LOG.txt", 'a')
    log_file.write(f"LOGS FOR: {datetime.datetime.now()}\n")

    oauth, token = getOAuth()

    DateTupleList = [
        ("2023-09-29T23:59:59Z", "2023-10-30T00:00:00Z"),
        ("2023-01-29T23:59:59Z", "2023-02-27T00:00:00Z"),
        ("2023-06-29T23:59:59Z", "2023-07-30T00:00:00Z"),
        ("2023-11-29T23:59:59Z", "2023-12-30T00:00:00Z")
    ]

    randomDateTimeTuple = random.choice(DateTupleList)
    fromDateTime = randomDateTimeTuple[0]
    toDateTime = randomDateTimeTuple[1]

    for row in tqdm(df.itertuples(), total=df.shape[0]) :

        idx = row.Index

        # Refresh token
        if time() > (token["expires_at"] - 20) :
            oauth, token = getOAuth()

        long = row.Longitude
        lat = row.Latitude

        fileName, region = saveImage(oauth, long, lat, idx, log_file, fromDateTime, toDateTime, csv_file_name)
        
        if fileName.strip().lower() == "error" :
            continue

    log_file.close()