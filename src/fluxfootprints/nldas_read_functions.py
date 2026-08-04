import requests
import io
import pandas as pd
import numpy as np


def call_nldas_time_series(lat,lon,time_start,time_end,data,token):
    """
    INPUTS:
    lat - latitude
    lon - longitude
    time_start - start of time series in YYYY-MM-DDThh:mm:ss format (UTC)
    end_time - end of the time series in YYYY-MM-DDThh:mm:ss format (UTC)
    data - name of the data parameter for the time series
    token - earthaccess bearer token
    
    OUTPUT:
    time series csv output string

    Function modified from How_to_Access_GiC_Time_Series_Service.ipynb
    https://github.com/nasa/gesdisc-tutorials/tree/main/notebooks
    """
    time_series_url = "https://api.giovanni.earthdata.nasa.gov/timeseries"

    query_parameters = {
        "data":data,
        "location":"[{},{}]".format(lat,lon),
        "time":"{}/{}".format(time_start,time_end)
    }
   
    headers = {
    'Authorization': f'Bearer {token}'
    }
    
    response=requests.get(time_series_url, params=query_parameters, headers=headers)
    print("Status Code:", response.status_code)
    return response.text


def parse_nldas_csv(ts):
    """
    INPUTS:
    ts - time series output of the time series service
    
    OUTPUTS:
    headers,df - the headers from the CSV as a dict and the values in a pandas dataframe
    Function modified from How_to_Access_GiC_Time_Series_Service.ipynb
    https://github.com/nasa/gesdisc-tutorials/tree/main/notebooks
    """
    with io.StringIO(ts) as f:
        # the first 13 rows are header
        headers = {}
        try:
            for i in range(13):
                line = f.readline()
                key,value = line.split(",")
                headers[key] = value.strip()
        except ValueError as e:
            raise ValueError(
                "The returned CSV is empty.\n"
                "Please ensure that your subsetting bounds are within the extent of your dataset\n"
                "or that your permissions are set up correctly"
            ) from e

        # Read the csv proper
        df = pd.read_csv(
            f,
            header=1,
            names=("Timestamp",headers["param_name"]),
            converters={"Timestamp":pd.Timestamp}
        )

    return headers, df


def fetch_nldas_forcing_dataset(lat, lon, time_start, time_end, token):
    """Fetches all NLDAS forcing variables required for ASCE ET_o calculation

    and combines them into a single pandas DataFrame.
    """
    forcing_vars = {
        "temp_K": "NLDAS_FORA0125_H_2_0_Tair",
        "spec_hum": "NLDAS_FORA0125_H_2_0_Qair",
        "pressure_pa": "NLDAS_FORA0125_H_2_0_PSurf",
        "wind_u10": "NLDAS_FORA0125_H_2_0_Wind_E",
        "wind_v10": "NLDAS_FORA0125_H_2_0_Wind_N",
        "solar_rad": "NLDAS_FORA0125_H_2_0_SWdown",
    }

    dfs = []

    for name, param_id in forcing_vars.items():
        print(f"Fetching {name}...")
        csv_str = call_nldas_time_series(
            lat, lon, time_start, time_end, param_id, token
        )
        headers, df = parse_nldas_csv(csv_str)

        df = df.rename(columns={headers["param_name"]: name})
        df = df.set_index("Timestamp")
        dfs.append(df)

    # Merge all variables along the timestamp index
    combined_df = pd.concat(dfs, axis=1)
    return combined_df