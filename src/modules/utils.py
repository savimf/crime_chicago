import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import params_cfg as pc

brz_path = pc.BRZ_PATH
slv_path = pc.SLV_PATH
img_path = pc.IMG_PATH
model_path = pc.MODEL_PATH
colors = pc.COLORS

# pre processing functions

def rm_columns(df: pd.DataFrame, rm: str='computed') -> pd.DataFrame:
    """Remove columns from a DataFrame that contain a specific substring in their names.
    """
    df_ = df.copy()
    return df_[[c for c in df_.columns if rm not in c]].copy()


def process_lat_lon(df: pd.DataFrame, orig_names: list, prefix: str='') -> pd.DataFrame:
    """Process latitude and longitude columns in a DataFrame, such as changing their names
    and converting them to float type.
    """
    df_ = df.copy()

    df_ = df_.rename(
        columns={
            orig_names[0]: f'{prefix}lat',
            orig_names[1]: f'{prefix}lon'
        }
    )

    for c in (f'{prefix}lat', f'{prefix}lon'):
        df_[c] = df_[c].astype(float)

    return df_


# feature engineering functions

def haversine_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance, in km, between two points, given their latitudes
    and longitudes.
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # apply haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in km
    return c * r


def get_min_distance(lat: float, lon: float, stations: pd.DataFrame) -> float:
    """Calculate the minimum distance from a given point (lat, lon) to a set of
    police stations, with corresponding (lat, lon) coordinates.
    """
    df = stations.copy()

    df['crime_lat'] = np.repeat(lat, len(df))
    df['crime_lon'] = np.repeat(lon, len(df))

    df['distance'] = df.apply(
        lambda row: haversine_dist(
            row["crime_lat"],
            row["crime_lon"],
            row["station_lat"],
            row["station_lon"]
        ),
        axis=1
    )
    return df['distance'].min()


def get_max_distance(lat: float, lon: float, stations: pd.DataFrame) -> float:
    """Calculate the maximum distance from a given point (lat, lon) to a set of
    police stations, with corresponding (lat, lon) coordinates.
    """
    df = stations.copy()

    df['crime_lat'] = np.repeat(lat, len(df))
    df['crime_lon'] = np.repeat(lon, len(df))

    df['distance'] = df.apply(
        lambda row: haversine_dist(
            row["crime_lat"],
            row["crime_lon"],
            row["station_lat"],
            row["station_lon"]
        ),
        axis=1
    )
    return df['distance'].max()
