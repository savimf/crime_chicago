import pandas as pd
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
