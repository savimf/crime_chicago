import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency
import statsmodels.api as sm
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


# statistical tests and functions

def shapiro_test(s: pd.Series) -> None:
    """Perform the Shapiro-Wilk test for normality on a sample and prints the
    results.
    """
    stats, p = shapiro(s)

    print("--- Shapiro-Wilk Test --- ")
    print(f'Statistic: {stats}')
    print(f'P-value: {p}')


def student_test(s1: pd.Series, s2: pd.Series, alternative: str='greater') -> None:
    """Perform the Student's t test to compare two independent samples and prints the
    results.
    """
    stats, p = ttest_ind(s1, s2, alternative=alternative)

    print("--- Student T-Test --- ")
    print(f'Statistic: {stats}')
    print(f'P-value: {p}')


def mannwhitneyu_test(s1: pd.Series, s2: pd.Series, alternative: str='greater') -> None:
    """Perform the Mann-Whitney U test to compare two independent samples and prints the
    results.
    """
    stats, p = mannwhitneyu(s1, s2, alternative=alternative)

    print("---  Mann-Whitney U Test --- ")
    print(f'Statistic: {stats}')
    print(f'P-value: {p}')


def chi2_test(cont_table: pd.DataFrame) -> None:
    """Perform the Chi-squared test of independence on a contingency table and prints the
    results.
    """
    stats, p, _, _ = chi2_contingency(cont_table)

    print("--- Chi-squared Test of Independence --- ")
    print(f'Statistic: {stats}')
    print(f'P-value: {p}')


def run_log_model(df: pd.DataFrame, x_col: str, y_col: str):
    """Run a logarithmic regression model using statsmodels OLS.
    Receives the original dataframe, applies a log transformation to
    the target variable, and fits an OLS regression model with x_col as
    the predictor and y_col as the target.
    Returns the fitted model.
    """
    df_model = df.copy()
    X = df_model[x_col].copy()
    y = np.log(df_model[y_col]).copy()

    # Adding a constant for the intercept
    X = sm.add_constant(X)
    lr = sm.OLS(y, X).fit()
    return lr


def grp_features(
        df: pd.DataFrame,
        grp_by: str='district',
        crime_group: str='ASLT',
    ) -> pd.DataFrame:
    """Group the DataFrame by a specified column and date, aggregating various features
    based on the crime group.
    Returns the grouped dataframe.
    """
    df_ = df.copy()
    grp_df = pd.DataFrame()

    if grp_by == 'district':
        grp_df = df_\
            .query('crime_group == @crime_group')\
            .groupby([grp_by, 'date']).agg(
                count=('id', 'count'),
                domestic_rate=('domestic', 'mean'),
                arrest_rate=('arrest', 'mean'),
                avg_dist=('crime_distance', 'mean'),
                avg_max_dist=('max_dist_station', 'mean'),
                avg_cm_area=('avg_cm_area', 'mean'),
                avg_cm_len=('avg_cm_len', 'mean'),
                avg_total_cm_area=('total_cm_area', 'mean'),
                avg_total_cm_len=('total_cm_len', 'mean'),
                avg_hardship_index=('hardship_index', 'mean')
            )\
            .reset_index()

        grp_df.set_index('date', inplace=True)
    elif grp_by == 'community_area':
        grp_df = df_\
            .query('crime_group == @crime_group')\
            .groupby([grp_by, 'date']).agg(
                total=('id', 'count')
            )\
            .reset_index()

        grp_df.set_index('date', inplace=True)
    return grp_df


def rolling_grp(
        df: pd.DataFrame,
        window: int=7,
        grp_by: str='district',
        prefix: str='ASLT'
    ) -> pd.DataFrame:
    """
    Calculate rolling statistics for grouped features in the DataFrame.
    Depending on the grouping column, it computes rolling averages or sums
    over a specified window.
    """
    rolling = pd.DataFrame()

    if grp_by == 'district':
        rolling = df\
            .groupby('district')\
            .rolling(window).agg({
                'count': 'mean',
                'domestic_rate': 'mean',
                'arrest_rate': 'mean',
                'avg_dist': 'mean',
                'avg_max_dist': 'mean',
                'avg_cm_len': 'mean',
                'avg_total_cm_len': 'mean',
                'avg_hardship_index': 'mean',
                'avg_count_cm_area': 'mean',
                'avg_drate_cm_area': 'mean',
                'avg_arate_cm_area': 'mean',
                'avg_crime_density': 'mean'
                
            }).reset_index().copy()

        rolling.columns = [
            'district', 'date',
            f'{prefix}_RollAvg{window}',
            f'{prefix}_RollDRate{window}',
            f'{prefix}_RollARate{window}',
            f'{prefix}_RollAvgDist{window}',
            f'{prefix}_RollAvgMaxDist{window}',
            f'{prefix}_RollAvgCMLen{window}',
            f'{prefix}_RollAvgLen{window}',
            f'{prefix}_RollAvgHardship{window}',
            f'{prefix}_RollAvgCountCMArea{window}',
            f'{prefix}_RollAvgDRateCMArea{window}',
            f'{prefix}_RollAvgARateCMArea{window}',
            f'{prefix}_RollAvgDensity{window}'
        ]
    elif grp_by == 'community_area':
        # for community area, we care only for its rolling sum
        rolling = df\
            .groupby('community_area')\
            .rolling(window).agg({
                'total': 'sum'
            })\
            .reset_index().copy()
        
        rolling.columns = [
            'community_area', 'date',
            f'{prefix}_RollSum{window}'
        ]

    return rolling


# plotting functions
def gen_paired_hist_box(df: pd.DataFrame, grp: str='ASLT') -> None:
    cols = df.columns
    n_cols = len(cols)

    fig, axes = plt.subplots(2, n_cols, figsize=(15, 8), sharey='row')

    # Plot histograms in the upper row
    for i, col in enumerate(cols):
        sns.histplot(
            data=df,
            x=col,
            ax=axes[0, i],
            kde=True,
            color=colors.get(grp)
        )
        axes[0, i].set_title(f'{col} Distribution')
        axes[0, i].set_xlabel('')
        axes[0, i].set_xticks([])
        axes[0, i].spines[['top', 'right', 'left']].set_visible(False)

    # Plot boxplots in the lower row
    for i, col in enumerate(cols):
        sns.boxplot(
            data=df,
            x=col,
            ax=axes[1, i],
            color=colors.get(grp)
        )
        axes[1, i].set_title('')
        axes[1, i].set_ylabel('')
        axes[1, i].set_xlabel('')
        axes[1, i].spines[['top', 'right', 'bottom', 'left']].set_visible(False)