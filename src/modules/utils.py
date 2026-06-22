import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt, log1p
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
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


def hopkins(X: np.ndarray, m: int=None) -> float:
    if m is None:
        m = int(.1 * len(X))

    d = X.shape[1]

    # sample m observations
    idx = np.random.choice(len(X), m, replace=False)
    X_sample = X[idx]

    # generate random points
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    X_random = np.random.uniform(mins, maxs, (m, d))

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    # distances from sampled observations to nearest neighbor
    w_dist, _ = nbrs.kneighbors(X_sample)
    w = w_dist[:, 1]

    # distances from random points to nearest observation
    u_dist, _ = nbrs.kneighbors(X_random, n_neighbors=1)
    u = u_dist[:, 0]

    H = np.sum(u) / (np.sum(u) + np.sum(w))
    return H


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
            .groupby([grp_by, 'ddate']).agg(
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

        grp_df.set_index('ddate', inplace=True)
    elif grp_by == 'community_area':
        grp_df = df_\
            .query('crime_group == @crime_group')\
            .groupby([grp_by, 'ddate']).agg(
                total=('id', 'count')
            )\
            .reset_index()

        grp_df.set_index('ddate', inplace=True)
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
            'district', 'ddate',
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
            'community_area', 'ddate',
            f'{prefix}_RollSum{window}'
        ]

    rolling['ddate'] = pd.to_datetime(rolling['ddate'])
    rolling = rolling.set_index('ddate')
    rolling = rolling.sort_index()
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


def rm_outliers(df: pd.DataFrame, cols: list=[], s: float=1.5) -> pd.DataFrame:
    """
    Receives an input dataframe and outputs another dataframe with its outliers removed. They
    are calculated based on the Interquartile Range (IQR) method and considers only the specified columns.
    The parameter 's' is a scaling factor for the IQR to define the bounds for:

    IQR = Q3 - Q1
    Lower Bound = Q1 - s * IQR
    Upper Bound = Q3 + s * IQR
    """
    cols = cols if cols else df.columns.tolist()

    df_ = df.copy()
    q1 = df_[cols].quantile(0.25)
    q3 = df_[cols].quantile(0.75)
    iqr = q3 - q1

    iqr_lower = q1 - s * iqr
    iqr_upper = q3 + s * iqr

    outliers = (
        (df_ < iqr_lower) | (df_ > iqr_upper)
    ).any(axis=1)

    n_out = outliers.sum()
    total = len(outliers)

    print(f'N. of outliers: {n_out} ({round(n_out * 100 / total, 3)}%)')
    return df_[~outliers].reset_index(drop=True)


def KMeans_features(x: np.ndarray, k_values: list | tuple) -> dict:
    """
    Perform KMeans clustering for a range of cluster numbers and compute
    the sum of squared errors (SSE) and silhouette scores for each k in k_values.

    Returns a dictionary with SSE and silhouette scores."""
    sse = []
    silhouettes = []
    for k in k_values:
        print(f'Initiating {k=}')
        kmeans = KMeans(n_clusters=k, random_state=1, n_init='auto')
        kmeans.fit(x)
        sse.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(x, kmeans.labels_))

    return {
        'sse': sse,
        'silhouettes': silhouettes,
    }


# metrics

def cluster_intensity(x: int | float | str, scale: str='log') -> float:
    """
    Calculate the intensity of a cluster based on the given scale.
    
    If linear scale is chosen, returns x as is, that is, I(x) = x.
    If logarithmic scale is chosen, returns I(x) = log(1 + x).
    """
    try:
        x = int(x)
    except ValueError:
        raise TypeError('x must be a number: int, float or numeric string')

    if scale not in ('linear', 'log'):
        raise TypeError("scale must be 'linear' or 'log' (logarithmic)")

    return x if scale == 'linear' else log1p(x)


def cluster_vol(cf: int, c0: int, d: int=7, scale: str='log') -> float:
    """
    Calculate the cluster volatility between two clusters cf and c0 over a
    specified duration d, using the given scale for intensity calculation.

    If linear scale is chosen, considers x as is, that is, I(x) = x.
    If logarithmic scale is chosen, considers I(x) = log(1 + x).
    """
    return (1/d) * abs(
        cluster_intensity(cf, scale) - cluster_intensity(c0, scale)
    )
