import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt, log1p
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from itertools import combinations
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


def KMeans_features(
        x: np.ndarray,
        x_latest: np.ndarray,
        k_values: list | tuple,
        seed: int=1
    ) -> dict:
    """
    Perform KMeans clustering for a range of cluster numbers and compute
    the sum of squared errors (SSE), silhouette, Calinski-Harabasz, and
    Davies-Bouldin scores for each k in k_values.

    Returns a dictionary with the scores.
    """
    sse = []
    silhouettes = []
    ch = []
    db = []
    for k in k_values:
        print(f'Initiating {k=}')
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init='auto')
        kmeans.fit(x)
        labels = kmeans.predict(x_latest)
        sse.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(x_latest, labels))
        ch.append(
            calinski_harabasz_score(x_latest, labels)
        )
        db.append(
            davies_bouldin_score(x_latest, labels)
        )

    return {
        'sse': sse,
        'silhouettes': silhouettes,
        'ch': ch,
        'db': db
    }


def fwd_feature_selection(
    df: pd.DataFrame,
    df_latest: np.ndarray,
    candidate_features: list,
    k_values: list=[2, 3, 4, 5],
    seed: int=1
) -> pd.DataFrame:
    """Forward feature selection for K-Means using multiple validation metrics.
    """
    remaining = candidate_features.copy()
    selected = []
    history = []
    detailed = []
    step = 1

    while len(remaining) > 0:
        candidate_summary = []
        for feature in remaining:
            current_features = selected + [feature]
            X = df[current_features].copy()
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            sil_scores, ch_scores, db_scores = [], [], []

            for k in k_values:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=seed,
                    n_init=50
                )
                kmeans.fit(X)
                X_latest = df_latest[current_features].copy()
                X_latest = scaler.transform(X_latest)
                labels = kmeans.predict(X_latest)

                sil = silhouette_score(X_latest, labels)
                ch = calinski_harabasz_score(X_latest, labels)
                db = davies_bouldin_score(X_latest, labels)

                sil_scores.append(sil)
                ch_scores.append(ch)
                db_scores.append(db)

                detailed.append({
                    'step': step,
                    'candidate': feature,
                    'features': current_features,
                    'k': k,
                    'silhouette': sil,
                    'calinski': ch,
                    'davies': db
                })
            
            candidate_summary.append({
                'candidate': feature,
                'features': current_features,
                'silhouette_mean': np.mean(sil_scores),
                'calinski_mean': np.mean(ch_scores),
                'davies_mean': np.mean(db_scores)
            })
        
        summary = pd.DataFrame(candidate_summary)

        # composite ranking
        summary['rank_sil'] = summary['silhouette_mean'].rank(ascending=False)
        summary['rank_ch'] = summary['calinski_mean'].rank(ascending=False)
        summary['rank_db'] = summary['davies_mean'].rank(ascending=False)

        summary['overall_rank'] = (
            summary['rank_sil']
            + summary['rank_ch']
            + summary['rank_db']
        )

        best = summary.sort_values('overall_rank').iloc[0]

        selected.append(best['candidate'])
        remaining.remove(best['candidate'])
        history.append(best)

        print('-' * 30)
        print(f'STEP {step}')
        print(f"Added feature: {best['candidate']}")
        print(f'Current set: {selected}')
        print(f"Mean Silhouette: {best['silhouette_mean']:.3f}")
        print(f"Mean Calinski-Harabasz: {best['calinski_mean']:.3f}")
        print(f"Mean Davies-Bouldin: {best['davies_mean']:.3f}")
        print('-' * 30)

        step += 1
    
    history = pd.DataFrame(history)
    detailed = pd.DataFrame(detailed)
    return selected, history, detailed


def exhaustive_feature_search(
    df: pd.DataFrame,
    df_latest: pd.DataFrame,
    candidate_features: list,
    k_values: list=[2, 3, 4, 5],
    seed: int=1,
    n_init: int=50,
    max_features: int=None
) -> pd.DataFrame:
    results = []
    n = len(candidate_features)

    if max_features is None:
        max_features = n

    for r in range(1, max_features + 1):
        for subset in combinations(candidate_features, r):
            scaler = StandardScaler()
            X = scaler.fit_transform(df[list(subset)])
            df_latest_ = df_latest[list(subset)].copy()
            X_latest = scaler.transform(df_latest_)

            subset_results = []

            for k in k_values:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=seed,
                    n_init=n_init
                )
                kmeans.fit(X)
                labels = kmeans.predict(X_latest)
                sil = silhouette_score(X_latest, labels)
                ch = calinski_harabasz_score(X_latest, labels)
                db = davies_bouldin_score(X_latest, labels)

                subset_results.append({
                    'k': k,
                    'silhouette': sil,
                    'calinski': ch,
                    'davies': db
                })
            subset_results = pd.DataFrame(subset_results)

            subset_results['rank_sil'] = (
                subset_results['silhouette']
                .rank(ascending=False)
            )
            subset_results['rank_ch'] = (
                subset_results['calinski']
                .rank(ascending=False)
            )
            subset_results['rank_db'] = (
                subset_results['davies']
                .rank(ascending=False)
            )

            subset_results['score'] = (
                subset_results['rank_sil']
                + subset_results['rank_ch']
                + subset_results['rank_db']
            )

            best = subset_results.sort_values('score').iloc[0]

            results.append({
                'features': subset,
                'n_features': len(subset),
                'best_k': int(best['k']),
                'silhouette': best['silhouette'],
                'calinski': best['calinski'],
                'davies': best['davies'],
                'score': best['score']
            })
    results = pd.DataFrame(results)

    results = results.sort_values(
        by=[
            'score',
            'silhouette',
            'calinski',
            'davies'
        ],
        ascending=[True, False, False, True]
    )
    return results.reset_index(drop=True)


# metrics
def transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the transition matrix for clusters in a DataFrame.
    The transition matrix shows the probabilities of moving from one cluster to another
    between consecutive time periods.

    Returns a DataFrame representing the transition matrix.
    """
    df_ = df.copy()

    df_ = df_.sort_values(['district', 'ddate'])

    # capture consecutive cluster transitions for each district
    df_['next_cluster'] = df_.groupby('district')['cluster'].shift(-1)

    # filtering out last entries (no next state)
    transitions = df_.dropna(subset=['next_cluster']).reset_index(drop=True)

    # count transitions
    counts = pd.crosstab(
        transitions['cluster'],
        transitions['next_cluster']
    )

    # converting to probabilities
    transition_matrix = counts.div(counts.sum(axis=1), axis=0)
    return transition_matrix


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
