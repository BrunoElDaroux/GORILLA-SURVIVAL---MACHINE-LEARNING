"""
src/data_utils.py
==================
Reusable data loading, cleaning, and feature engineering utilities
for the Gorilla Survival ML project.

Import in any notebook:
    import sys; sys.path.append('..')
    from src.data_utils import load_all_data, build_feature_matrix
"""

import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


# ── FEATURE DEFINITIONS ───────────────────────────────────────────────────────
FEATURES = [
    # Demographic
    'age', 'age_class_ord', 'sex_binary',
    # Group structure
    'group_size', 'n_silverbacks', 'multi_male_group',
    'prop_silverbacks', 'prop_adult_females', 'prop_infants',
    'dependency_ratio', 'habitat_quality_score',
    # Individual status
    'dom_rank_norm', 'body_condition_score',
    # Behaviour (annual averages)
    'forage_pct', 'rest_pct', 'social_pct', 'travel_pct',
    'mean_foraging_pct', 'mean_agonistic', 'mean_affiliation',
    'social_integration', 'foraging_efficiency', 'cohesion_idx',
    # Environment
    'annual_rainfall_mm', 'lagged_rainfall_mm',
    'rainfall_anomaly_z', 'lagged_rain_z', 'rainfall_seasonality_idx',
]

TARGET = 'survived_next_year'

AGE_CLASS_MAP = {
    'infant': 0, 'juvenile': 1, 'subadult': 2, 'prime_adult': 3, 'senior': 4
}


def load_all_data(data_dir=None):
    """Load all four raw CSV files. Returns a dict of DataFrames."""
    d = data_dir or DATA_DIR
    return {
        'individuals' : pd.read_csv(os.path.join(d, 'gorilla_individuals.csv')),
        'groups'      : pd.read_csv(os.path.join(d, 'gorilla_groups.csv')),
        'rainfall'    : pd.read_csv(os.path.join(d, 'rainfall_monthly.csv')),
        'observations': pd.read_csv(os.path.join(d, 'gorilla_observations.csv')),
    }


def load_feature_matrix(data_dir=None):
    """Load the pre-built feature matrix (created by Notebook 03)."""
    d = data_dir or DATA_DIR
    path = os.path.join(d, 'feature_matrix.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"feature_matrix.csv not found at {path}. "
            "Run notebooks 01–03 first."
        )
    df = pd.read_csv(path)
    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df[TARGET]
    return X, y, df


def assign_age_class(age):
    """Map numeric age to biological age class label."""
    if age < 3:   return 'infant'
    if age < 8:   return 'juvenile'
    if age < 12:  return 'subadult'
    if age < 20:  return 'prime_adult'
    return 'senior'


def engineer_features(df, rain_df):
    """
    Apply all feature engineering transforms to the analytical DataFrame.

    Parameters
    ----------
    df      : pd.DataFrame — analytical_dataset.csv (with SQL joins applied)
    rain_df : pd.DataFrame — rainfall_monthly.csv

    Returns
    -------
    pd.DataFrame with all engineered features
    """
    df = df.copy()

    # Age class
    df['age_class']     = df['age'].apply(assign_age_class)
    df['age_class_ord'] = df['age_class'].map(AGE_CLASS_MAP)

    # Sex
    df['sex_binary'] = (df['sex'] == 'M').astype(int)

    # Group composition ratios
    gs = df['group_size'].clip(lower=1)
    df['prop_silverbacks']   = df['n_silverbacks']   / gs
    df['prop_adult_females'] = df['n_adult_females'] / gs
    df['prop_infants']       = df['n_infants']       / gs
    df['prop_juveniles']     = df['n_juveniles']     / gs

    denom = (df['n_adult_females'] + df['n_silverbacks'] + df['n_blackbacks']).clip(lower=1)
    df['dependency_ratio'] = (df['n_infants'] + df['n_juveniles']) / denom

    # Normalised dominance rank within group-year
    df['dom_rank_norm'] = df.groupby(['group_id', 'year'])['dominance_rank'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )

    # Rainfall features
    wet = rain_df[rain_df['season'] == 'wet'].groupby('year')['monthly_rainfall_mm'].sum()
    dry = rain_df[rain_df['season'] == 'dry'].groupby('year')['monthly_rainfall_mm'].sum()
    seasonality = (wet / (dry + 1)).rename('rainfall_seasonality_idx').reset_index()
    seasonality.columns = ['year', 'rainfall_seasonality_idx']
    df = df.merge(seasonality, on='year', how='left')

    ann_rain = df.groupby('year')['annual_rainfall_mm'].first()
    df['rainfall_anomaly_z'] = df['year'].map(
        (ann_rain - ann_rain.mean()) / ann_rain.std()
    )
    lag_mean = df['lagged_rainfall_mm'].mean()
    lag_std  = df['lagged_rainfall_mm'].std()
    df['lagged_rain_z'] = (df['lagged_rainfall_mm'] - lag_mean) / lag_std

    # Behavioural indices (only if observation-aggregated cols are present)
    if 'mean_affiliation' in df.columns and 'mean_agonistic' in df.columns:
        df['social_integration']  = df['mean_affiliation'] / (df['mean_agonistic'] + 1)
        df['foraging_efficiency'] = df['forage_pct'] * df['body_condition_score'] / 100
    if 'mean_nn_dist_m' in df.columns:
        df['cohesion_idx'] = 1 / (df['mean_nn_dist_m'] + 1)

    return df


def impute_observation_features(df):
    """Median-impute observation-derived features that may be missing."""
    obs_cols = [
        'n_obs_sessions', 'mean_foraging_pct', 'mean_agonistic',
        'mean_affiliation', 'mean_nn_dist_m', 'n_wet_sessions', 'n_dry_sessions',
        'social_integration', 'foraging_efficiency', 'cohesion_idx',
    ]
    present = [c for c in obs_cols if c in df.columns]
    if present:
        imputer = SimpleImputer(strategy='median')
        df[present] = imputer.fit_transform(df[present])
    return df


def temporal_split(df, X, y, train_cutoff=2020):
    """
    Split feature matrix temporally to avoid data leakage.

    Parameters
    ----------
    df           : full DataFrame (must contain 'year' column)
    X, y         : feature matrix and target
    train_cutoff : years <= this go to train, rest to test

    Returns
    -------
    X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    train_mask = df['year'] <= train_cutoff
    test_mask  = ~train_mask
    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        df[train_mask].index, df[test_mask].index
    )


def conservation_priority_table(ind_df):
    """
    Compute a per-group conservation priority score.

    Returns a DataFrame sorted by descending risk.
    """
    at_risk = ind_df.groupby('group_name').agg(
        mean_group_size = ('group_size', 'mean'),
        survival_rate   = ('survived_next_year', 'mean'),
        mean_age        = ('age', 'mean'),
        n_individuals   = ('individual_id', 'nunique'),
    ).reset_index()

    at_risk['mortality_rate'] = 1 - at_risk['survival_rate']

    # Composite risk score: small group penalty + high mortality penalty
    at_risk['risk_score'] = (
        - (at_risk['mean_group_size'] - at_risk['mean_group_size'].mean()) /
          at_risk['mean_group_size'].std()
        + (at_risk['mortality_rate'] - at_risk['mortality_rate'].mean()) /
          at_risk['mortality_rate'].std()
    )
    at_risk['priority'] = pd.cut(at_risk['risk_score'], bins=3,
                                  labels=['Low', 'Medium', 'High'])
    return at_risk.sort_values('risk_score', ascending=False).reset_index(drop=True)
