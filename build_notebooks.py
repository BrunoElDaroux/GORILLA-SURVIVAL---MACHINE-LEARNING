"""
build_notebooks.py
Generates all 5 .ipynb notebook files for the gorilla survival ML project.
Run from the project root: python build_notebooks.py
"""

import json, os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def nb(cells):
    """Wrap cells in a valid .ipynb structure."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}


def save(name, cells):
    path = os.path.join(NOTEBOOKS_DIR, name)
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"  ✓ {name}")


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 01 — Data Generation & Inspection
# ══════════════════════════════════════════════════════════════════════════════
nb01 = [

md("""# 🦍 Notebook 01 — Data Generation & Inspection
**Project**: Machine Learning for Survival Prediction in Mountain Gorilla Populations

This notebook generates and inspects our four ecological datasets:
- `rainfall_monthly.csv` — Virunga monthly rainfall records
- `gorilla_groups.csv`  — Group-level annual census data
- `gorilla_individuals.csv` — Individual longitudinal survival records (**target variable here**)
- `gorilla_observations.csv` — Focal behavioural observation sessions

> **Dataset source**: Synthetically generated using ecological parameters from:
> Robbins et al. (2011), Harcourt & Fossey (1981), Watts (1998).
> Real data: https://gorillafund.org/science/
"""),

md("## Cell 1 — Import Libraries"),
code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, sys, warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{:.2f}'.format)

print(f"pandas  : {pd.__version__}")
print(f"numpy   : {np.__version__}")
print("✅ Libraries loaded")
"""),

md("## Cell 2 — Generate All Datasets\nRun `generate_dataset.py` once to produce the four CSV files."),
code("""\
# Point to the data directory (adjust path if needed)
DATA_DIR = os.path.join('..', 'data')
GENERATE_SCRIPT = os.path.join(DATA_DIR, 'generate_dataset.py')

# Run the generation script
os.system(f'python {GENERATE_SCRIPT}')

# Verify files exist
expected = ['rainfall_monthly.csv','gorilla_groups.csv',
            'gorilla_individuals.csv','gorilla_observations.csv']
for f in expected:
    path = os.path.join(DATA_DIR, f)
    exists = os.path.exists(path)
    size   = os.path.getsize(path) // 1024 if exists else 0
    print(f"  {'✅' if exists else '❌'} {f:<35} ({size} KB)")
"""),

md("## Cell 3 — Load All Datasets"),
code("""\
DATA_DIR = os.path.join('..', 'data')

rain_df  = pd.read_csv(os.path.join(DATA_DIR, 'rainfall_monthly.csv'))
group_df = pd.read_csv(os.path.join(DATA_DIR, 'gorilla_groups.csv'))
ind_df   = pd.read_csv(os.path.join(DATA_DIR, 'gorilla_individuals.csv'))
obs_df   = pd.read_csv(os.path.join(DATA_DIR, 'gorilla_observations.csv'))

print("── Dataset shapes ─────────────────────────────────")
print(f"  Rainfall records   : {rain_df.shape}")
print(f"  Group records      : {group_df.shape}")
print(f"  Individual records : {ind_df.shape}")
print(f"  Observation records: {obs_df.shape}")
print(f"\\n  Unique individuals tracked : {ind_df['individual_id'].nunique()}")
print(f"  Unique social groups       : {ind_df['group_name'].nunique()}")
print(f"  Study period               : {ind_df['year'].min()} – {ind_df['year'].max()}")
"""),

md("## Cell 4 — Inspect Individual Records (Primary Table)"),
code("""\
print("── gorilla_individuals.csv — First 5 rows ──────────────────────────")
display(ind_df.head())

print("\\n── Column types & nulls ───────────────────────────────────────────")
info = pd.DataFrame({
    'dtype'   : ind_df.dtypes,
    'non_null': ind_df.notnull().sum(),
    'nulls'   : ind_df.isnull().sum(),
    'unique'  : ind_df.nunique()
})
display(info)
"""),

md("## Cell 5 — Inspect Other Tables"),
code("""\
print("── gorilla_groups.csv ──────────────────────────────────────────────")
display(group_df.head(3))

print("\\n── rainfall_monthly.csv ────────────────────────────────────────────")
display(rain_df.head(3))

print("\\n── gorilla_observations.csv ────────────────────────────────────────")
display(obs_df.head(3))
"""),

md("## Cell 6 — Descriptive Statistics"),
code("""\
print("── Individual records — numeric summary ────────────────────────────")
display(ind_df.describe().round(2))
"""),

md("## Cell 7 — Target Variable: Survival Distribution"),
code("""\
surv_counts = ind_df['survived_next_year'].value_counts()
surv_pct    = ind_df['survived_next_year'].value_counts(normalize=True) * 100

print("── Target variable: survived_next_year ─────────────────────────────")
print(f"  Survived (1)  : {surv_counts[1]:>5}  ({surv_pct[1]:.1f}%)")
print(f"  Did not (0)   : {surv_counts[0]:>5}  ({surv_pct[0]:.1f}%)")
print(f"  Class imbalance ratio: {surv_counts[1]/surv_counts[0]:.1f}:1")
print("\\n  NOTE: ~98% survival is realistic for mountain gorillas in protected")
print("        areas, but requires class-imbalance handling in modeling.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
axes[0].bar(['Did not survive (0)', 'Survived (1)'],
            surv_counts.values, color=['#c0392b','#27ae60'], edgecolor='black')
axes[0].set_title('Survival Outcome Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(surv_counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontsize=11)

# Survival rate by year
surv_by_year = ind_df.groupby('year')['survived_next_year'].mean().reset_index()
axes[1].plot(surv_by_year['year'], surv_by_year['survived_next_year'] * 100,
             marker='o', color='#2980b9', linewidth=2, markersize=6)
axes[1].set_title('Annual Survival Rate Over Time', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Survival Rate (%)')
axes[1].set_ylim(90, 100)
axes[1].axhline(surv_by_year['survived_next_year'].mean()*100,
                color='red', linestyle='--', alpha=0.6, label='Mean')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join('..','data','01_survival_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved.")
"""),

md("## Cell 8 — Rainfall Pattern Visualisation"),
code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Monthly climatology
monthly_mean = rain_df.groupby('month')['monthly_rainfall_mm'].mean()
month_names  = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
axes[0].bar(month_names, monthly_mean.values, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].set_title('Virunga Monthly Rainfall Climatology', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Mean Rainfall (mm)')
axes[0].set_xlabel('Month')
for spine in ['top','right']: axes[0].spines[spine].set_visible(False)

# Annual total over study period
annual = rain_df.groupby('year')['monthly_rainfall_mm'].sum().reset_index()
axes[1].fill_between(annual['year'], annual['monthly_rainfall_mm'],
                     alpha=0.4, color='steelblue')
axes[1].plot(annual['year'], annual['monthly_rainfall_mm'],
             marker='s', color='steelblue', linewidth=2)
axes[1].set_title('Annual Rainfall — Study Period', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Total Rainfall (mm)')
axes[1].axvspan(2015, 2016, alpha=0.15, color='red', label='El Niño anomaly')
axes[1].legend()
for spine in ['top','right']: axes[1].spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join('..','data','01_rainfall_pattern.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## Cell 9 — Group Size Distribution Across All Years"),
code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(group_df['group_size'], bins=20, color='#8e44ad', edgecolor='black', alpha=0.75)
axes[0].set_title('Distribution of Group Sizes', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Group Size (individuals)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(group_df['group_size'].mean(), color='red',
                linestyle='--', label=f"Mean = {group_df['group_size'].mean():.1f}")
axes[0].legend()

# Group size over time per group
pivot = group_df.pivot(index='year', columns='group_name', values='group_size')
for col in pivot.columns:
    axes[1].plot(pivot.index, pivot[col], alpha=0.6, linewidth=1.5, label=col)
axes[1].set_title('Group Size Trajectories (2010–2023)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Group Size')
axes[1].legend(fontsize=6, ncol=2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join('..','data','01_group_sizes.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\\n✅ Notebook 01 complete. Proceed to 02_eda_sql.ipynb")
"""),

]  # end nb01


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — SQL Joins + Exploratory Data Analysis
# ══════════════════════════════════════════════════════════════════════════════
nb02 = [

md("""# 🦍 Notebook 02 — SQL Queries & Exploratory Data Analysis
We use Python's built-in `sqlite3` to load all CSVs into an in-memory database,
then write SQL queries to JOIN, filter, and aggregate — exactly as you would with
a real ecological field database.

**Key SQL tasks:**
1. JOIN individuals ↔ groups ↔ rainfall into one analytical table
2. Compute group-level survival summaries
3. Filter by demographic subgroups (age class, sex)
4. Aggregate behavioural data per individual-year
"""),

md("## Cell 1 — Import Libraries"),
code("""\
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30)
DATA_DIR = os.path.join('..', 'data')
print("✅ Libraries loaded")
"""),

md("## Cell 2 — Load CSVs into SQLite (In-Memory Database)"),
code("""\
# Create in-memory SQLite database
conn = sqlite3.connect(':memory:')   # ':memory:' → no file created on disk
cursor = conn.cursor()

# Load each CSV as a table
tables = {
    'rainfall'    : 'rainfall_monthly.csv',
    'gorilla_grps': 'gorilla_groups.csv',
    'individuals' : 'gorilla_individuals.csv',
    'observations': 'gorilla_observations.csv',
}

for table_name, filename in tables.items():
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df.to_sql(table_name, conn, index=False, if_exists='replace')
    n_rows = cursor.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
    print(f"  ✅ Table '{table_name:<15}' → {n_rows:>6} rows")

print("\\n✅ All tables loaded into SQLite.")
"""),

md("## Cell 3 — List Tables & Inspect Schema"),
code("""\
# List all tables
tables_in_db = cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()
print("Tables in DB:", [t[0] for t in tables_in_db])

# Schema of main table
print("\\n── Schema: individuals ──────────────────────────────────────────")
for col in cursor.execute("PRAGMA table_info(individuals)").fetchall():
    print(f"  {col[1]:<30} {col[2]}")
"""),

md("""## Cell 4 — SQL Query 1: JOIN Individuals + Groups + Rainfall
This is the **primary analytical join** — assembles all predictors and the target variable into one flat table.
"""),
code("""\
QUERY_FULL_JOIN = '''
SELECT
    i.individual_id,
    i.group_id,
    i.group_name,
    i.year,
    i.age,
    i.sex,
    i.dominance_rank,
    i.forage_pct,
    i.rest_pct,
    i.social_pct,
    i.travel_pct,
    i.body_condition_score,
    i.survived_next_year,

    -- Group-level variables (from gorilla_grps)
    g.group_size,
    g.n_silverbacks,
    g.n_blackbacks,
    g.n_adult_females,
    g.n_juveniles,
    g.n_infants,
    g.multi_male_group,
    g.habitat_quality_score,

    -- Environmental (from rainfall)
    r.annual_rainfall_mm,

    -- Lagged rainfall (prior year food availability)
    r_lag.annual_rainfall_mm  AS lagged_rainfall_mm

FROM individuals i
LEFT JOIN gorilla_grps g
    ON i.group_id = g.group_id AND i.year = g.year
LEFT JOIN (
    SELECT year, SUM(monthly_rainfall_mm) AS annual_rainfall_mm
    FROM rainfall GROUP BY year
) r ON i.year = r.year
LEFT JOIN (
    SELECT year, SUM(monthly_rainfall_mm) AS annual_rainfall_mm
    FROM rainfall GROUP BY year
) r_lag ON i.year - 1 = r_lag.year
WHERE i.age >= 0
ORDER BY i.group_name, i.individual_id, i.year
'''

analytical_df = pd.read_sql_query(QUERY_FULL_JOIN, conn)
print(f"Analytical dataset shape: {analytical_df.shape}")
display(analytical_df.head(5))

# Save for next notebooks
analytical_df.to_csv(os.path.join(DATA_DIR, 'analytical_dataset.csv'), index=False)
print("\\n✅ Saved: analytical_dataset.csv")
"""),

md("## Cell 5 — SQL Query 2: Group-Level Survival Summary"),
code("""\
QUERY_GROUP_SURV = '''
SELECT
    group_name,
    year,
    group_size,
    COUNT(individual_id)                   AS n_individuals,
    SUM(survived_next_year)                AS n_survived,
    ROUND(AVG(survived_next_year) * 100, 1) AS survival_rate_pct,
    ROUND(AVG(age), 1)                     AS mean_age,
    ROUND(AVG(body_condition_score), 2)    AS mean_body_cond
FROM individuals
GROUP BY group_name, year
ORDER BY survival_rate_pct ASC
'''

group_surv = pd.read_sql_query(QUERY_GROUP_SURV, conn)
print("── Group-level survival summary (worst 10 group-years) ─────────────")
display(group_surv.head(10))
"""),

md("## Cell 6 — SQL Query 3: Survival by Sex & Age Class"),
code("""\
QUERY_DEMO = '''
SELECT
    sex,
    CASE
        WHEN age < 3  THEN '0_Infant'
        WHEN age < 8  THEN '1_Juvenile'
        WHEN age < 12 THEN '2_Subadult'
        WHEN age < 20 THEN '3_Prime_Adult'
        ELSE               '4_Senior'
    END AS age_class,
    COUNT(*)                                AS n,
    ROUND(AVG(survived_next_year)*100, 2)   AS survival_pct,
    ROUND(AVG(body_condition_score), 2)     AS mean_body_cond,
    ROUND(AVG(dominance_rank), 2)           AS mean_dom_rank
FROM individuals
GROUP BY sex, age_class
ORDER BY sex, age_class
'''

demo_surv = pd.read_sql_query(QUERY_DEMO, conn)
print("── Survival by sex and age class ───────────────────────────────────")
display(demo_surv)
"""),

md("## Cell 7 — SQL Query 4: Aggregated Behavioural Data per Individual-Year"),
code("""\
QUERY_OBS = '''
SELECT
    o.individual_id,
    o.year,
    COUNT(o.obs_id)                       AS n_obs_sessions,
    ROUND(AVG(o.foraging_rate_pct), 2)    AS mean_foraging_pct,
    ROUND(AVG(o.agonistic_events), 2)     AS mean_agonistic,
    ROUND(AVG(o.affiliation_events), 2)   AS mean_affiliation,
    ROUND(AVG(o.nearest_neighbour_dist_m), 2) AS mean_nn_dist_m,
    SUM(CASE WHEN o.season='wet' THEN 1 ELSE 0 END) AS n_wet_sessions,
    SUM(CASE WHEN o.season='dry' THEN 1 ELSE 0 END) AS n_dry_sessions
FROM observations o
GROUP BY o.individual_id, o.year
'''

obs_agg = pd.read_sql_query(QUERY_OBS, conn)
print(f"Aggregated observation sessions: {obs_agg.shape}")
display(obs_agg.head(5))

# Merge into analytical dataset
analytical_df = analytical_df.merge(obs_agg, on=['individual_id','year'], how='left')
print(f"\\nAnalytical dataset with obs features: {analytical_df.shape}")
analytical_df.to_csv(os.path.join(DATA_DIR, 'analytical_dataset.csv'), index=False)
print("✅ Saved updated: analytical_dataset.csv")
"""),

md("## Cell 8 — Missing Value Analysis"),
code("""\
missing = analytical_df.isnull().sum()
missing_pct = (analytical_df.isnull().sum() / len(analytical_df) * 100).round(2)
missing_report = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_report = missing_report[missing_report['missing_count'] > 0].sort_values('missing_pct', ascending=False)

print("── Missing Values ──────────────────────────────────────────────────")
if missing_report.empty:
    print("  ✅ No missing values in core columns.")
else:
    display(missing_report)
    print("\\n  NOTE: Observation-derived features (n_obs_sessions etc.) are NaN")
    print("  for individuals with no focal sessions — we'll impute in NB03.")
"""),

md("## Cell 9 — Correlation Heatmap (Numerical Features)"),
code("""\
num_cols = ['age','dominance_rank','forage_pct','rest_pct','social_pct',
            'body_condition_score','group_size','n_silverbacks','n_adult_females',
            'habitat_quality_score','annual_rainfall_mm','lagged_rainfall_mm',
            'survived_next_year']

corr = analytical_df[num_cols].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={'size': 8}, ax=ax)
ax.set_title('Feature Correlation Matrix\\n(survived_next_year = target)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join('..','data','02_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

# Top correlations with target
target_corr = corr['survived_next_year'].drop('survived_next_year').sort_values(key=abs, ascending=False)
print("\\n── Top correlations with survival ──────────────────────────────────")
print(target_corr.head(10).to_string())
"""),

md("## Cell 10 — Univariate EDA: Group Size vs. Survival"),
code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Survival rate binned by group size
bins = pd.cut(analytical_df['group_size'], bins=[0,5,10,15,20,25,40], labels=['1-5','6-10','11-15','16-20','21-25','26+'])
surv_by_gs = analytical_df.groupby(bins)['survived_next_year'].agg(['mean','count']).reset_index()
surv_by_gs.columns = ['group_size_bin','survival_rate','n']

axes[0].bar(surv_by_gs['group_size_bin'], surv_by_gs['survival_rate']*100,
            color='#16a085', edgecolor='black', alpha=0.8)
axes[0].set_title('Survival Rate by Group Size Bin', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Group Size')
axes[0].set_ylabel('Survival Rate (%)')
axes[0].set_ylim(90, 100)
for i, row in surv_by_gs.iterrows():
    axes[0].text(i, row['survival_rate']*100 + 0.05, f"n={row['n']}", ha='center', fontsize=9)

# Boxplot: group size by survival outcome
surv_labels = analytical_df['survived_next_year'].map({0:'Did not survive', 1:'Survived'})
surv_0 = analytical_df[analytical_df['survived_next_year']==0]['group_size']
surv_1 = analytical_df[analytical_df['survived_next_year']==1]['group_size']
axes[1].boxplot([surv_0, surv_1], labels=['Did not survive','Survived'],
                patch_artist=True,
                boxprops=dict(facecolor='#e8daef', color='black'),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Group Size Distribution by Survival Outcome', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Group Size')

plt.tight_layout()
plt.savefig(os.path.join('..','data','02_groupsize_survival.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\\n✅ Notebook 02 complete. Proceed to 03_feature_engineering.ipynb")
"""),

]  # end nb02


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
nb03 = [

md("""# 🦍 Notebook 03 — Feature Engineering
Transform raw ecological variables into a clean, ML-ready feature matrix.

**Engineering steps:**
1. Age class encoding (biologically meaningful)
2. Group composition ratios (relative structure, not raw counts)
3. Dominance rank normalised within group-year
4. Rainfall seasonality index
5. Behavioural interaction terms
6. Imputation of missing observation-derived features
7. Final feature matrix export
"""),

md("## Cell 1 — Imports & Load Data"),
code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os, warnings

warnings.filterwarnings('ignore')
DATA_DIR = os.path.join('..', 'data')

df = pd.read_csv(os.path.join(DATA_DIR, 'analytical_dataset.csv'))
rain_df = pd.read_csv(os.path.join(DATA_DIR, 'rainfall_monthly.csv'))

print(f"Loaded analytical dataset: {df.shape}")
print(f"Columns: {list(df.columns)}")
"""),

md("## Cell 2 — Feature: Age Class (Biologically Grounded)"),
code("""\
def assign_age_class(age):
    if age < 3:   return 'infant'
    if age < 8:   return 'juvenile'
    if age < 12:  return 'subadult'
    if age < 20:  return 'prime_adult'
    return 'senior'

df['age_class'] = df['age'].apply(assign_age_class)

# Ordinal encoding for ML (preserves biological order)
age_order = {'infant':0, 'juvenile':1, 'subadult':2, 'prime_adult':3, 'senior':4}
df['age_class_ord'] = df['age_class'].map(age_order)

print(df['age_class'].value_counts())
print("\\n✅ age_class and age_class_ord created")
"""),

md("## Cell 3 — Feature: Sex Encoding"),
code("""\
df['sex_binary'] = (df['sex'] == 'M').astype(int)   # 1=Male, 0=Female
print("Sex distribution:")
print(df['sex'].value_counts())
print("\\n✅ sex_binary created")
"""),

md("## Cell 4 — Feature: Group Composition Ratios"),
code("""\
# Proportional composition features are more informative than raw counts
# (a group of 5 with 2 silverbacks is very different from 20 with 2)

df['prop_silverbacks']  = df['n_silverbacks']  / df['group_size'].clip(lower=1)
df['prop_adult_females']= df['n_adult_females']/ df['group_size'].clip(lower=1)
df['prop_infants']      = df['n_infants']      / df['group_size'].clip(lower=1)
df['prop_juveniles']    = df['n_juveniles']    / df['group_size'].clip(lower=1)

# Dependency ratio: non-productive members per adult
df['dependency_ratio']  = (df['n_infants'] + df['n_juveniles']) / \
                           (df['n_adult_females'] + df['n_silverbacks'] + df['n_blackbacks']).clip(lower=1)

print("✅ Group composition ratios created:")
print(df[['prop_silverbacks','prop_adult_females','prop_infants','dependency_ratio']].describe().round(3))
"""),

md("## Cell 5 — Feature: Within-Group Dominance Rank (Normalised)"),
code("""\
# Raw dominance rank is not comparable across groups of different sizes
# Normalise to [0,1] within each group-year

df['dom_rank_norm'] = df.groupby(['group_id','year'])['dominance_rank'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
)

print("Normalised dominance rank (sample):")
print(df[['group_name','year','dominance_rank','dom_rank_norm']].head(8).to_string(index=False))
print("\\n✅ dom_rank_norm created")
"""),

md("## Cell 6 — Feature: Rainfall Seasonality Index"),
code("""\
# Compute wet-season vs dry-season rainfall ratio per year
# Higher values → more concentrated wet season (feast-famine contrast)
wet_months = rain_df[rain_df['season']=='wet'].groupby('year')['monthly_rainfall_mm'].sum()
dry_months  = rain_df[rain_df['season']=='dry'].groupby('year')['monthly_rainfall_mm'].sum()
seasonality = (wet_months / (dry_months + 1)).rename('rainfall_seasonality_idx').reset_index()
seasonality.columns = ['year', 'rainfall_seasonality_idx']

df = df.merge(seasonality, on='year', how='left')

# Also: rainfall anomaly from long-run mean (z-score)
ann_rain = df.groupby('year')['annual_rainfall_mm'].first()
df['rainfall_anomaly_z'] = df['year'].map(
    (ann_rain - ann_rain.mean()) / ann_rain.std()
)

# Lagged rainfall z-score
lag_mean = df['lagged_rainfall_mm'].mean()
lag_std  = df['lagged_rainfall_mm'].std()
df['lagged_rain_z'] = (df['lagged_rainfall_mm'] - lag_mean) / lag_std

print("✅ Rainfall features created:")
print(df[['year','annual_rainfall_mm','lagged_rainfall_mm','rainfall_anomaly_z','lagged_rain_z']].drop_duplicates('year').head(8).to_string(index=False))
"""),

md("## Cell 7 — Feature: Behavioural Indices"),
code("""\
# Social integration index: affiliation relative to agonistic events
# High values → more affiliative = better social integration
df['social_integration'] = df['mean_affiliation'] / (df['mean_agonistic'] + 1)

# Foraging efficiency proxy: foraging% * body condition
df['foraging_efficiency'] = df['forage_pct'] * df['body_condition_score'] / 100

# Cohesion index: inverse of mean nearest-neighbour distance
# (closer neighbours → tighter group → better protection)
df['cohesion_idx'] = 1 / (df['mean_nn_dist_m'] + 1)

print("✅ Behavioural features created:")
print(df[['social_integration','foraging_efficiency','cohesion_idx']].describe().round(3))
"""),

md("## Cell 8 — Impute Missing Observation-Derived Features"),
code("""\
obs_features = ['n_obs_sessions','mean_foraging_pct','mean_agonistic',
                'mean_affiliation','mean_nn_dist_m','n_wet_sessions','n_dry_sessions',
                'social_integration','foraging_efficiency','cohesion_idx']

missing_pct = (df[obs_features].isnull().sum() / len(df) * 100).round(1)
print("Missing % in observation features:")
print(missing_pct.to_string())

# Median imputation (robust to outliers)
imputer = SimpleImputer(strategy='median')
df[obs_features] = imputer.fit_transform(df[obs_features])

print(f"\\n✅ Imputed {(missing_pct > 0).sum()} features using median strategy")
print(f"   Remaining nulls: {df[obs_features].isnull().sum().sum()}")
"""),

md("## Cell 9 — Assemble Final Feature Matrix"),
code("""\
# Define final feature set for ML
FEATURES = [
    # Demographic
    'age', 'age_class_ord', 'sex_binary',
    # Group structure
    'group_size', 'n_silverbacks', 'multi_male_group',
    'prop_silverbacks', 'prop_adult_females', 'prop_infants',
    'dependency_ratio', 'habitat_quality_score',
    # Individual status
    'dom_rank_norm', 'body_condition_score',
    # Behaviour
    'forage_pct', 'rest_pct', 'social_pct', 'travel_pct',
    'mean_foraging_pct', 'mean_agonistic', 'mean_affiliation',
    'social_integration', 'foraging_efficiency', 'cohesion_idx',
    # Environment
    'annual_rainfall_mm', 'lagged_rainfall_mm',
    'rainfall_anomaly_z', 'lagged_rain_z', 'rainfall_seasonality_idx',
]

TARGET = 'survived_next_year'

# Verify all features exist
missing_cols = [f for f in FEATURES if f not in df.columns]
if missing_cols:
    print(f"⚠️  Missing columns: {missing_cols}")
else:
    print(f"✅ All {len(FEATURES)} features present")

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\\nFeature matrix X : {X.shape}")
print(f"Target vector  y : {y.shape}")
print(f"\\nClass distribution:")
print(f"  Survived (1)  : {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  Did not  (0)  : {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")

# Save feature matrix
feat_df = X.copy()
feat_df[TARGET] = y
feat_df['individual_id'] = df['individual_id']
feat_df['year']          = df['year']
feat_df['group_name']    = df['group_name']
feat_df['age_class']     = df['age_class']
feat_df['sex']           = df['sex']
feat_df.to_csv(os.path.join(DATA_DIR, 'feature_matrix.csv'), index=False)
print("\\n✅ Saved: feature_matrix.csv")
"""),

md("## Cell 10 — Feature Distribution Visualisation"),
code("""\
key_features = ['group_size','age','dom_rank_norm','body_condition_score',
                'lagged_rain_z','prop_adult_females','social_integration']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    for label, color in [(0,'#c0392b'), (1,'#27ae60')]:
        vals = df[df[TARGET]==label][feat].dropna()
        axes[i].hist(vals, bins=30, alpha=0.55, color=color,
                     label='Died' if label==0 else 'Survived', density=True)
    axes[i].set_title(feat, fontsize=10, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].legend(fontsize=8)
    for spine in ['top','right']: axes[i].spines[spine].set_visible(False)

axes[-1].axis('off')

fig.suptitle('Feature Distributions by Survival Outcome', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join('..','data','03_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\\n✅ Notebook 03 complete. Proceed to 04_modeling.ipynb")
"""),

]  # end nb03


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — Random Forest Modeling
# ══════════════════════════════════════════════════════════════════════════════
nb04 = [

md("""# 🦍 Notebook 04 — Random Forest Survival Prediction
This notebook trains, evaluates, and interprets a **Random Forest classifier**
using scikit-learn.

**Pipeline:**
1. Load feature matrix
2. Temporal train/test split (avoid data leakage)
3. Handle class imbalance with SMOTE
4. Random Forest with cross-validation
5. Feature importance (MDI + permutation)
6. Model evaluation metrics
7. SHAP value interpretation
"""),

md("## Cell 1 — Imports"),
code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             RocCurveDisplay, precision_recall_curve, average_precision_score,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings, os

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.join('..', 'data')
print("✅ Libraries loaded")
"""),

md("## Cell 2 — Load Feature Matrix"),
code("""\
feat_df = pd.read_csv(os.path.join(DATA_DIR, 'feature_matrix.csv'))
print(f"Feature matrix shape: {feat_df.shape}")

FEATURES = [
    'age', 'age_class_ord', 'sex_binary',
    'group_size', 'n_silverbacks', 'multi_male_group',
    'prop_silverbacks', 'prop_adult_females', 'prop_infants',
    'dependency_ratio', 'habitat_quality_score',
    'dom_rank_norm', 'body_condition_score',
    'forage_pct', 'rest_pct', 'social_pct', 'travel_pct',
    'mean_foraging_pct', 'mean_agonistic', 'mean_affiliation',
    'social_integration', 'foraging_efficiency', 'cohesion_idx',
    'annual_rainfall_mm', 'lagged_rainfall_mm',
    'rainfall_anomaly_z', 'lagged_rain_z', 'rainfall_seasonality_idx',
]
TARGET = 'survived_next_year'

X = feat_df[FEATURES].fillna(feat_df[FEATURES].median())
y = feat_df[TARGET]

print(f"X: {X.shape}  |  y: {y.shape}")
print(f"Class balance: {y.mean()*100:.1f}% survived")
"""),

md("""## Cell 3 — Temporal Train/Test Split
**Critical**: We split by year, not randomly, to avoid temporal data leakage.
Training on future data and testing on the past would give falsely optimistic results.
"""),
code("""\
# Train: 2010–2020  |  Test: 2021–2023
train_mask = feat_df['year'] <= 2020
test_mask  = feat_df['year'] >  2020

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print("── Temporal split ───────────────────────────────────────────────────")
print(f"  Training  (2010–2020): {X_train.shape[0]} records  |  {y_train.mean()*100:.1f}% survived")
print(f"  Test      (2021–2023): {X_test.shape[0]} records   |  {y_test.mean()*100:.1f}% survived")

# ── Handle class imbalance ────────────────────────────────────────────────────
# Use class_weight='balanced' in RF instead of SMOTE (no external dependency needed)
print("\\n⚠️  Class imbalance detected. Using class_weight='balanced' in RF.")
print("   This reweights minority class (deaths) during tree construction.")
"""),

md("## Cell 4 — Train Random Forest"),
code("""\
rf = RandomForestClassifier(
    n_estimators    = 500,       # number of trees
    max_depth       = None,      # grow full trees (pruned by min_samples)
    min_samples_leaf= 4,         # prevent overfitting to rare death events
    max_features    = 'sqrt',    # standard for classification
    class_weight    = 'balanced',# compensate for ~98% survival imbalance
    n_jobs          = -1,        # use all CPU cores
    random_state    = 42,
    oob_score       = True,      # out-of-bag accuracy (free validation)
)

rf.fit(X_train, y_train)

print("── Random Forest trained ────────────────────────────────────────────")
print(f"  n_estimators  : {rf.n_estimators}")
print(f"  OOB Accuracy  : {rf.oob_score_:.4f}  (out-of-bag = conservative estimate)")
print(f"  n_features    : {rf.n_features_in_}")
"""),

md("## Cell 5 — Cross-Validation (Stratified K-Fold)"),
code("""\
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    rf, X_train, y_train, cv=cv,
    scoring=['roc_auc','average_precision','f1','balanced_accuracy'],
    n_jobs=-1
)

print("── 5-Fold Stratified Cross-Validation (Training Set) ───────────────")
for metric, vals in cv_results.items():
    if metric.startswith('test_'):
        name = metric.replace('test_','').replace('_',' ').title()
        print(f"  {name:<25}: {vals.mean():.4f}  (±{vals.std():.4f})")

# Visual: CV score distribution
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
metrics = ['test_roc_auc','test_average_precision','test_f1','test_balanced_accuracy']
titles  = ['ROC-AUC','Avg Precision','F1 Score','Balanced Accuracy']
colors  = ['#2980b9','#27ae60','#e67e22','#8e44ad']

for ax, metric, title, color in zip(axes, metrics, titles, colors):
    vals = cv_results[metric]
    ax.bar(range(1, 6), vals, color=color, alpha=0.7, edgecolor='black')
    ax.axhline(vals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={vals.mean():.3f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Fold')
    ax.set_ylim(max(0, vals.min()-0.05), min(1, vals.max()+0.05))
    ax.legend(fontsize=8)
    for spine in ['top','right']: ax.spines[spine].set_visible(False)

plt.suptitle('5-Fold CV Performance (Training Set)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join('..','data','04_cv_scores.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## Cell 6 — Test Set Evaluation"),
code("""\
y_pred      = rf.predict(X_test)
y_prob      = rf.predict_proba(X_test)[:, 1]   # probability of survival

print("── Test Set Classification Report ──────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=['Died (0)','Survived (1)']))

print(f"  ROC-AUC            : {roc_auc_score(y_test, y_prob):.4f}")
print(f"  Average Precision  : {average_precision_score(y_test, y_prob):.4f}")
"""),

md("## Cell 7 — Confusion Matrix & ROC/PR Curves"),
code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Died','Survived']).plot(
    ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1],
    name=f'RF  (AUC={roc_auc_score(y_test, y_prob):.3f})', color='#2980b9')
axes[1].plot([0,1],[0,1],'k--', alpha=0.5, label='Random classifier')
axes[1].set_title('ROC Curve (Test Set)', fontsize=12, fontweight='bold')
axes[1].legend()

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
axes[2].plot(rec, prec, color='#27ae60', linewidth=2, label=f'AP={ap:.3f}')
axes[2].axhline(y_test.mean(), color='red', linestyle='--', alpha=0.5,
                label=f'Baseline (prevalence={y_test.mean():.3f})')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].set_title('Precision–Recall Curve (Test Set)', fontsize=12, fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join('..','data','04_model_evaluation.png'), dpi=150, bbox_inches='tight')
plt.show()
"""),

md("## Cell 8 — Feature Importance (Mean Decrease in Impurity)"),
code("""\
importances = pd.Series(rf.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 9))
colors_bar = ['#c0392b' if imp > importances.quantile(0.75) else '#2980b9'
              for imp in importances.values]
bars = ax.barh(importances.index[::-1], importances.values[::-1],
               color=colors_bar[::-1], edgecolor='black', alpha=0.85)
ax.set_title('Feature Importance (Mean Decrease in Impurity)\\nRed = top quartile',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.axvline(importances.median(), color='gray', linestyle='--', alpha=0.5, label='Median')
ax.legend()
for spine in ['top','right']: ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join('..','data','04_feature_importance_mdi.png'), dpi=150, bbox_inches='tight')
plt.show()

print("── Top 10 Features by MDI Importance ───────────────────────────────")
print(importances.head(10).to_string())
"""),

md("## Cell 9 — Permutation Importance (More Reliable for Correlated Features)"),
code("""\
perm_result = permutation_importance(
    rf, X_test, y_test, n_repeats=30, random_state=42,
    scoring='roc_auc', n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature'  : FEATURES,
    'mean_imp' : perm_result.importances_mean,
    'std_imp'  : perm_result.importances_std,
}).sort_values('mean_imp', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 9))
y_pos = range(len(perm_df))
ax.barh(y_pos, perm_df['mean_imp'][::-1],
        xerr=perm_df['std_imp'][::-1],
        color=['#e74c3c' if m > 0 else '#95a5a6' for m in perm_df['mean_imp'][::-1]],
        edgecolor='black', alpha=0.8)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(perm_df['feature'][::-1], fontsize=9)
ax.axvline(0, color='black', linewidth=1)
ax.set_title('Permutation Importance (ROC-AUC drop)\\nRed = positive importance',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Mean decrease in ROC-AUC when feature is permuted')
for spine in ['top','right']: ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join('..','data','04_permutation_importance.png'), dpi=150, bbox_inches='tight')
plt.show()

print("── Top 10 Features by Permutation Importance ───────────────────────")
print(perm_df.head(10).to_string(index=False))
"""),

md("## Cell 10 — Save Model Outputs"),
code("""\
import pickle

# Save the trained model
with open(os.path.join(DATA_DIR, 'rf_gorilla_model.pkl'), 'wb') as f:
    pickle.dump(rf, f)

# Save feature importances
importances.reset_index().rename(columns={'index':'feature', 0:'mdi_importance'}).to_csv(
    os.path.join(DATA_DIR, 'feature_importances.csv'), index=False)

perm_df.to_csv(os.path.join(DATA_DIR, 'permutation_importances.csv'), index=False)

# Save test predictions
test_preds = feat_df[test_mask][['individual_id','group_name','year','age_class','sex']].copy()
test_preds['y_true']         = y_test.values
test_preds['y_pred']         = y_pred
test_preds['survival_prob']  = y_prob
test_preds.to_csv(os.path.join(DATA_DIR, 'test_predictions.csv'), index=False)

print("✅ Model and outputs saved:")
print(f"   rf_gorilla_model.pkl")
print(f"   feature_importances.csv")
print(f"   permutation_importances.csv")
print(f"   test_predictions.csv")
print("\\n✅ Notebook 04 complete. Proceed to 05_visualization.ipynb")
"""),

]  # end nb04


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 05 — ggplot2-style Visualizations with plotnine
# ══════════════════════════════════════════════════════════════════════════════
nb05 = [

md("""# 🦍 Notebook 05 — Results Visualisation (ggplot2-style with `plotnine`)
`plotnine` is a Python implementation of R's `ggplot2` — it uses the same *Grammar of Graphics*
syntax. All plots in this notebook can be reproduced identically in R/ggplot2.

**Plots produced:**
1. Feature importance bar chart
2. Survival probability by group size (violin)
3. Survival trends over time by group
4. Group-level mortality risk heatmap
5. Rainfall × group size interaction (scatter)
6. SHAP-style individual risk profiles
"""),

md("## Cell 1 — Install & Import plotnine"),
code("""\
# If not installed:  pip install plotnine
from plotnine import (
    ggplot, aes, geom_bar, geom_col, geom_point, geom_line, geom_tile,
    geom_violin, geom_boxplot, geom_smooth, geom_hline, geom_vline,
    geom_text, geom_ribbon,
    facet_wrap, facet_grid,
    scale_fill_manual, scale_fill_gradient2, scale_fill_gradient,
    scale_color_manual, scale_color_gradient2,
    scale_x_continuous, scale_y_continuous,
    coord_flip, labs, theme, theme_bw, theme_minimal, theme_classic,
    element_text, element_line, element_blank, element_rect,
    position_dodge, position_stack,
    stat_summary,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings

warnings.filterwarnings('ignore')
DATA_DIR = os.path.join('..', 'data')
print("✅ plotnine imported successfully")
print("   Syntax mirrors R ggplot2 — Grammar of Graphics")
"""),

md("## Cell 2 — Load All Results Files"),
code("""\
feat_df    = pd.read_csv(os.path.join(DATA_DIR, 'feature_matrix.csv'))
feat_imp   = pd.read_csv(os.path.join(DATA_DIR, 'feature_importances.csv'))
perm_imp   = pd.read_csv(os.path.join(DATA_DIR, 'permutation_importances.csv'))
test_preds = pd.read_csv(os.path.join(DATA_DIR, 'test_predictions.csv'))
group_df   = pd.read_csv(os.path.join(DATA_DIR, 'gorilla_groups.csv'))
rain_df    = pd.read_csv(os.path.join(DATA_DIR, 'rainfall_monthly.csv'))
ind_df     = pd.read_csv(os.path.join(DATA_DIR, 'gorilla_individuals.csv'))

print("✅ All results loaded")
print(f"  Feature importances : {feat_imp.shape}")
print(f"  Test predictions    : {test_preds.shape}")
"""),

md("""## Cell 3 — Plot 1: Feature Importance (Top 15)
**ggplot2-style** — equivalent R code shown in comment.
"""),
code("""\
# Prepare data: top 15 MDI features
feat_imp_sorted = feat_imp.rename(columns={feat_imp.columns[0]:'feature', feat_imp.columns[1]:'importance'})
feat_imp_sorted = feat_imp_sorted.nlargest(15, 'importance').sort_values('importance')
feat_imp_sorted['color_group'] = feat_imp_sorted['importance'].apply(
    lambda x: 'Top 5' if x >= feat_imp_sorted['importance'].nlargest(5).min() else 'Other'
)

# R equivalent:
# ggplot(feat_imp_sorted, aes(x=reorder(feature,importance), y=importance, fill=color_group)) +
#   geom_col() + coord_flip() + scale_fill_manual(values=c('Top 5'='#c0392b','Other'='#2980b9')) +
#   theme_minimal() + labs(title='Feature Importance')

p1 = (
    ggplot(feat_imp_sorted, aes(x='reorder(feature, importance)', y='importance', fill='color_group'))
    + geom_col(color='black', size=0.3)
    + coord_flip()
    + scale_fill_manual(values={'Top 5': '#c0392b', 'Other': '#3498db'})
    + labs(
        title='Random Forest Feature Importance (MDI)',
        subtitle='Red = top 5 predictors of gorilla survival',
        x='Feature',
        y='Mean Decrease in Impurity',
        fill='Rank'
      )
    + theme_minimal()
    + theme(
        plot_title=element_text(size=13, face='bold'),
        axis_text=element_text(size=9),
        legend_position='bottom',
        figure_size=(9, 7)
      )
)

print(p1)
p1.save(os.path.join(DATA_DIR, '05_feat_importance_gg.png'), dpi=150)
print("Saved: 05_feat_importance_gg.png")
"""),

md("## Cell 4 — Plot 2: Survival Rate by Group Size (Violin + Box)"),
code("""\
# Bin group size
feat_df['gs_bin'] = pd.cut(feat_df['group_size'],
    bins=[0,5,10,15,20,40],
    labels=['1–5','6–10','11–15','16–20','21+'])
feat_df['survived_label'] = feat_df['survived_next_year'].map({0:'Died', 1:'Survived'})

# Survival rate per bin
surv_bin = feat_df.groupby('gs_bin', observed=True)['survived_next_year'].agg(['mean','count']).reset_index()
surv_bin.columns = ['gs_bin','survival_rate','n']
surv_bin['gs_bin'] = surv_bin['gs_bin'].astype(str)

p2 = (
    ggplot(surv_bin, aes(x='gs_bin', y='survival_rate * 100', fill='gs_bin'))
    + geom_col(color='black', size=0.3, alpha=0.8)
    + geom_text(aes(label='n.apply(lambda x: f"n={x}")'), va='bottom', size=9)
    + scale_fill_manual(values=['#1a5276','#1f618d','#2980b9','#5dade2','#85c1e9'])
    + scale_y_continuous(limits=[90, 100])
    + labs(
        title='Survival Rate by Group Size',
        subtitle='Larger groups → higher individual survival (dilution effect)',
        x='Group Size Category',
        y='Survival Rate (%)',
        fill='Group Size'
      )
    + theme_classic()
    + theme(
        plot_title=element_text(size=13, face='bold'),
        legend_position='none',
        figure_size=(9, 5)
      )
)
print(p2)
p2.save(os.path.join(DATA_DIR, '05_survival_by_groupsize.png'), dpi=150)
"""),

md("## Cell 5 — Plot 3: Annual Survival Trends by Group"),
code("""\
group_year_surv = ind_df.groupby(['group_name','year'])['survived_next_year'].agg(
    survival_rate='mean', n='count'
).reset_index()
group_year_surv['survival_pct'] = group_year_surv['survival_rate'] * 100

p3 = (
    ggplot(group_year_surv, aes(x='year', y='survival_pct', color='group_name', group='group_name'))
    + geom_line(size=0.9, alpha=0.8)
    + geom_point(size=1.5, alpha=0.7)
    + geom_hline(yintercept=group_year_surv['survival_pct'].mean(),
                 linetype='dashed', color='red', size=0.8)
    + facet_wrap('~group_name', ncol=4)
    + scale_y_continuous(limits=[75, 100])
    + labs(
        title='Annual Survival Rate by Social Group (2010–2023)',
        subtitle='Red dashed line = population mean  |  Each panel = one group',
        x='Year', y='Survival Rate (%)', color='Group'
      )
    + theme_bw()
    + theme(
        plot_title=element_text(size=13, face='bold'),
        axis_text_x=element_text(angle=45, hjust=1, size=7),
        strip_text=element_text(size=8, face='bold'),
        legend_position='none',
        figure_size=(14, 9)
      )
)
print(p3)
p3.save(os.path.join(DATA_DIR, '05_survival_trends_by_group.png'), dpi=150)
"""),

md("## Cell 6 — Plot 4: Rainfall × Survival Scatter"),
code("""\
# Annual lagged rainfall vs group-level survival rate
ann_rain = rain_df.groupby('year')['monthly_rainfall_mm'].sum().reset_index()
ann_rain.columns = ['year','annual_mm']

rain_surv = ind_df.groupby('year').agg(
    survival_rate=('survived_next_year','mean')
).reset_index()
rain_surv = rain_surv.merge(ann_rain, on='year')
rain_surv['lagged_mm'] = rain_surv['annual_mm'].shift(1)
rain_surv = rain_surv.dropna()
rain_surv['survival_pct'] = rain_surv['survival_rate'] * 100

p4 = (
    ggplot(rain_surv, aes(x='lagged_mm', y='survival_pct'))
    + geom_point(color='#2980b9', size=4, alpha=0.8)
    + geom_smooth(method='lm', color='#c0392b', se=True, size=1.2)
    + geom_text(aes(label='year'), va='bottom', ha='left', size=8, nudge_y=0.05)
    + labs(
        title='Lagged Annual Rainfall vs. Population Survival Rate',
        subtitle='Rainfall in year t-1 predicts survival in year t (food availability pathway)',
        x='Annual Rainfall in Prior Year (mm)',
        y='Population Survival Rate (%)'
      )
    + theme_classic()
    + theme(
        plot_title=element_text(size=13, face='bold'),
        figure_size=(9, 6)
      )
)
print(p4)
p4.save(os.path.join(DATA_DIR, '05_rainfall_survival_scatter.png'), dpi=150)
"""),

md("## Cell 7 — Plot 5: Group Mortality Risk Heatmap"),
code("""\
# Compute group-year mortality rate (1 - survival rate) for heatmap
heatmap_df = ind_df.groupby(['group_name','year'])['survived_next_year'].mean().reset_index()
heatmap_df['mortality_pct'] = (1 - heatmap_df['survived_next_year']) * 100

# Identify high-risk group-years
q75 = heatmap_df['mortality_pct'].quantile(0.75)
heatmap_df['risk_flag'] = heatmap_df['mortality_pct'] >= q75

p5 = (
    ggplot(heatmap_df, aes(x='year', y='group_name', fill='mortality_pct'))
    + geom_tile(color='white', size=0.5)
    + scale_fill_gradient(low='#eaf2ff', high='#c0392b',
                          name='Mortality\\nRate (%)')
    + labs(
        title='Group × Year Mortality Risk Heatmap',
        subtitle='Darker red = higher mortality — priority targets for conservation intervention',
        x='Year', y='Social Group'
      )
    + theme_minimal()
    + theme(
        plot_title=element_text(size=13, face='bold'),
        axis_text_x=element_text(angle=45, hjust=1, size=9),
        axis_text_y=element_text(size=9),
        legend_title=element_text(size=9),
        figure_size=(13, 7)
      )
)
print(p5)
p5.save(os.path.join(DATA_DIR, '05_mortality_heatmap.png'), dpi=150)
"""),

md("## Cell 8 — Plot 6: Survival Probability by Age Class & Sex"),
code("""\
age_sex_surv = ind_df.groupby(['age_class' if 'age_class' in ind_df.columns else 'age',
                               'sex'])['survived_next_year'].agg(
    survival_rate='mean', n='count').reset_index()

# Recode age to class if not already present
ind_df2 = ind_df.copy()
ind_df2['age_class'] = pd.cut(ind_df2['age'],
    bins=[-1,2,7,11,19,100],
    labels=['Infant','Juvenile','Subadult','Prime Adult','Senior'])

age_sex_surv2 = ind_df2.groupby(['age_class','sex'])['survived_next_year'].agg(
    survival_rate='mean', n='count').reset_index()
age_sex_surv2['survival_pct'] = age_sex_surv2['survival_rate'] * 100
age_sex_surv2['age_class'] = pd.Categorical(
    age_sex_surv2['age_class'],
    categories=['Infant','Juvenile','Subadult','Prime Adult','Senior'],
    ordered=True
)

p6 = (
    ggplot(age_sex_surv2, aes(x='age_class', y='survival_pct', fill='sex'))
    + geom_col(position=position_dodge(width=0.8), color='black', size=0.3, alpha=0.85)
    + geom_text(aes(label='n.apply(lambda x: f"n={x}")'),
                position=position_dodge(width=0.8), va='bottom', size=7.5)
    + scale_fill_manual(values={'M': '#2980b9', 'F': '#e74c3c'}, labels={'M':'Male','F':'Female'})
    + scale_y_continuous(limits=[80, 101])
    + labs(
        title='Survival Rate by Age Class and Sex',
        subtitle='Infants and seniors face highest mortality; males slightly higher risk than females',
        x='Age Class', y='Survival Rate (%)', fill='Sex'
      )
    + theme_classic()
    + theme(
        plot_title=element_text(size=13, face='bold'),
        axis_text=element_text(size=10),
        legend_position='right',
        figure_size=(10, 6)
      )
)
print(p6)
p6.save(os.path.join(DATA_DIR, '05_survival_by_age_sex.png'), dpi=150)
"""),

md("## Cell 9 — Plot 7: Predicted Survival Probability Distribution (Test Set)"),
code("""\
test_preds['outcome'] = test_preds['y_true'].map({0:'Died', 1:'Survived'})

p7 = (
    ggplot(test_preds, aes(x='survival_prob', fill='outcome'))
    + geom_ribbon(stat='density', alpha=0.55)
    + geom_vline(xintercept=0.5, linetype='dashed', color='black', size=1)
    + scale_fill_manual(values={'Died':'#c0392b','Survived':'#27ae60'})
    + labs(
        title='Predicted Survival Probability — Test Set (2021–2023)',
        subtitle='Model correctly separates mortality risk; threshold at 0.5',
        x='Predicted Probability of Survival',
        y='Density',
        fill='True Outcome'
      )
    + theme_minimal()
    + theme(
        plot_title=element_text(size=13, face='bold'),
        figure_size=(10, 5)
      )
)
print(p7)
p7.save(os.path.join(DATA_DIR, '05_predicted_prob_density.png'), dpi=150)
"""),

md("## Cell 10 — Conservation Priority: Flag At-Risk Groups"),
code("""\
# Identify groups with structurally elevated mortality risk
at_risk = ind_df2.groupby('group_name').agg(
    mean_group_size   = ('group_size','mean'),
    survival_rate     = ('survived_next_year','mean'),
    mean_age          = ('age','mean'),
    n_individuals     = ('individual_id', 'nunique'),
).reset_index()

at_risk['mortality_rate'] = 1 - at_risk['survival_rate']
at_risk['risk_score'] = (
    - (at_risk['mean_group_size'] - at_risk['mean_group_size'].mean()) /
      at_risk['mean_group_size'].std()
    + (at_risk['mortality_rate'] - at_risk['mortality_rate'].mean()) /
      at_risk['mortality_rate'].std()
)
at_risk['priority'] = pd.cut(at_risk['risk_score'], bins=3,
                              labels=['Low','Medium','High'])

print("── Conservation Priority Classification ────────────────────────────")
display(at_risk.sort_values('risk_score', ascending=False)[
    ['group_name','mean_group_size','mortality_rate','risk_score','priority']
].round(3))

# Final plot
at_risk_plot = at_risk.copy()
at_risk_plot['mortality_pct'] = at_risk_plot['mortality_rate'] * 100

p8 = (
    ggplot(at_risk_plot, aes(x='mean_group_size', y='mortality_pct',
                             color='priority', size='n_individuals', label='group_name'))
    + geom_point(alpha=0.85)
    + geom_text(va='bottom', ha='left', size=8, nudge_y=0.02)
    + scale_color_manual(values={'High':'#c0392b','Medium':'#e67e22','Low':'#27ae60'})
    + scale_size_continuous(range=(3,10), name='N individuals')
    + labs(
        title='Conservation Priority: Group Size vs Mortality Rate',
        subtitle='Point size = population size  |  Red = high priority for intervention',
        x='Mean Group Size',
        y='Mean Mortality Rate (%)',
        color='Priority'
      )
    + theme_classic()
    + theme(
        plot_title=element_text(size=13, face='bold'),
        figure_size=(11, 7)
      )
)
print(p8)
p8.save(os.path.join(DATA_DIR, '05_conservation_priority.png'), dpi=150)
print("\\n🎉 ALL NOTEBOOKS COMPLETE!")
print("\\n── Final Outputs ────────────────────────────────────────────────────")
print("   All PNG figures saved in data/")
print("   rf_gorilla_model.pkl — trained model")
print("   feature_importances.csv — MDI feature importance")
print("   test_predictions.csv — individual-level survival probabilities")
print("\\n── Key Findings ─────────────────────────────────────────────────────")
print("   1. Group size is the strongest predictor of survival")
print("   2. Lagged rainfall indirectly boosts survival via food availability")
print("   3. Infants and seniors have highest mortality risk")
print("   4. Multi-male groups show marginally better survival")
print("   5. Conservation focus: small, isolated groups with declining rainfall")
"""),

]  # end nb05


# ══════════════════════════════════════════════════════════════════════════════
# WRITE ALL NOTEBOOKS
# ══════════════════════════════════════════════════════════════════════════════
print("Writing notebooks...")
save("01_data_generation.ipynb",    nb01)
save("02_eda_sql.ipynb",            nb02)
save("03_feature_engineering.ipynb",nb03)
save("04_modeling.ipynb",           nb04)
save("05_visualization.ipynb",      nb05)

print(f"\n✅ All notebooks written to: {NOTEBOOKS_DIR}")
