"""
generate_dataset.py
====================
Generates four ecologically grounded synthetic CSV datasets for the
Mountain Gorilla Survival Prediction ML project.

Ecological parameters sourced from:
- Robbins et al. (2011). Gorilla beringei conservation assessment.
- Harcourt & Fossey (1981). The Virunga gorillas.
- Grueter et al. (2013). Multilevel societies in primates.
- Watts (1998). Seasonality of rainfall and food availability.

Run:   python generate_dataset.py
Output: 4 CSV files in the same directory as this script.
"""

import numpy as np
import pandas as pd
import os
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS  (ecologically grounded)
# ─────────────────────────────────────────────────────────────────────────────
N_GROUPS       = 12          # Number of social groups
STUDY_YEARS    = range(2010, 2024)  # 14-year longitudinal study
START_POP      = 110         # Starting number of individuals
GROUP_NAMES    = [
    "Susa", "Sabyinyo", "Amahoro", "Umubano", "Kwitonda",
    "Hirwa", "Agashya", "Bwenge", "Ntambara", "Ugenda",
    "Pablo", "Muhoza"
]

# Virunga bimodal rainfall pattern (mm) – monthly means ± noise
VIRUNGA_MONTHLY_RAIN = {   # Long-term monthly averages (mm)
    1: 55,  2: 75,  3: 120, 4: 160,  5: 130,  6: 30,
    7: 15,  8: 20,  9: 65,  10: 150, 11: 180, 12: 85
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE RAINFALL TABLE
# ─────────────────────────────────────────────────────────────────────────────
def generate_rainfall():
    records = []
    for year in STUDY_YEARS:
        for month in range(1, 13):
            base = VIRUNGA_MONTHLY_RAIN[month]
            # Interannual variability (~20% CV) + El Niño anomaly in 2015–2016
            anomaly = 1.25 if year in [2015, 2016] else 1.0
            rainfall = max(0, np.random.normal(base * anomaly, base * 0.18))
            records.append({
                "year": year,
                "month": month,
                "monthly_rainfall_mm": round(rainfall, 1),
                "season": "wet" if month in [3,4,5,9,10,11] else "dry"
            })
    df = pd.DataFrame(records)
    # Annual total for each year
    annual = df.groupby("year")["monthly_rainfall_mm"].sum().reset_index()
    annual.columns = ["year", "annual_rainfall_mm"]
    df = df.merge(annual, on="year")
    df["annual_rainfall_mm"] = df["annual_rainfall_mm"].round(1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE GROUPS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def generate_groups():
    records = []
    # Each group has a baseline size + natural variation over years
    group_baselines = {g: np.random.randint(5, 22) for g in GROUP_NAMES}
    group_silverbacks = {g: np.random.choice([1, 1, 1, 2, 2, 3]) for g in GROUP_NAMES}

    for year in STUDY_YEARS:
        for group_name in GROUP_NAMES:
            base_size = group_baselines[group_name]
            # Group size drifts stochastically over time (births, deaths, fissions)
            drift = int(np.random.normal(0, 1.5))
            group_size = max(2, min(38, base_size + drift))
            group_baselines[group_name] = group_size  # persistent

            n_silverbacks = group_silverbacks[group_name]
            n_blackbacks  = max(0, int(group_size * np.random.uniform(0.05, 0.15)))
            n_adult_female = max(1, int(group_size * np.random.uniform(0.35, 0.55)))
            n_subadult    = max(0, group_size - n_silverbacks - n_blackbacks - n_adult_female)
            n_infants     = max(0, int(n_adult_female * np.random.uniform(0.3, 0.7)))
            n_juveniles   = max(0, n_subadult - n_infants)

            # Home range quality (1–5 scale, stable per group)
            habitat_quality = round(np.random.uniform(1.5, 5.0), 2) if year == min(STUDY_YEARS) \
                              else None

            records.append({
                "group_id"          : GROUP_NAMES.index(group_name) + 1,
                "group_name"        : group_name,
                "year"              : year,
                "group_size"        : group_size,
                "n_silverbacks"     : n_silverbacks,
                "n_blackbacks"      : n_blackbacks,
                "n_adult_females"   : n_adult_female,
                "n_juveniles"       : n_juveniles,
                "n_infants"         : n_infants,
                "multi_male_group"  : int(n_silverbacks > 1),
            })

    df = pd.DataFrame(records)
    # Add stable habitat quality per group
    habitat_map = {g: round(np.random.uniform(2.0, 5.0), 2) for g in GROUP_NAMES}
    df["habitat_quality_score"] = df["group_name"].map(habitat_map)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATE INDIVIDUALS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def generate_individuals(groups_df):
    """
    Creates one row per individual per year (longitudinal panel).
    Survival is determined probabilistically using ecological drivers.
    """
    individual_id = 1
    records = []

    # Annual rainfall for lagged effect calculation
    rain_df = generate_rainfall()
    annual_rain = rain_df.groupby("year")["monthly_rainfall_mm"].sum()

    for group_name in GROUP_NAMES:
        gid = GROUP_NAMES.index(group_name) + 1

        # Seed individuals for this group at study start
        g_start = groups_df[(groups_df["group_name"] == group_name) &
                            (groups_df["year"] == min(STUDY_YEARS))].iloc[0]
        n_start = g_start["group_size"]

        # Create initial cohort
        active_individuals = []
        for _ in range(n_start):
            sex = np.random.choice(["M", "F"], p=[0.40, 0.60])
            probs = np.array([0.06,0.05,0.05,0.05,0.05,0.06,0.06,0.06,0.07,0.08,0.08,0.07,0.07,0.07,0.07])
            probs = probs / probs.sum()
            age = int(np.random.choice(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25],
                p=probs
            ))
            active_individuals.append({
                "individual_id": individual_id,
                "sex": sex,
                "birth_year": min(STUDY_YEARS) - age,
                "group_id": gid,
                "group_name": group_name,
            })
            individual_id += 1

        for year in STUDY_YEARS:
            g_row = groups_df[(groups_df["group_name"] == group_name) &
                              (groups_df["year"] == year)].iloc[0]
            group_size = g_row["group_size"]
            n_silverbacks = g_row["n_silverbacks"]

            # Lagged rainfall effect (prior year food availability)
            prior_rain = annual_rain.get(year - 1, annual_rain[min(STUDY_YEARS)])
            rain_z = (prior_rain - annual_rain.mean()) / annual_rain.std()  # standardised

            surviving_next_year = []
            for ind in active_individuals:
                age_this_year = year - ind["birth_year"]
                sex = ind["sex"]

                # ── Dominance rank (ordinal, lower = more dominant) ──────────
                # Silverbacks: rank 1–n_silverbacks (by age, roughly)
                if sex == "M" and age_this_year >= 12:
                    dom_rank = np.random.randint(1, max(2, n_silverbacks + 1))
                elif sex == "M":
                    dom_rank = np.random.randint(3, 8)
                else:
                    dom_rank = np.random.randint(1, max(2, int(g_row["n_adult_females"]) + 1))

                # ── Behavioural observations (daily % budget, sums to ~100) ──
                forage_pct = round(np.random.normal(45, 8), 1)   # % time foraging
                rest_pct   = round(np.random.normal(30, 6), 1)   # % time resting
                social_pct = round(np.random.normal(15, 4), 1)   # % social
                travel_pct = max(0, round(100 - forage_pct - rest_pct - social_pct, 1))

                # ── Survival probability (ecological model) ───────────────────
                # Base rates by age class (Robbins et al.)
                if age_this_year < 1:
                    base_surv = 0.72
                elif age_this_year < 3:
                    base_surv = 0.82
                elif age_this_year < 8:
                    base_surv = 0.94
                elif age_this_year < 20:
                    base_surv = 0.96
                else:
                    base_surv = 0.88  # senescence

                # Group size effect: protective (dilution, cooperative defence)
                group_effect = 0.008 * min(group_size, 20)

                # Multi-male bonus (more vigilant groups)
                multimale_bonus = 0.015 if g_row["multi_male_group"] else 0.0

                # Dominance rank penalty for low-rank males (stress, injury)
                rank_penalty = -0.005 * dom_rank if sex == "M" else 0.0

                # Rainfall (lagged 1 yr): wet years → more food → better survival
                rain_effect = 0.018 * rain_z

                # Habitat quality
                hab_effect = 0.01 * g_row["habitat_quality_score"]

                # Sex penalty (males have higher mortality from injuries/poaching)
                sex_effect = -0.02 if sex == "M" else 0.0

                survival_prob = np.clip(
                    base_surv + group_effect + multimale_bonus +
                    rank_penalty + rain_effect + hab_effect + sex_effect,
                    0.50, 0.999
                )

                survived = int(np.random.random() < survival_prob)

                # Body condition score (proxy; correlates with forage, rainfall)
                body_condition = round(
                    np.clip(np.random.normal(3.0 + 0.3*rain_z + 0.05*forage_pct/10, 0.5), 1, 5),
                    2
                )

                records.append({
                    "individual_id"       : ind["individual_id"],
                    "group_id"            : gid,
                    "group_name"          : group_name,
                    "year"                : year,
                    "age"                 : age_this_year,
                    "sex"                 : sex,
                    "dominance_rank"      : dom_rank,
                    "forage_pct"          : forage_pct,
                    "rest_pct"            : rest_pct,
                    "social_pct"          : social_pct,
                    "travel_pct"          : travel_pct,
                    "body_condition_score": body_condition,
                    "group_size"          : group_size,
                    "n_silverbacks"       : n_silverbacks,
                    "multi_male_group"    : int(g_row["multi_male_group"]),
                    "habitat_quality"     : g_row["habitat_quality_score"],
                    "survived_next_year"  : survived,   # TARGET VARIABLE
                    "annual_rainfall_mm"  : round(float(annual_rain.get(year, 1800)), 1),
                    "lagged_rainfall_mm"  : round(float(prior_rain), 1),
                })

                if survived:
                    surviving_next_year.append({**ind, "birth_year": ind["birth_year"]})

            # Add births (~0.35 per adult female per year)
            n_births = max(0, int(np.random.poisson(g_row["n_adult_females"] * 0.35)))
            for _ in range(n_births):
                surviving_next_year.append({
                    "individual_id": individual_id,
                    "sex": np.random.choice(["M","F"]),
                    "birth_year": year + 1,
                    "group_id": gid,
                    "group_name": group_name,
                })
                individual_id += 1

            active_individuals = surviving_next_year

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERATE OBSERVATIONS TABLE (session-level behavioural records)
# ─────────────────────────────────────────────────────────────────────────────
def generate_observations(individuals_df):
    """
    Creates 3–6 focal observation sessions per individual per year.
    Mirrors real field data structure (focal animal sampling).
    """
    obs_id = 1
    records = []
    sample = individuals_df.sample(frac=0.6, random_state=42)  # not every ind every session

    for _, row in sample.iterrows():
        n_sessions = np.random.randint(3, 7)
        for _ in range(n_sessions):
            # Session date within the study year
            day_of_year = np.random.randint(1, 365)
            obs_date = datetime(row["year"], 1, 1) + timedelta(days=day_of_year)
            month = obs_date.month
            season = "wet" if month in [3,4,5,9,10,11] else "dry"

            # Behavioural rates vary with season
            foraging_rate = round(np.random.normal(
                row["forage_pct"] + (5 if season == "wet" else -5), 5), 1)
            agonistic_events = max(0, int(np.random.poisson(
                1.5 if row["sex"] == "M" else 0.5)))
            affiliation_events = max(0, int(np.random.poisson(
                2.5 if row["sex"] == "F" else 1.2)))
            nearest_neighbour_dist_m = round(np.random.exponential(8), 1)

            records.append({
                "obs_id"                   : obs_id,
                "individual_id"            : row["individual_id"],
                "group_id"                 : row["group_id"],
                "obs_date"                 : obs_date.strftime("%Y-%m-%d"),
                "year"                     : row["year"],
                "month"                    : month,
                "season"                   : season,
                "session_duration_min"     : int(np.random.normal(55, 12)),
                "foraging_rate_pct"        : np.clip(foraging_rate, 10, 80),
                "agonistic_events"         : agonistic_events,
                "affiliation_events"       : affiliation_events,
                "nearest_neighbour_dist_m" : nearest_neighbour_dist_m,
                "observer_id"              : f"OBS_{np.random.randint(1,8):02d}",
            })
            obs_id += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🦍 Generating Mountain Gorilla Survival Dataset...")
    print("   Parameters grounded in Virunga population research.\n")

    print("  [1/4] Generating rainfall table...")
    rain_df = generate_rainfall()
    rain_df.to_csv(os.path.join(OUTPUT_DIR, "rainfall_monthly.csv"), index=False)
    print(f"        → rainfall_monthly.csv  ({len(rain_df)} rows)")

    print("  [2/4] Generating group-level data...")
    groups_df = generate_groups()
    groups_df.to_csv(os.path.join(OUTPUT_DIR, "gorilla_groups.csv"), index=False)
    print(f"        → gorilla_groups.csv    ({len(groups_df)} rows)")

    print("  [3/4] Generating individual longitudinal records...")
    ind_df = generate_individuals(groups_df)
    ind_df.to_csv(os.path.join(OUTPUT_DIR, "gorilla_individuals.csv"), index=False)
    print(f"        → gorilla_individuals.csv ({len(ind_df)} rows, {ind_df['individual_id'].nunique()} unique individuals)")

    print("  [4/4] Generating focal observation sessions...")
    obs_df = generate_observations(ind_df)
    obs_df.to_csv(os.path.join(OUTPUT_DIR, "gorilla_observations.csv"), index=False)
    print(f"        → gorilla_observations.csv ({len(obs_df)} rows)\n")

    print("✅ All datasets generated successfully!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("\n── Dataset Summary ──────────────────────────────────────")
    print(f"   Study period    : {min(STUDY_YEARS)}–{max(STUDY_YEARS)} ({len(list(STUDY_YEARS))} years)")
    print(f"   Social groups   : {N_GROUPS}")
    print(f"   Unique individuals tracked: {ind_df['individual_id'].nunique()}")
    print(f"   Survival rate   : {ind_df['survived_next_year'].mean():.1%}")
    print(f"   Observation sessions: {len(obs_df)}")
    print("─────────────────────────────────────────────────────────")
