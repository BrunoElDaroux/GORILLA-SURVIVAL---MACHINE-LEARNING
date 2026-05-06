# 🦍 Machine Learning for Survival Prediction in Mountain Gorilla Populations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BrunoElDaroux/gorilla-survival-ml/main)


## Project Overview
A full end-to-end machine learning pipeline predicting **individual survival outcomes** in mountain gorilla (*Gorilla beringei beringei*) populations using behavioral, social, and environmental variables collected longitudinally across gorilla groups.


## Research Questions

1. Can individual survival outcomes in mountain gorillas be predicted from behavioral, social, and environmental variables collected longitudinally?
2. Which variables are the strongest predictors of survival — and are those findings consistent with ecological theory on group-living in social mammals?
3. Which social groups are demographically at-risk and should be prioritised for conservation intervention?


## Key Findings Summary

1. `Group structure:` Group size was the strongest predictor of individual survival. Larger groups confer a dilution effect and cooperative defence advantage — consistent with Harcourt & Fossey (1981) and broader social mammal theory.
2. `Environment:` Lagged annual rainfall (prior year) positively predicted survival, operating indirectly through food availability. El Niño years (2015–2016) showed elevated mortality, especially in smaller groups.
3. `Demographics:` Infants and seniors faced the highest mortality risk. Males had slightly lower survival than females, driven by injury and dominance contest costs.
Group composition: Multi-male groups showed marginally better survival, likely reflecting enhanced vigilance and predator deterrence.
4. `Model performance:` The Random Forest achieved strong ROC-AUC under temporal cross-validation, though the severe class imbalance (~98% survival) means raw accuracy is misleading — balanced metrics matter here.
5. `Conservation output:` Small, isolated groups with declining group size trajectories were flagged as highest priority for intervention.

---

## Tools & Stack
| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas / numpy | Data manipulation |
| sqlite3 / pandasql | SQL queries in Python |
| scikit-learn | Random Forest classifier |
| plotnine | ggplot2-style visualizations in Python |
| matplotlib / seaborn | Supplementary plots |
| Jupyter Notebook | All analysis (VS Code compatible) |

---

## Project Structure
```
gorilla_survival_ml/
├── README.md
├── requirements.txt
├── data/
│   └── generate_dataset.py          ← Run this FIRST to create all CSVs
│       gorilla_individuals.csv       ← Auto-generated
│       gorilla_groups.csv            ← Auto-generated
│       gorilla_observations.csv      ← Auto-generated
│       rainfall_monthly.csv          ← Auto-generated
│       analytical_dataset.csv        ← Auto-generated (final ML-ready data)
├── notebooks/
│   ├── 01_data_generation.ipynb      ← Generate & inspect datasets
│   ├── 02_eda_sql.ipynb              ← SQL joins + Exploratory Data Analysis
│   ├── 03_feature_engineering.ipynb  ← Build ML-ready feature matrix
│   ├── 04_modeling.ipynb             ← Random Forest + evaluation
│   └── 05_visualization.ipynb        ← ggplot2-style results & plots
└── src/
    ├── data_utils.py                 ← Reusable data functions
    └── model_utils.py                ← Reusable modeling functions
```

---

## How to Run (VS Code)

### 1. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the datasets
```bash
cd data
python generate_dataset.py
```

### 4. Open notebooks in VS Code
- Install the **Jupyter extension** in VS Code
- Open notebooks in order: `01 → 02 → 03 → 04 → 05`
- Select your venv as the kernel

---

## Ecological Background
Parameters are grounded in published literature:
- **Survival rates**: Adults ~95%/yr, Infants ~82%/yr (Robbins et al., 2011)
- **Group size**: Ranges 2–38, mean ~10 (Harcourt & Fossey, 1981)
- **Rainfall**: Virunga Massif bimodal pattern ~1800mm/yr
- **Key predictors**: Group size (predator dilution, cooperative defense), rainfall (food availability lag), dominance rank (resource access)

---

## ⚠️ Dataset Source Note
> Since no fully open-access individual-level gorilla longitudinal survival CSV exists publicly, this project uses a **synthetic dataset generated with `generate_dataset.py`** using ecological parameters from peer-reviewed literature. The structure mirrors real datasets used by the Dian Fossey Gorilla Fund and similar field stations.
>
> For real gorilla data, apply at: [https://gorillafund.org/science/](https://gorillafund.org/what-we-do/scientific-research/)

---

## Key Results
- **Top predictor**: Group size (larger groups → higher survival)
- **Environmental**: Annual rainfall → food availability → survival (indirect path)
- **Demographic**: Age class and sex modulate baseline survival
- **Actionable**: Demographically small/isolated groups flagged for priority intervention


## Author

*Bioinformatician with research experience at the Dian Fossey Gorilla Fund, building end-to-end computational pipelines across four domains: spatial movement ecology (GeoPandas, KDE, permutation testing), population genetics (CERVUS microsatellite LOD scoring, Queller-Goodnight kinship estimation), machine learning survival analysis (Random Forest, temporal cross-validation), and conservation epidemiology (logistic regression, SciPy hypothesis testing, temporal linkage). Technical stack: Python · R · SQL · scikit-learn · SciPy · GeoPandas · Git. All work is grounded in longitudinal biological datasets with direct conservation policy implications across the Virunga Massif — Rwanda, Uganda, and DRC.*
