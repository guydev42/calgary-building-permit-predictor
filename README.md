# Building permit cost predictor

## Problem statement
Construction stakeholders in Calgary -- homeowners, developers, and city planners -- need reliable cost estimates early in the planning process. Budget overruns remain one of the biggest challenges in construction. This project uses 484K+ historical building permits and machine learning to predict project costs from permit characteristics, location, and scope.

## Approach
- Fetched 484,000+ permits from the Calgary Open Data API (Socrata)
- Cleaned and log-transformed the heavily right-skewed cost distribution
- Engineered community-level aggregates (avg/median cost, permit counts) and temporal features
- Trained and compared Ridge, Random Forest, Gradient Boosting, and XGBoost regressors
- Built an interactive Streamlit dashboard with a cost predictor and community explorer

## Key results

| Metric | XGBoost | Gradient Boosting | Random Forest |
|--------|---------|-------------------|---------------|
| R-squared | **~0.89** | ~0.85 | ~0.82 |
| MAE ($) | ~30,000 | ~32,000 | ~35,000 |
| RMSE ($) | ~80,000 | ~85,000 | ~90,000 |

## How to run
```bash
pip install -r requirements.txt
python src/data_loader.py    # fetch data from Calgary Open Data
streamlit run app.py         # launch dashboard
```

## Project structure
```
project_01_building_permit_cost_predictor/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching & feature engineering
    └── model.py            # Model training & evaluation
```

## Technical stack
pandas, NumPy, scikit-learn, XGBoost, Plotly, Streamlit, sodapy
