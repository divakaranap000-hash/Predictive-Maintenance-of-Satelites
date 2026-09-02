# Satellite Predictive Maintenance

Predicting satellite component failure before it happens, using telemetry data from a 10,000-unit fleet. This project covers data cleaning and EDA, training and comparing five classification models on a strongly imbalanced target, and a Power BI dashboard for exploring failure patterns by satellite type, subsystem, and operating hours.

## Problem statement

Satellites report continuous telemetry (temperature, reaction wheel speed, torque, operating hours) but only a small fraction ever fail — 339 out of 10,000 in this dataset, a 3.4% failure rate. The goal is to flag likely failures early enough to act on them, which means treating this as a classification problem where **recall on the failure class matters more than overall accuracy** — a model that predicts "healthy" for everything would already be 96.6% accurate and completely useless.

## Dataset

`cleaned_satellite_data_xls.xlsx` — 10,000 rows, 12 columns:

| Column | Description |
|---|---|
| Satellite Type | Fleet class: L (6,000), M (2,997), H (1,003) |
| Internal Temperature (°C) | Internal housing temperature |
| Payload Temperature (°C) | Payload compartment temperature |
| Reaction Wheel Speed (RPM) | Attitude-control wheel speed |
| Reaction Wheel Torque (Nm) | Attitude-control wheel torque |
| Component Operating Hours | Hours run at time of record |
| Satellite Failure | Target: 1 = failed, 0 = healthy |
| Solar Panel Failure | Subsystem flag |
| Thermal Control Failure | Subsystem flag |
| Power System Failure | Subsystem flag |
| Attitude Control Failure | Subsystem flag |
| Communication System Failure | Subsystem flag |

## Methodology

1. Cleaned and typed the raw data, checked for nulls and outliers.
2. Explored failure rates across satellite type, subsystem, and operating-hour bands.
3. Trained five classifiers on the same train/test split: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
4. Evaluated every model on **accuracy, precision, recall, and F1 for the failure class specifically** — the metric that actually reflects usefulness on an imbalanced target.
5. Exported a model comparison table to CSV and visualized it, alongside the fleet-level failure data, in a Power BI dashboard.

## Model comparison

| Model | Accuracy | Precision (Failure) | Recall (Failure) | F1 (Failure) |
|---|---|---|---|---|
| Logistic Regression | 0.8205 | 0.14 | 0.82 | 0.24 |
| Decision Tree | 0.9780 | 0.71 | 0.60 | 0.65 |
| Random Forest (tuned) | 0.9740 | 0.59 | 0.78 | 0.67 |
| **Gradient Boosting** | 0.9850 | 0.87 | 0.66 | **0.75** |
| XGBoost | 0.9565 | 0.43 | **0.87** | 0.58 |

Gradient Boosting gives the best overall balance (F1 = 0.75), while XGBoost trades precision for the highest recall (0.87) — catching more real failures at the cost of more false alarms. Which one is "better" depends on whether a missed failure or a false alert is more expensive in the deployment context.

## Key findings

- **Failures cluster by age, not randomly.** Failed satellites ran for an average of 143.8 operating hours before failing, versus 106.7 hours for the healthy fleet.
- **There's a clear failure spike between 190–230 operating hours**, well above the baseline rate seen earlier in a satellite's life — a strong candidate for a proactive maintenance window rather than reactive repair.
- **Thermal control (115) and attitude control (98) are the most failure-prone subsystems**, well ahead of power system (95), solar panel (46), and communications (19).
- Satellite type L has the highest raw failure count (235) but that tracks its much larger fleet share (6,000 of 10,000 units); failure *rate* is comparable across L, M, and H.

## Dashboard

Built in Power BI, covering:
- KPI cards (total satellites, total failures, failure rate, average torque/RPM)
- Donut charts of healthy vs. failed satellites, one per satellite type (L, M, H)
- Subsystem failure breakdown (horizontal bar chart)
- A line chart of failures across 20-hour operating bands, filterable by satellite type
- The model comparison table above, rendered as a Power BI table visual



## Tech stack

- **Python** — pandas, scikit-learn, XGBoost, matplotlib/seaborn for EDA
- **Jupyter Notebook** — analysis and modeling (`satellite_predictive_maintenance.ipynb`)
- **Power BI** — interactive dashboard and reporting
- **Excel** — source dataset format

## Project structure

```
.
├── cleaned_satellite_data_xls.xlsx      # source dataset
├── satellite_predictive_maintenance.ipynb   # EDA, modeling, evaluation
├── model_comparison.csv                 # exported model metrics table
├── dashboard/                           # Power BI file and/or screenshot
└── README.md
```

## How to run

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook satellite_predictive_maintenance.ipynb
```

Run all cells top to bottom — the final cells train and evaluate all five models and export `model_comparison.csv`. Open the Power BI file separately to explore the dashboard.

## Future work

- Feed predicted failure probabilities back into the dashboard as a live risk score per satellite.
- Test cost-sensitive learning or threshold tuning to explicitly trade off precision vs. recall based on a real maintenance-cost estimate.
- Extend the operating-hours analysis with confidence intervals, since the 190–230 hour spike sits on a smaller surviving sample than earlier bands.

## Author

Divakaran A P — Data Analyst
