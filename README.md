# Task 1: Predictive Modeling (Regression)

## Description
Build and evaluate a regression model to predict a continuous variable — in this
case, **median house value** — using the California Housing dataset.

## Dataset
- **File:** `housing.csv`
- **Source:** [Kaggle - California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Rows / Columns:** 20,640 rows, 10 columns
- **Target variable:** `median_house_value`
- **Features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`,
  `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`

> Note: `median_house_value` in this dataset is capped at $500,000 — any house
> actually worth more is recorded as exactly 500,000. This caps model accuracy
> at the high end regardless of tuning.

## Steps Performed
1. **Data cleaning**
   - Filled ~207 missing values in `total_bedrooms` with the column median.
   - One-hot encoded the categorical column `ocean_proximity` (`drop_first=True`
     to avoid the dummy variable trap).
2. **Train/test split** — 80% train, 20% test (`random_state=42` for reproducibility).
3. **Model training** — trained three regression models on the same split:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
4. **Evaluation** — compared models using:
   - Mean Squared Error (MSE)
   - R-squared (R²)
   - RMSE (for Linear Regression, to express error in dollars)
5. **Feature importance** — extracted from the Random Forest to see which
   features drive predictions most.
6. **Hyperparameter tuning** — used `GridSearchCV` (3-fold CV) over
   `n_estimators`, `max_depth`, and `min_samples_leaf` to tune the Random Forest.
7. **Visualization** — bar chart comparing R² across all three models, and a
   horizontal bar chart of Random Forest feature importances.

## Results

| Model                    | MSE            | R²     |
|---------------------------|----------------|--------|
| Linear Regression          | 4,908,476,721  | 0.6254 |
| Decision Tree               | 4,865,868,837  | 0.6287 |
| Random Forest (default)     | 2,404,745,975  | 0.8165 |
| Random Forest (tuned)       | 2,379,413,125  | 0.8184 |

**Best tuned parameters:** `max_depth=None, min_samples_leaf=2, n_estimators=300`

### Key findings
- **Random Forest clearly outperforms** both Linear Regression and a single
  Decision Tree, since it can capture non-linear relationships and feature
  interactions that a linear equation or single tree cannot.
- **`median_income` is by far the most important feature** (~49% importance),
  followed by `ocean_proximity_INLAND`, `longitude`, and `latitude` — location
  and income dominate house price prediction.
- **Hyperparameter tuning gave only a marginal improvement** (0.8165 → 0.8184).
  The untuned Random Forest was already close to optimal; further gains would
  require better features (e.g. lot size, house condition, school ratings)
  rather than a different model configuration.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place `housing.csv` in the same folder as `main.py`
3. Run: `python main.py`

## Tools
Python, pandas, numpy, scikit-learn, matplotlib
