# House Price Prediction using Regression & Pipeline

This project demonstrates an **end-to-end regression workflow** for predicting the **median house value of California neighbourhoods** (census block groups) from income, house age, rooms, population and location. It compares **Linear Regression, Ridge, Random Forest and Gradient Boosting**, tests whether **feature engineering** really helps, and checks how well the model works in **areas it has never seen**.

It follows **industrial best practices** by:

- Putting **feature engineering and scaling inside a Pipeline**, so outlier limits are learned from training data only (no data leakage)
- Giving every algorithm the **same cross-validation folds and a similar tuning effort**
- Choosing the final model with **cross-validation on training data only**, so the test set is never used to pick a winner
- **Proving** the value of feature engineering by re-scoring every model with and without it
- Reporting **RMSE, MAE and R-squared**, with a confidence interval and an **error analysis by price band**
- Running a **"new area" check**, because nearby neighbourhoods have similar prices
- Saving the trained pipeline with **Joblib** and reusing it from a separate **prediction script**
- Saving plots, tables and metrics to files for **reproducibility**

---

## Dependencies

Requires **Python 3.9+** (tested on Python 3.12). Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

or directly:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## Dataset Information

**Source:** `california_housing.csv`, the California Housing dataset as shipped with scikit-learn, built from the **1990 US census** (20,640 neighbourhoods, 8 features, no missing values, no duplicate rows)

**Features (8):**

| # | Feature | Description |
|---|---------|-------------|
| 1 | MedInc | Median income in the area, in units of $10,000 |
| 2 | HouseAge | Median age of the houses, in years |
| 3 | AveRooms | Average number of rooms per household |
| 4 | AveBedrms | Average number of bedrooms per household |
| 5 | Population | Number of people living in the area |
| 6 | AveOccup | Average number of people per household |
| 7 | Latitude | Latitude of the area |
| 8 | Longitude | Longitude of the area |

**Target:** `target` is the median house value in units of **$100,000** (so 2.5 means $250,000).

**Things to know about this data:**

- **The price is capped.** 992 areas (4.8%) sit exactly at 5.0, which means "$500,000 or more". The true prices above the cap are unknown, so the model cannot learn them.
- **Extreme outliers exist.** For example `AveOccup` reaches 1,243 people per household and `AveRooms` reaches 142. These are handled inside the pipeline (see below).
- **Income is the strongest single feature** (correlation 0.69 with price). Location matters a lot too, but its link with price is not a straight line, so a plain correlation hides it.
- This is 1990 data about **neighbourhoods**, not single houses, and not today's prices.

---

## Workflow

### Data Preparation

- Load the CSV and check columns, missing values and impossible values (negative prices, zero rooms)
- Explore the price distribution, a price map, and correlations

### Train-Test Split

- Random split into **80% train (16,512 areas)** and **20% test (4,128 areas)**

### Pipeline Construction

- **Step 1:** `FeatureEngineer` (in `house_features.py`)
  - Clips extreme outliers in `AveRooms`, `AveBedrms`, `AveOccup` and `Population` at the 0.5th and 99.5th percentiles of the training data
  - Adds **6 engineered features**: `BedroomRatio`, `RoomsPerPerson`, `LogPopulation`, `DistToNearestCity_km`, `DistToSF_km`, `DistToLA_km` (distances to 5 large cities)
- **Step 2:** `StandardScaler`
- **Step 3:** the regressor

### Models Compared

| Model | Tuning (5-fold CV on training data) |
|-------|-------------------------------------|
| Linear Regression | none (baseline) |
| Ridge | `alpha` in 0.1, 1, 10, 100 |
| Random Forest (150 trees) | fixed settings |
| Gradient Boosting (`HistGradientBoostingRegressor`) | `learning_rate` in 0.05, 0.1 and `max_leaf_nodes` in 31, 63 |

An "always predict the average price" baseline is also reported.

### Model Selection

- The final model has the lowest **cross-validated RMSE**
- The test set is evaluated **once**, after the choice is made

### Extra Checks

- **Does feature engineering help?** Every model is re-scored with the same settings and folds, with and without the engineered features.
- **Error analysis:** error by price band, including the capped areas, plus a 95% interval for the test RMSE.
- **New-area check:** the map is cut into 20 areas, and whole areas are held out together, to test how the model does in a place it has not seen.

### Model Saving & Loading

- Save the full pipeline with **Joblib** (`joblib.dump`)
- Save a **metadata file** (expected columns, training ranges, units, library versions, metrics)
- Load both in a separate script (`joblib.load`) and predict without retraining

---

## Running the Project

```bash
python house_price_prediction.py
```

The run takes about 4 minutes on my test machine (222 seconds), and longer on a slower PC. Expected output (shortened):

```text
Dataset Shape :: (20640, 9)
Houses at the price cap (5.0 = $500,000 or more) :: 992 (4.8%)
RMSE of always predicting the average price :: 1.154 (= $115,393)
Train_x Shape ::  (16512, 8)
Test_x Shape  ::  (4128, 8)

  Linear Regression  defaults                      CV RMSE 0.6434 (+/- 0.0134) | R2 0.690
  Ridge              alpha=1                       CV RMSE 0.6434 (+/- 0.0134) | R2 0.690
  Random Forest      defaults                      CV RMSE 0.4664 (+/- 0.0097) | R2 0.837
  Gradient Boosting  learning_rate=0.05, ...       CV RMSE 0.4471 (+/- 0.0107) | R2 0.850

Final model (lowest CV RMSE) :: Gradient Boosting (CV RMSE 0.4471)
Test RMSE :: 0.4409 (95% interval 0.422 to 0.461) = about $44,092
...
Model saved to outputs/house_price_model.joblib
```

**Generated files** (in `outputs/`):

```text
outputs/
├── figures/
│   ├── eda_overview.png
│   ├── model_comparison.png
│   ├── feature_engineering_check.png
│   ├── predictions_and_errors.png
│   └── feature_importance.png
├── model_comparison.csv
├── feature_engineering_check.csv
├── metrics.json
├── model_metadata.json
└── house_price_model.joblib
```

---

## Results

All numbers are from `random_state=42`. Prices are in units of $100,000, so an RMSE of 0.44 means a typical error of about $44,000.

**Model comparison:**

| Model | CV RMSE | CV R2 | Train RMSE | Test RMSE | Test MAE | Test R2 |
|-------|---------|-------|------------|-----------|----------|---------|
| Always predict the average | - | - | - | 1.145 | - | 0.000 |
| Linear Regression | 0.643 | 0.690 | 0.643 | 0.653 | 0.473 | 0.674 |
| Ridge (alpha = 1) | 0.643 | 0.690 | 0.643 | 0.653 | 0.473 | 0.674 |
| Random Forest | 0.466 | 0.837 | 0.219 | 0.458 | 0.292 | 0.840 |
| **Gradient Boosting (selected)** | **0.447** | **0.850** | 0.318 | **0.441** | **0.286** | **0.852** |

The selected model's test RMSE is **0.441 (about $44,100)** with a 95% interval of 0.422 to 0.461, and its test MAE is about **$28,600**. Selected settings: `learning_rate=0.05`, `max_leaf_nodes=63`.

**Does feature engineering help?** (cross-validated RMSE, same settings and folds):

| Model | 8 raw features | Raw + 6 engineered | Improvement |
|-------|----------------|--------------------|-------------|
| Linear Regression | 0.721 | 0.643 | 10.7% |
| Ridge | 0.721 | 0.643 | 10.7% |
| Random Forest | 0.501 | 0.466 | 6.9% |
| Gradient Boosting | 0.460 | 0.447 | 2.8% |

**Error by price band (selected model, test set):**

| Price band | Areas | MAE | Average error (predicted minus actual) |
|------------|-------|-----|----------------------------------------|
| under $150k | 1,516 | 0.195 | +0.127 (too high) |
| $150k to $250k | 1,487 | 0.241 | +0.045 |
| $250k to $350k | 620 | 0.373 | -0.069 |
| $350k to $500k | 321 | 0.581 | -0.363 (too low) |
| capped at $500k | 184 | 0.590 | -0.548 (too low) |

**New-area check (selected model, cross-validation on training data):**

| How the data is split | RMSE | R2 |
|-----------------------|------|----|
| Random split (neighbours in both train and test) | 0.447 | 0.850 |
| Whole map areas held out | 0.719 | 0.593 |

**Feature importance** (increase in test RMSE when one column is shuffled, in $100,000):

| Latitude | Longitude | MedInc | AveOccup | AveRooms | HouseAge | AveBedrms | Population |
|----------|-----------|--------|----------|----------|----------|-----------|------------|
| 0.553 | 0.527 | 0.380 | 0.187 | 0.121 | 0.048 | 0.011 | 0.009 |

**Key findings:**

1. **Tree models beat linear models by a wide margin.** Gradient Boosting reaches R2 0.85 against 0.67 for Linear Regression, because the link between location, income and price is not a straight line. Test results match the cross-validation results (0.441 against 0.447), so the models are not just memorising the training data.
2. **Ridge is no better than plain Linear Regression.** With only 8 features and 16,000 rows there is little to regularise.
3. **Feature engineering helps, most for the simplest model.** It cuts the error of linear models by 10.7%, Random Forest by 6.9% and Gradient Boosting by 2.8%. Boosting can already find some of those patterns by itself.
4. **Location is the biggest driver.** Latitude and longitude matter most, then income. This also explains the next finding.
5. **The model is much weaker in places it has not seen.** When whole map areas are held out, R2 drops from 0.85 to 0.59 and RMSE rises from 0.447 to 0.719. A normal random split makes the model look better than it would be in a new region, because it learns from the neighbours of every test area. Use the 0.59 as the more honest number for a new location.
6. **Expensive areas are underestimated.** The model is too low for areas above $350,000 (by $36,000 to $55,000 on average), partly because of the price cap and partly because there are fewer of them to learn from. Cheap areas are slightly overestimated.

**Limits:** the data is from 1990 and describes neighbourhoods, prices are capped at $500,000, and the random split flatters the model (see finding 5). This is **not a valuation tool**.

---

## Visualizations

- **Exploratory overview (price distribution, price map, income vs price):** `outputs/figures/eda_overview.png`
- **Model comparison:** `outputs/figures/model_comparison.png`
- **Feature engineering check:** `outputs/figures/feature_engineering_check.png`
- **Predicted vs actual, residuals, error by price band:** `outputs/figures/predictions_and_errors.png`
- **Feature importance:** `outputs/figures/feature_importance.png`

Set `SHOW_PLOTS = True` at the top of the script to also open the plots in a window.

---

## Model Persistence with Joblib (Train Once, Use Many Times)

**The idea:** training takes minutes and needs the full dataset. Predicting a price should take milliseconds and should not need either. So the trained model is written to a file with `joblib.dump`, and any other program can bring it back with `joblib.load` and call `predict`.

```text
 TRAINING (run once)                          PREDICTION (run any time)
 house_price_prediction.py                    predict_price.py
 ┌──────────────────────┐                     ┌──────────────────────────┐
 │ california_housing   │                     │ new_houses.csv           │
 │ tune + compare models│                     │ load model + metadata    │
 │ joblib.dump(model)   │ ──► .joblib  ──►    │ check columns and ranges │
 │ write metadata.json  │ ──► .json    ──►    │ predict                  │
 └──────────────────────┘                     └──────────────────────────┘
```

**What is stored:**

- `outputs/house_price_model.joblib`: the **whole pipeline** (feature engineering with its learned outlier limits, the fitted scaler, and the trained regressor), so new data gets exactly the same preparation as the training data
- `outputs/model_metadata.json`: the 8 expected columns, the value ranges seen in training, the price unit and cap, library versions and the test metrics

**Why `house_features.py` is a separate file:** the saved model contains a `FeatureEngineer` object, and Joblib stores it as a reference to `house_features.FeatureEngineer`. If the class lived inside the training script, no other script could load the model. **Keep `house_features.py` in the same folder as the model whenever you load it.** The prediction script also checks that the file still matches the model.

**Predict new areas (no training, no dataset needed):**

```bash
python predict_price.py
```

The file name is optional: with no argument it scores `new_houses.csv` (4 areas taken from the held-out test set). Train first (`python house_price_prediction.py`) so the `outputs/` folder with the model exists.

```text
   PredictedValue_100k  PredictedPrice_USD  AtOrAboveCap  Actual_USD
0                0.535               53500         False       47700
1                0.783               78300         False       45800
2                5.132              513200          True      500000
3                2.374              237400         False      218600
```

The second area is a real miss ($78,300 predicted against $45,800 actual), which is normal for a model with a typical error of about $44,000.

**What the prediction script checks:** missing columns, non-numeric values, empty values, zero or negative rooms and occupancy, values outside the training range (warning), and a scikit-learn version different from the one that trained the model (warning).

**Rules for using saved models:**

1. **Same library version:** load the model with the scikit-learn version that trained it.
2. **Trusted files only:** Joblib files can run code when opened, so load only files you created yourself.
3. **Retrain when data changes:** a saved model does not learn. With new sales data, rerun `house_price_prediction.py`.
4. **Git:** `.joblib` files are in `.gitignore`. After cloning, run `python house_price_prediction.py` once to create the model.

---

## Sample Prediction (from Python code)

```python
import pandas as pd
from house_price_prediction import load_model, RAW_FEATURES

model = load_model("outputs/house_price_model.joblib")

area = pd.DataFrame([[3.5, 30, 5.0, 1.1, 1400, 3.0, 34.2, -118.4]], columns=RAW_FEATURES)
price = model.predict(area)[0] * 100_000
print(f"Predicted median house value: ${price:,.0f}")
```

> **Disclaimer:** This is an educational project on 1990 census data. It must **not** be used to value real property.

---

## Author

**Prathmesh Girme**  
GitHub: [@Prathmesh-Girme](https://github.com/Prathmesh-Girme)
