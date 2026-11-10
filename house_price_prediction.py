########################################################
# House Price Prediction using Regression
#
# Pipeline : FeatureEngineer -> StandardScaler -> regressor
# Models   : Linear Regression, Ridge, Random Forest, Gradient Boosting
# Target   : median house value of a California census block, in units of
#            $100,000 (so 2.5 means $250,000). Values are capped at 5.0.
########################################################

########################################################
# Required Python Packages
########################################################
import json
import os
import platform
import warnings
from datetime import datetime

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from house_features import ENGINEERED_FEATURES, RAW_FEATURES, FeatureEngineer

########################################################
# Settings
########################################################

warnings.filterwarnings("ignore", category=FutureWarning)

SHOW_PLOTS = False  # False: only save plots to files (works without a display)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must come after backend selection

########################################################
# File Paths
########################################################

DATA_PATH = "california_housing.csv"
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_PATH = os.path.join(OUTPUT_DIR, "house_price_model.joblib")
METADATA_PATH = os.path.join(OUTPUT_DIR, "model_metadata.json")
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")
ABLATION_PATH = os.path.join(OUTPUT_DIR, "feature_engineering_check.csv")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

########################################################
# Columns and Units
########################################################

TARGET = "target"        # median house value in $100,000
PRICE_UNIT = 100_000
PRICE_CAP = 5.0          # values at 5.0 mean "$500,000 or more"

########################################################
# Model Settings
########################################################

RANDOM_STATE = 42
TRAIN_PERCENTAGE = 0.8
CV_FOLDS = 5
SPATIAL_GROUPS = 20      # number of map areas used for the "new area" check
MODEL_STEP = "model"

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}

CANDIDATE_MODELS = {
    "Linear Regression": (LinearRegression(), {}),
    "Ridge": (Ridge(random_state=RANDOM_STATE), {f"{MODEL_STEP}__alpha": [0.1, 1, 10, 100]}),
    "Random Forest": (
        RandomForestRegressor(n_estimators=150, min_samples_leaf=2, max_features=0.5,
                              n_jobs=1, random_state=RANDOM_STATE),
        {},
    ),
    "Gradient Boosting": (
        HistGradientBoostingRegressor(max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
        {f"{MODEL_STEP}__learning_rate": [0.05, 0.1], f"{MODEL_STEP}__max_leaf_nodes": [31, 63]},
    ),
}


########################################################
# Function name : read_data
# Description   : Read the dataset into a pandas dataframe
# Input         : Path of CSV file
# Output        : Dataframe
# Author        : Prathmesh Sunil Girme
########################################################
def read_data(path):
    """Read the data into pandas dataframe"""
    return pd.read_csv(path)


########################################################
# Function name : validate_dataset
# Description   : Check columns, missing values and impossible values
# Input         : Dataframe
# Output        : Nothing (raises ValueError if the data is not usable)
# Author        : Prathmesh Sunil Girme
########################################################
def validate_dataset(dataset):
    """Stop early with a clear message if the dataset is not as expected"""
    missing_columns = [c for c in RAW_FEATURES + [TARGET] if c not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    if dataset[RAW_FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Dataset contains missing values")

    if (dataset[TARGET] <= 0).any():
        raise ValueError("House values must be positive")

    if (dataset[["AveRooms", "AveOccup"]] <= 0).any().any():
        raise ValueError("AveRooms and AveOccup must be positive")


########################################################
# Function name : dataset_statistics
# Description   : Display shape, statistics, the price cap and correlations
# Input         : Dataframe
# Output        : Nothing (prints)
# Author        : Prathmesh Sunil Girme
########################################################
def dataset_statistics(dataset):
    print("Dataset Shape ::", dataset.shape)
    print("\nStatistics:\n", dataset.describe().T.round(2))

    capped = (dataset[TARGET] >= PRICE_CAP).sum()
    print(f"\nHouses at the price cap ({PRICE_CAP} = $500,000 or more) :: {capped} ({capped / len(dataset):.1%})")
    print("These values are cut off, so true prices above the cap are unknown.")

    outliers = {c: int((dataset[c] > dataset[c].quantile(0.999)).sum()) for c in ("AveRooms", "AveOccup")}
    print(f"Extreme outliers (above the 99.9th percentile) :: {outliers}")

    print("\nCorrelation with house value:\n",
          dataset[RAW_FEATURES].corrwith(dataset[TARGET]).sort_values().round(3))

    mean_baseline = np.sqrt(mean_squared_error(dataset[TARGET], np.full(len(dataset), dataset[TARGET].mean())))
    print(f"\nRMSE of always predicting the average price :: {mean_baseline:.3f} (= ${mean_baseline * PRICE_UNIT:,.0f})")


########################################################
# Function name : check_missing_values
# Description   : Count missing values and duplicate rows
# Input         : Dataframe
# Output        : Total missing values
# Author        : Prathmesh Sunil Girme
########################################################
def check_missing_values(dataset):
    missing = int(dataset.isnull().sum().sum())
    print("Total missing values ::", missing)
    print("Duplicate rows       ::", int(dataset.duplicated().sum()))
    return missing


########################################################
# Function name : finish_plot
# Description   : Save the current figure (and show it if SHOW_PLOTS is True)
# Input         : File name
# Output        : Nothing (writes a PNG)
# Author        : Prathmesh Sunil Girme
########################################################
def finish_plot(file_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, file_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"Plot saved to {path}")


########################################################
# Function name : plot_eda
# Description   : Price distribution, price map, income vs price, correlations
# Input         : Dataframe
# Output        : outputs/figures/eda_overview.png
# Author        : Prathmesh Sunil Girme
########################################################
def plot_eda(dataset):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(dataset[TARGET] * PRICE_UNIT / 1000, bins=50, color="#4c78a8", edgecolor="white")
    axes[0, 0].set_title("House Value Distribution (note the spike at the cap)")
    axes[0, 0].set_xlabel("Median house value ($ thousands)")
    axes[0, 0].set_ylabel("Census blocks")

    scatter = axes[0, 1].scatter(dataset["Longitude"], dataset["Latitude"], c=dataset[TARGET],
                                 s=3, cmap="viridis")
    fig.colorbar(scatter, ax=axes[0, 1], label="House value ($100,000)")
    axes[0, 1].set_title("Where the Expensive Blocks Are")
    axes[0, 1].set_xlabel("Longitude")
    axes[0, 1].set_ylabel("Latitude")

    hexbin = axes[1, 0].hexbin(dataset["MedInc"], dataset[TARGET], gridsize=40, cmap="Blues", mincnt=1)
    fig.colorbar(hexbin, ax=axes[1, 0], label="Census blocks")
    axes[1, 0].set_title("Income vs House Value")
    axes[1, 0].set_xlabel("MedInc (income in $10,000)")
    axes[1, 0].set_ylabel("House value ($100,000)")

    correlation = dataset[RAW_FEATURES].corrwith(dataset[TARGET]).sort_values()
    axes[1, 1].barh(correlation.index, correlation.values, color="#72b7b2")
    axes[1, 1].set_title("Correlation with House Value")
    finish_plot("eda_overview.png")


########################################################
# Function name : split_dataset
# Description   : Random train/test split
# Input         : Dataframe, train percentage
# Output        : train_x, test_x, train_y, test_y
# Author        : Prathmesh Sunil Girme
########################################################
def split_dataset(dataset, train_percentage, random_state=RANDOM_STATE):
    return train_test_split(dataset[RAW_FEATURES], dataset[TARGET],
                            train_size=train_percentage, random_state=random_state)


########################################################
# Function name : build_pipeline
# Description   : Build a Pipeline:
#                 FeatureEngineer : clip outliers, add ratio/log/distance features
#                                   (left out when engineer=False, for the check)
#                 StandardScaler  : puts features on one scale
#                 model           : the regressor
# Input         : Regressor, engineer flag
# Output        : Untrained pipeline
# Author        : Prathmesh Sunil Girme
########################################################
def build_pipeline(estimator, engineer=True):
    steps = []
    if engineer:
        steps.append(("features", FeatureEngineer()))
    steps.extend([("scaler", StandardScaler()), (MODEL_STEP, estimator)])
    return Pipeline(steps=steps)


########################################################
# Function name : make_cv
# Description   : Cross-validation splitter shared by every comparison
# Input         : Nothing
# Output        : KFold object
# Author        : Prathmesh Sunil Girme
########################################################
def make_cv():
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


########################################################
# Function name : tune_candidate
# Description   : Grid search one algorithm with 5-fold CV on TRAINING data
# Input         : Model name, regressor, parameter grid, training data
# Output        : Fitted best pipeline, dictionary of CV scores
# Author        : Prathmesh Sunil Girme
########################################################
def tune_candidate(name, estimator, grid, train_x, train_y):
    search = GridSearchCV(build_pipeline(estimator), grid, scoring=SCORING,
                          refit="rmse", cv=make_cv(), n_jobs=-1)
    search.fit(train_x, train_y)

    best = search.best_index_
    results = search.cv_results_
    settings = ", ".join(f"{k.split('__', 1)[1]}={v}" for k, v in search.best_params_.items()) or "defaults"
    row = {
        "Model": name,
        "Settings": settings,
        "CV_RMSE": -results["mean_test_rmse"][best],
        "CV_RMSE_Std": results["std_test_rmse"][best],
        "CV_MAE": -results["mean_test_mae"][best],
        "CV_R2": results["mean_test_r2"][best],
    }
    print(f"  {name:<18} {settings:<36} CV RMSE {row['CV_RMSE']:.4f} (+/- {row['CV_RMSE_Std']:.4f}) | R2 {row['CV_R2']:.3f}")
    return search.best_estimator_, row


########################################################
# Function name : evaluate_model
# Description   : RMSE, MAE and R-squared
# Input         : Model, features, true values
# Output        : Dictionary of metrics, predictions
# Author        : Prathmesh Sunil Girme
########################################################
def evaluate_model(model, x, y):
    predictions = model.predict(x)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }
    return metrics, predictions


########################################################
# Function name : compare_models
# Description   : Evaluate every tuned model on the test set ONCE
# Input         : Dictionary {name: model}, CV rows, train and test data
# Output        : Comparison dataframe
# Author        : Prathmesh Sunil Girme
########################################################
def compare_models(models, cv_rows, train_x, train_y, test_x, test_y):
    rows = []
    for row in cv_rows:
        model = models[row["Model"]]
        train_metrics, _ = evaluate_model(model, train_x, train_y)
        test_metrics, _ = evaluate_model(model, test_x, test_y)
        rows.append({
            **row,
            "Train_RMSE": train_metrics["rmse"],
            "Test_RMSE": test_metrics["rmse"],
            "Test_MAE": test_metrics["mae"],
            "Test_R2": test_metrics["r2"],
        })
    return pd.DataFrame(rows)


########################################################
# Function name : select_best_model
# Description   : Choose the final model by lowest CV RMSE.
#                 The test set plays no part in this choice.
# Input         : Comparison dataframe
# Output        : Name of the best model
# Author        : Prathmesh Sunil Girme
########################################################
def select_best_model(comparison):
    return comparison.sort_values("CV_RMSE").iloc[0]["Model"]


########################################################
# Function name : check_feature_engineering
# Description   : Does feature engineering actually help? Re-score every tuned
#                 model with the SAME settings and CV folds, with and without
#                 the engineered features.
# Input         : Dictionary {name: tuned pipeline}, training data
# Output        : Dataframe with CV RMSE for both versions
# Author        : Prathmesh Sunil Girme
########################################################
def check_feature_engineering(models, train_x, train_y):
    rows = []
    for name, tuned in models.items():
        estimator = clone(tuned.named_steps[MODEL_STEP])
        scores = {}
        for label, engineer in (("Raw", False), ("Engineered", True)):
            result = cross_validate(build_pipeline(clone(estimator), engineer), train_x, train_y,
                                    cv=make_cv(), scoring=SCORING, n_jobs=-1)
            scores[label] = -result["test_rmse"].mean()
        rows.append({
            "Model": name,
            "CV_RMSE_Raw": scores["Raw"],
            "CV_RMSE_Engineered": scores["Engineered"],
            "Improvement_Percent": 100 * (scores["Raw"] - scores["Engineered"]) / scores["Raw"],
        })
    return pd.DataFrame(rows)


########################################################
# Function name : plot_feature_engineering
# Description   : Bar chart of CV RMSE with and without engineered features
# Input         : Dataframe from check_feature_engineering
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
########################################################
def plot_feature_engineering(check):
    positions = np.arange(len(check))
    plt.figure(figsize=(9, 4.8))
    plt.bar(positions - 0.2, check["CV_RMSE_Raw"], width=0.4, label="8 raw features")
    plt.bar(positions + 0.2, check["CV_RMSE_Engineered"], width=0.4, label="raw + 6 engineered features")
    plt.xticks(positions, check["Model"])
    plt.ylabel("Cross-validated RMSE ($100,000, lower is better)")
    plt.title("Does Feature Engineering Help?")
    plt.legend()
    finish_plot("feature_engineering_check.png")


########################################################
# Function name : plot_model_comparison
# Description   : CV RMSE next to test RMSE for every model
# Input         : Comparison dataframe
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
########################################################
def plot_model_comparison(comparison):
    positions = np.arange(len(comparison))
    plt.figure(figsize=(9, 4.8))
    plt.bar(positions - 0.2, comparison["CV_RMSE"], width=0.4, label="Cross-validation (train)")
    plt.bar(positions + 0.2, comparison["Test_RMSE"], width=0.4, label="Test set")
    plt.xticks(positions, comparison["Model"])
    plt.ylabel("RMSE ($100,000, lower is better)")
    plt.title("Model Comparison")
    plt.legend()
    finish_plot("model_comparison.png")


########################################################
# Function name : spatial_cross_validation
# Description   : "New area" check. Nearby blocks have similar prices, so a
#                 random split lets the model learn from the neighbours of
#                 every test block. Here whole map areas are held out together.
# Input         : Pipeline, training data
# Output        : Dictionary with random-split and new-area RMSE and R2
# Author        : Prathmesh Sunil Girme
########################################################
def spatial_cross_validation(pipeline, train_x, train_y):
    areas = KMeans(n_clusters=SPATIAL_GROUPS, n_init=10, random_state=RANDOM_STATE).fit_predict(
        train_x[["Latitude", "Longitude"]])

    random_scores = cross_validate(clone(pipeline), train_x, train_y, cv=make_cv(), scoring=SCORING, n_jobs=-1)
    area_scores = cross_validate(clone(pipeline), train_x, train_y, groups=areas,
                                 cv=GroupKFold(n_splits=CV_FOLDS), scoring=SCORING, n_jobs=-1)
    return {
        "random_split_rmse": float(-random_scores["test_rmse"].mean()),
        "random_split_r2": float(random_scores["test_r2"].mean()),
        "new_area_rmse": float(-area_scores["test_rmse"].mean()),
        "new_area_r2": float(area_scores["test_r2"].mean()),
    }


########################################################
# Function name : bootstrap_rmse_interval
# Description   : 95% interval for the test RMSE by resampling test errors
# Input         : True values, predictions
# Output        : Lower and upper RMSE
# Author        : Prathmesh Sunil Girme
########################################################
def bootstrap_rmse_interval(y_true, predictions, runs=1000):
    rng = np.random.RandomState(RANDOM_STATE)
    errors = (np.asarray(y_true) - predictions) ** 2
    samples = [np.sqrt(rng.choice(errors, size=len(errors)).mean()) for _ in range(runs)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


########################################################
# Function name : analyse_errors
# Description   : Where does the model go wrong? Error by price band and for
#                 the capped houses.
# Input         : True values, predictions
# Output        : Dataframe with one row per price band
# Author        : Prathmesh Sunil Girme
########################################################
def analyse_errors(y_true, predictions):
    data = pd.DataFrame({"actual": np.asarray(y_true), "predicted": predictions})
    data["error"] = data["predicted"] - data["actual"]
    bands = pd.cut(data["actual"], [0, 1.5, 2.5, 3.5, 4.99, 5.01],
                   labels=["under $150k", "$150k-250k", "$250k-350k", "$350k-500k", "capped at $500k"])
    rows = []
    for band, part in data.groupby(bands, observed=True):
        rows.append({"Band": band, "Houses": len(part),
                     "MAE": float(part["error"].abs().mean()), "Average_Error": float(part["error"].mean())})
    return pd.DataFrame(rows)


########################################################
# Function name : plot_predictions
# Description   : Predicted vs actual, residuals, and error by price band
# Input         : True values, predictions, error table, model name
# Output        : PNG file
# Author        : Prathmesh Sunil Girme
########################################################
def plot_predictions(y_true, predictions, bands, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(y_true, predictions, s=4, alpha=0.4)
    axes[0].plot([0, 5], [0, 5], color="red", linestyle="--", label="perfect prediction")
    axes[0].set_xlabel("Actual value ($100,000)")
    axes[0].set_ylabel("Predicted value ($100,000)")
    axes[0].set_title(f"Predicted vs Actual - {model_name}")
    axes[0].legend()

    residuals = np.asarray(y_true) - predictions
    axes[1].scatter(predictions, residuals, s=4, alpha=0.4)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted value ($100,000)")
    axes[1].set_ylabel("Actual minus predicted")
    axes[1].set_title("Residuals")

    axes[2].bar(bands["Band"].astype(str), bands["MAE"], color="#e45756")
    axes[2].set_ylabel("Mean absolute error ($100,000)")
    axes[2].set_title("Error by Price Band")
    axes[2].tick_params(axis="x", rotation=20)
    finish_plot("predictions_and_errors.png")


########################################################
# Function name : plot_feature_importance
# Description   : Permutation importance on the test set: how much the RMSE
#                 grows when one original column is shuffled
# Input         : Fitted pipeline, test data, title
# Output        : PNG file, importance series
# Author        : Prathmesh Sunil Girme
########################################################
def plot_feature_importance(model, test_x, test_y, title):
    result = permutation_importance(model, test_x, test_y, scoring="neg_root_mean_squared_error",
                                    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(result.importances_mean, index=RAW_FEATURES).sort_values(ascending=False)
    ordered = importance.iloc[::-1]
    errors = pd.Series(result.importances_std, index=RAW_FEATURES)[ordered.index]

    plt.figure(figsize=(7.5, 4.8))
    plt.barh(ordered.index, ordered.values, xerr=errors.values, color="#4c78a8")
    plt.xlabel("Increase in RMSE when the column is shuffled ($100,000)")
    plt.title(title)
    finish_plot("feature_importance.png")
    return importance


########################################################
# Function name : predict_new_houses
# Description   : Predict the median house value for new census blocks
# Input         : Trained model, houses
# Output        : Dataframe with predicted value
# Author        : Prathmesh Sunil Girme
########################################################
def predict_new_houses(model, houses):
    predicted = model.predict(houses[RAW_FEATURES])
    return pd.DataFrame({
        "PredictedValue_100k": predicted.round(3),
        "PredictedPrice_USD": (predicted * PRICE_UNIT).round(-2).astype(int),
    }, index=houses.index)


########################################################
# Function name : save_model / load_model
# Description   : Save or load the trained Pipeline with joblib.
#                 house_features.py must be importable when loading.
#                 Only load model files you created yourself.
# Input         : Model and path / path
# Output        : Nothing / model
# Author        : Prathmesh Sunil Girme
########################################################
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


########################################################
# Function name : json_default
# Description   : Let json.dump handle numpy numbers
# Author        : Prathmesh Sunil Girme
########################################################
def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot save {type(value)} to JSON")


########################################################
# Function name : save_model_metadata
# Description   : Save a "model card" next to the .joblib file: expected
#                 columns, training ranges, units, library versions, metrics.
# Output        : outputs/model_metadata.json
# Author        : Prathmesh Sunil Girme
########################################################
def save_model_metadata(model_name, model, train_x, test_metrics, cv_rmse, path=METADATA_PATH):
    estimator = model.named_steps[MODEL_STEP]
    simple_types = (int, float, str, bool, type(None))
    metadata = {
        "model_name": model_name,
        "model_class": type(estimator).__name__,
        "model_parameters": {k: v for k, v in estimator.get_params().items() if isinstance(v, simple_types)},
        "selection": {"rule": "lowest cross-validated RMSE on training data", "cv_rmse": cv_rmse},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "training_rows": int(len(train_x)),
        "features": RAW_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "training_ranges": {c: [float(train_x[c].min()), float(train_x[c].max())] for c in RAW_FEATURES},
        "target_unit": "median house value in units of $100,000",
        "price_cap": PRICE_CAP,
        "test_metrics": test_metrics,
        "disclaimer": "Educational project on 1990 census data. Not for real valuation.",
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=json_default)
    print(f"Model metadata saved to {path}")


########################################################
# Function name : main
# Description   : Main function from where execution starts
# Author        : Prathmesh Sunil Girme
########################################################
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load and understand the dataset
    dataset = read_data(DATA_PATH)
    validate_dataset(dataset)
    print("\n--- 1) Dataset ---")
    dataset_statistics(dataset)

    # 2) Missing values
    print("\n--- 2) Missing values ---")
    check_missing_values(dataset)

    # 3) Exploratory plots
    print("\n--- 3) Exploratory analysis ---")
    plot_eda(dataset)

    # 4) Split
    print("\n--- 4) Train/test split ---")
    train_x, test_x, train_y, test_y = split_dataset(dataset, TRAIN_PERCENTAGE)
    print("Train_x Shape :: ", train_x.shape)
    print("Test_x Shape  :: ", test_x.shape)

    # 5) Baseline: always predict the average training price
    baseline = DummyRegressor(strategy="mean").fit(train_x, train_y)
    baseline_metrics, _ = evaluate_model(baseline, test_x, test_y)
    print(f"\nBaseline (always the average price) :: test RMSE {baseline_metrics['rmse']:.3f}, "
          f"R2 {baseline_metrics['r2']:.3f}")

    # 6) Tune every model with the same CV folds
    print("\n--- 5) Tuning models (5-fold CV on training data) ---")
    models, cv_rows = {}, []
    for name, (estimator, grid) in CANDIDATE_MODELS.items():
        models[name], row = tune_candidate(name, estimator, grid, train_x, train_y)
        cv_rows.append(row)

    # 7) Compare on the test set (once) and choose the final model
    print("\n--- 6) Model comparison ---")
    comparison = compare_models(models, cv_rows, train_x, train_y, test_x, test_y)
    comparison.round(4).to_csv(COMPARISON_PATH, index=False)
    print(comparison[["Model", "CV_RMSE", "CV_R2", "Train_RMSE", "Test_RMSE", "Test_MAE", "Test_R2"]]
          .round(3).to_string(index=False))
    plot_model_comparison(comparison)

    final_name = select_best_model(comparison)
    final_model = models[final_name]
    final_row = comparison[comparison["Model"] == final_name].iloc[0]
    print(f"\nFinal model (lowest CV RMSE) :: {final_name} (CV RMSE {final_row['CV_RMSE']:.4f})")

    # 8) Does feature engineering help?
    print("\n--- 7) Does feature engineering help? (same settings, same CV folds) ---")
    check = check_feature_engineering(models, train_x, train_y)
    check.round(4).to_csv(ABLATION_PATH, index=False)
    print(check.round(3).to_string(index=False))
    plot_feature_engineering(check)

    # 9) Detailed report for the final model
    final_metrics, predictions = evaluate_model(final_model, test_x, test_y)
    low, high = bootstrap_rmse_interval(test_y, predictions)
    final_metrics["rmse_95_interval"] = [low, high]
    final_metrics["train_rmse"] = float(final_row["Train_RMSE"])
    print(f"\n===== FINAL: {final_name} =====")
    print(f"Test RMSE :: {final_metrics['rmse']:.4f} (95% interval {low:.3f} to {high:.3f}) "
          f"= about ${final_metrics['rmse'] * PRICE_UNIT:,.0f}")
    print(f"Test MAE  :: {final_metrics['mae']:.4f} = about ${final_metrics['mae'] * PRICE_UNIT:,.0f}")
    print(f"Test R2   :: {final_metrics['r2']:.4f}")

    # 10) Error analysis
    print("\n--- 8) Error analysis by price band ---")
    bands = analyse_errors(test_y, predictions)
    print(bands.round(3).to_string(index=False))
    plot_predictions(test_y, predictions, bands, final_name)

    # 11) New-area check
    print("\n--- 9) New-area check (hold out whole map areas) ---")
    spatial = spatial_cross_validation(final_model, train_x, train_y)
    print(f"Random split CV :: RMSE {spatial['random_split_rmse']:.3f}, R2 {spatial['random_split_r2']:.3f}")
    print(f"New-area CV     :: RMSE {spatial['new_area_rmse']:.3f}, R2 {spatial['new_area_r2']:.3f}")

    # 12) Feature importance
    print("\n--- 10) Feature importance (permutation, test set) ---")
    importance = plot_feature_importance(final_model, test_x, test_y, f"Feature Importance - {final_name}")
    print(importance.round(4).to_string())

    # 13) Predict for new houses (rows from the held-out test set)
    print("\n--- 11) Predictions for 4 held-out census blocks ---")
    sample = test_x.head(4)
    result = predict_new_houses(final_model, sample)
    result["ActualPrice_USD"] = (test_y.head(4) * PRICE_UNIT).round(-2).astype(int)
    print(result.to_string())

    # 14) Save the model and metadata, load the model, and test a sample
    print("\n--- 12) Save and load ---")
    save_model(final_model, MODEL_PATH)
    save_model_metadata(final_name, final_model, train_x, final_metrics, float(final_row["CV_RMSE"]))
    loaded = load_model(MODEL_PATH)
    print(f"Loaded model prediction for first test block: ${loaded.predict(test_x.iloc[[0]])[0] * PRICE_UNIT:,.0f}")

    summary = {
        "final_model": final_name,
        "selection_rule": "lowest cross-validated RMSE on training data (test set not used)",
        "baseline_average_price_test": baseline_metrics,
        "final_model_test": final_metrics,
        "new_area_check": spatial,
        "error_by_price_band": bands.to_dict(orient="records"),
        "feature_importance_rmse_increase": importance.to_dict(),
        "feature_engineering_check": check.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=json_default)
    print(f"Metrics saved to {METRICS_PATH}")


########################################################
# Application starter
########################################################

if __name__ == "__main__":
    main()
