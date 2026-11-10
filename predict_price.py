#!/usr/bin/env python3
"""House price prediction - inference only.

Loads the model saved by house_price_prediction.py (joblib) and predicts the
median house value of California census blocks. No training happens here.
house_features.py must stay in the same folder, because the saved model
uses it. Educational project, not for real valuation.

Usage:
    python predict_price.py                          (scores new_houses.csv)
    python predict_price.py new_houses.csv
    python predict_price.py new_houses.csv --output predicted_prices.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import pandas as pd
import sklearn

import house_features  # the saved model needs this module to load

LOGGER = logging.getLogger("house_price_predict")

DEFAULT_MODEL_PATH = Path("outputs/house_price_model.joblib")
DEFAULT_METADATA_PATH = Path("outputs/model_metadata.json")
DEFAULT_INPUT_PATH = Path("new_houses.csv")
PRICE_UNIT = 100_000


def load_metadata(path: Path) -> Dict[str, Any]:
    """Read the model card written during training."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}. Run house_price_prediction.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(path: Path):
    """Load the saved pipeline (feature engineering + scaler + regressor)."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run house_price_prediction.py first.")
    return joblib.load(path)


def warn_if_version_differs(metadata: Dict[str, Any]) -> None:
    """A model should be loaded with the scikit-learn version that built it."""
    trained_with = metadata.get("scikit_learn_version")
    if trained_with and trained_with != sklearn.__version__:
        LOGGER.warning("Model was trained with scikit-learn %s but %s is installed. "
                       "Results may be wrong or loading may fail.", trained_with, sklearn.__version__)


def read_houses(path: Path, metadata: Dict[str, Any]) -> pd.DataFrame:
    """Read houses from CSV and check they match what the model expects."""
    if not path.exists():
        raise FileNotFoundError(f"House file not found: {path}")

    data = pd.read_csv(path)
    features = metadata["features"]

    missing_columns = [c for c in features if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")

    for column in features:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must contain numbers only")

    empty_rows = data.index[data[features].isnull().any(axis=1)].tolist()
    if empty_rows:
        raise ValueError(f"Rows with missing values (0-based row numbers): {empty_rows}")

    if (data[["AveRooms", "AveOccup"]] <= 0).any().any():
        raise ValueError("AveRooms and AveOccup must be greater than zero")

    outside = {}
    for column, (low, high) in metadata["training_ranges"].items():
        rows = data.index[(data[column] < low) | (data[column] > high)].tolist()
        if rows:
            outside[column] = rows
    if outside:
        LOGGER.warning("Values outside the training range (treat these results with care): %s", outside)
    return data


def predict_prices(model, data: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """Predict the value of each census block."""
    predicted = model.predict(data[metadata["features"]])
    result = pd.DataFrame({
        "PredictedValue_100k": predicted.round(3),
        "PredictedPrice_USD": (predicted * PRICE_UNIT).round(-2).astype(int),
    }, index=data.index)
    result["AtOrAboveCap"] = predicted >= metadata["price_cap"]
    if "Actual_USD" in data.columns:
        result["Actual_USD"] = data["Actual_USD"]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_price.py",
        description="Predict California house values with the saved model (no retraining).")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT_PATH,
                        help=f"CSV file with the 8 feature columns (default: {DEFAULT_INPUT_PATH})")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="joblib model file")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH, help="model metadata JSON")
    parser.add_argument("--output", type=Path, default=None, help="save predictions to CSV")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    try:
        metadata = load_metadata(args.metadata)
        warn_if_version_differs(metadata)
        if metadata["features"] != house_features.RAW_FEATURES:
            raise ValueError("house_features.py does not match the model: run house_price_prediction.py again")
        model = load_model(args.model)
        data = read_houses(args.input, metadata)
    except (OSError, ValueError, KeyError) as error:
        LOGGER.error("%s", error)
        return 1

    result = predict_prices(model, data, metadata)
    LOGGER.info("Predicted %d census block(s)", len(result))
    print(result.to_string())
    print("\nValues are for census blocks in 1990 data (not single houses). The training data is cut off at "
          "$500,000, so the model cannot predict far above that. Not for real valuation.")

    if args.output:
        result.to_csv(args.output, index=False)
        LOGGER.info("Saved predictions to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
