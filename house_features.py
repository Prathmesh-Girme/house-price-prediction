########################################################
# Feature engineering for California house prices
#
# This lives in its own file on purpose. The saved model contains a
# FeatureEngineer object, and Joblib stores it as a reference to
# "house_features.FeatureEngineer". If the class were defined inside the
# training script, the prediction script could not load the model.
# Keep this file next to the model when you use it.
########################################################
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

ENGINEERED_FEATURES = [
    "BedroomRatio", "RoomsPerPerson", "LogPopulation",
    "DistToNearestCity_km", "DistToSF_km", "DistToLA_km",
]

# Big cities used to describe how close a block is to a job and price centre
CITIES = {
    "San Francisco": (37.7749, -122.4194),
    "Los Angeles": (34.0522, -118.2437),
    "San Diego": (32.7157, -117.1611),
    "San Jose": (37.3382, -121.8863),
    "Sacramento": (38.5816, -121.4944),
}

# These columns contain extreme outliers (for example 1,243 people per house)
CLIP_COLUMNS = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
CLIP_QUANTILES = (0.005, 0.995)


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in kilometres between points on the Earth (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Clip outliers, then add ratio, log and distance features.

    The clipping limits are learned in fit() from the TRAINING data only,
    so no information from the test set leaks into the model.
    """

    def fit(self, X, y=None):
        data = pd.DataFrame(X, columns=RAW_FEATURES)
        low, high = CLIP_QUANTILES
        self.clip_bounds_ = {c: (float(data[c].quantile(low)), float(data[c].quantile(high)))
                             for c in CLIP_COLUMNS}
        return self

    def transform(self, X):
        data = pd.DataFrame(X, columns=RAW_FEATURES).copy()
        for column, (low, high) in self.clip_bounds_.items():
            data[column] = data[column].clip(low, high)

        data["BedroomRatio"] = data["AveBedrms"] / data["AveRooms"]
        data["RoomsPerPerson"] = data["AveRooms"] / data["AveOccup"]
        data["LogPopulation"] = np.log1p(data["Population"])

        distances = {name: haversine_km(data["Latitude"], data["Longitude"], lat, lon)
                     for name, (lat, lon) in CITIES.items()}
        data["DistToNearestCity_km"] = np.minimum.reduce(list(distances.values()))
        data["DistToSF_km"] = distances["San Francisco"]
        data["DistToLA_km"] = distances["Los Angeles"]
        return data[RAW_FEATURES + ENGINEERED_FEATURES]
