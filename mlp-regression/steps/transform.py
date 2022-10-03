"""
This module defines the following routines used by the 'transform' step of the regression pipeline:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder


# def transformer_fn():
#     """
#     Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
#     The transformer's input and output signatures should be compatible with scikit-learn
#     transformers.
#     """
#     return Pipeline(
#         steps=[
#             (
#                 "encoder",
#                 ColumnTransformer(
#                     transformers=[
#                         (
#                             "hour_encoder",
#                             OneHotEncoder(categories="auto", sparse=False),
#                             ["pickup_hour"],
#                         ),
#                         (
#                             "day_encoder",
#                             OneHotEncoder(categories="auto", sparse=False),
#                             ["pickup_dow"],
#                         ),
#                         (
#                             "std_scaler",
#                             StandardScaler(),
#                             ["trip_distance", "trip_duration"],
#                         ),
#                     ]
#                 ),
#             ),
#         ]
#     )



def transformer_fn():

    return Pipeline(
        steps=[
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                         (
                            "housing_ordinal",
                            OrdinalEncoder(),
                            ["ocean_proximity"]
                         ),
                        (
                            "housing_categorical",
                            OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore"),
                            ["ocean_proximity"]
                         ),
                        
                        (
                            "std_scaler",
                            StandardScaler(),
                            ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
                             "households", "median_income"],
                        ),
                    ]
                ),
            ),
        ]
    )
