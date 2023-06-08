import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, median_absolute_error, max_error,
)

from typing import Callable, List, Dict, Any
from xgboost import XGBRegressor

from .features import TrainingData


METRICS = {
    "R2": r2_score,
    "MAE": mean_absolute_error,
    "MedAE": median_absolute_error,
    "MaxErr": max_error,
    "MAPE": mean_absolute_percentage_error,
    "MSE": mean_squared_error,
}


class CreateModel:
    def __init__(self, feat_cols):
        """Model creator factory

        Args:
            feat_cols (list): list of selected features to use in the models (partly supported)
        """
        self.feat_cols = feat_cols

    def __call__(self, model_tag, **kwargs):
        """Model creator call

        Args:
            model_tag (str): model tag defining the model or pipeline to use
            **kwargs: keyword arguments to pass to the model or pipeline
        """

        feature_selector = ColumnTransformer(
                [("selector", "passthrough", self.feat_cols)],
                remainder="drop",
        )

        pipes = {
            "XGBCV": lambda kwargs: Pipeline(
                [
                    ("column_selector", feature_selector),
                    ("model", GridSearchCV(XGBRegressor(), **kwargs)),
                ]
            ),
            "GBT": lambda kwargs: Pipeline(
                [
                    ("column_selector", feature_selector),
                    ("scaler", StandardScaler()),
                    ("model", GradientBoostingRegressor(**kwargs)),
                ]
            ),
            "XGB": lambda kwargs: Pipeline(
                [
                    ("column_selector", feature_selector),
                    ("scaler", StandardScaler()),
                    ("model", XGBRegressor(**kwargs)),
                ]
            ),
            "LR": lambda kwargs: Pipeline(
                [
                    ("column_selector", feature_selector),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            "naive_last": lambda kwargs: Pipeline(
                [
                    ("naive", FunctionTransformer(lambda x: x[["p"]])),
                    ("LR", LinearRegression()),
                ]
            ),
            "single_best_corr": lambda kwargs: Pipeline(
                [
                    ("column_selector", feature_selector),
                    ("selector", SelectKBest(score_func=f_regression, k=1)),
                    ("LR", LinearRegression()),
                ]
            ),
        }

        return pipes[model_tag](kwargs)


def train_multiple_steps(
        data: TrainingData,
        get_model: Callable,
        shifts_steps: List[int],
        verb: bool = True,
        split_date: str = None,
        train_split: float = None,
        with_time: bool = False,
        **model_kwargs
) -> Dict[float, Any]:
    """Train multiple models with different shift steps

    Args:
        data (TrainingData): training data that provides train_test_split method
        get_model (Callable): model creator function
        shifts_steps (List[int]): list of shift steps to use
        verb (bool, optional): verbosity. Defaults to True.

    Returns:
        Dict[float, Any]: dictionary of trained models
    """

    feat_cols = data.feature_columns
    print(type(feat_cols))
    print(feat_cols)
    if with_time:
        feat_cols.append("ssrd_diff")
        feat_cols.append("ssr_diff")
    trained_models = {}
    for shift_steps in shifts_steps:
        df_train, df_test = data.train_test_split(shift_steps=shift_steps, split_date=split_date, train_split=train_split)
        if with_time:
            df_train, df_test = forecast_Bastorf(shift_steps, split_date, df_train, df_test)
        model = get_model(**model_kwargs)
        model.fit(df_train[feat_cols], df_train["target"])
        ytrue = df_test["target"]
        ypred = model.predict(df_test[feat_cols])
        if verb:
            print(f"{shift_steps:5.02f} ->", " | ".join(["{}: {:.2f}".format(metric_name, metric(ytrue, ypred)) for metric_name, metric in METRICS.items()]))

        trained_models[shift_steps] = model

    return trained_models


def compare_bar(models, data, metric="MAE", split_date=None, train_split=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    vals = []
    for label,models in models.items():
        for ahead,model in models.items():
            _, df_test = data.train_test_split(shift_steps=ahead, train_split=train_split, split_date=split_date)
            y = df_test["target"]
            yp = model.predict(df_test[data.feature_columns])
            met = METRICS[metric](y, yp)
            vals.append([ahead/4, label, met])
    print(vals)
    sns.barplot(
        pd.DataFrame(vals, columns=["ahead [h]", "labels", metric]),
        x="ahead [h]",
        y=metric,
        hue="labels",
    )
    plt.grid()


def forecast_Bastorf(
    shift: int, split_date, df_train, df_test 
) -> pd.DataFrame:
    """Take a Multiindex ("time", "step", "values") dataframe from ECMWF with forecasts
    and create a shifted feature with the correct forecast creation time.
    Args:
        df: dataframe with the forecasts.
        shift: shift in 15 min steps.
        index: index of the chosen latitude longitude combination from the grid for
        Mutltilevelindex values.
        forecasts_per_day: number of forecasts per day in the dataframe.
        start_date: start date of the dataframe.
        end_date: end date of the dataframe.
        diff: if True, the absolute values of ssr and ssrd are calculated.
    Returns:
        dataframe with the shifted feature. If apply_diff is True, ssr and ssrd are
        also renamed to ssr_diff and ssrd_diff.
    """
    df = pd.read_parquet("data/Bastorf_01012021-15052020_36steps.parquet")
    secs = 60
    window_duration = 15
    s_to_ns = 1000000000
    forecasts_per_day = 2
    start_date = "2021-11-16"
    end_date = "2023-04-01"
    
    # how long a forecast is valid, until a newer one is available, measured in 15
    # min steps
    valid_duration = (24 / forecasts_per_day) * 4

    df_copy = df.copy()

    df = df.xs(1, level="values")
    print(df.shape)

    df.insert(5, "ssr_diff", df_copy.xs(1, level="values")["ssr"].diff())
    df.insert(6, "ssrd_diff", df_copy.xs(1, level="values")["ssrd"].diff())
    df.pop("ssr")
    df.pop("ssrd")
    df["ssr_diff"] = df["ssr_diff"].apply(lambda x: max(0, x))
    df["ssrd_diff"] = df["ssrd_diff"].apply(lambda x: max(0, x))
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    print(df.shape)

    if start_date:
        df = df[df["time"] >= pd.to_datetime(start_date)]
        df_train = df_train[df_train["time"] >= pd.to_datetime(start_date)]
        df_test = df_test[df_test["time"] >= pd.to_datetime(start_date)]
        print(df.shape)

    if end_date:
        df = df[df["time"] < pd.to_datetime(end_date)]
        df_train = df_train[df_train["time"] < pd.to_datetime(end_date)]
        df_test = df_test[df_test["time"] < pd.to_datetime(end_date)]
        print(df.shape)

    print(s_to_ns * secs * window_duration * shift)
    print(s_to_ns * secs * window_duration * (shift + valid_duration))
    fc = df[
        (df["step"] >= s_to_ns * secs * window_duration * shift)
        & (df["step"] < s_to_ns * secs * window_duration * (shift + valid_duration))
    ]
    print(fc.shape)

    # the forecasts are given in one hour windows, so we need to resample to 15 min
    fc = fc.loc[fc.index.repeat(4)].reset_index(drop=True)
    # remove first row, so the resampling is applied in a way that one forecasted hour
    # covers the timeframe from xx:45 to xx:30
    fc.drop([0], inplace=True)
    empty_row = pd.DataFrame(0, index=range(1), columns=fc.columns)
    print(empty_row)
    print(fc.shape)
    fc = pd.concat([fc, empty_row]).reset_index(drop=True)
    data = pd.concat([df_train, df_test])
    data["ssrd_abs"] = fc["ssrd"].values
    data["ssr_abs"] = fc["ssr"].values


    return df[df.index < split_date], df[df.index >= split_date]