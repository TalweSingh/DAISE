from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np


@dataclass
class ClientData:
    """
    Dataclass for client data

    Args:
        path: path to csv file
        latitude: latitude of client location
        longitude: longitude of client location
        altitude: altitude of client location
    """

    path: str
    latitude: float
    longitude: float
    altitude: float
    _df: pd.Series = None

    @property
    def df(self) -> pd.Series:
        """
        Returns:
            pd.Series: client data as pandas series
        """
        if self._df is None:
            self._df = pd.read_csv(
                self.path,
                names=["ts", "p"],
                header=0,
                dtype={"ts": str, "p": float},
                parse_dates=["ts"],
            ).set_index("ts")["p"]
        return self._df


_data_dir = "data"
client_data = {
    "Bastorf": ClientData(
        path=f"{_data_dir}/bast_pv_15min_2021-06-18_2023-04-06.csv",
        latitude=54.1266,
        longitude=11.6947,
        altitude=51,
    ),
    "Bona": ClientData(
        path=f"{_data_dir}/mid40_pv_15min_2022-12-24_2023-04-15.csv",
        latitude=50.3757,
        longitude=8.05648,
        altitude=138,
    ),
    "Wriezen": ClientData(
        path=f"{_data_dir}/mid01_pv_15min_2022-01-01_2023-03-28.csv",
        latitude=52.71231,
        longitude=14.13422,
        altitude=3,
    ),
    "Borna": ClientData(
        path=f"{_data_dir}/mid13_pv_15min_2022-04-01_2023-02-28.csv",
        latitude=51.115478,
        longitude=12.489368,
        altitude=146,
    ),
}


def create_forecast_df(
    df: pd.DataFrame, shift: int, forecasts_per_day: int, index: int = -1, start_date: str = None, end_date: str = None, diff: bool = False
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
    secs = 60
    window_duration = 15
    s_to_ns = 1000000000

    # how long a forecast is valid, until a newer one is available, measured in 15
    # min steps
    valid_duration = (24 / forecasts_per_day) * 4

    if diff:
        df_copy = df.copy()

    if index != -1:
        df = df.xs(index, level="values")
        print(df.shape)

    if diff: 
        df.insert(5, "ssr_diff", df_copy.xs(index, level="values")["ssr"].diff())
        df.insert(6, "ssrd_diff", df_copy.xs(index, level="values")["ssrd"].diff())
        df.pop("ssr")
        df.pop("ssrd")
        df["ssr_diff"] = df["ssr_diff"].apply(lambda x: max(0, x))
        df["ssrd_diff"] = df["ssrd_diff"].apply(lambda x: max(0, x))
        df.fillna(0, inplace=True)
        df.reset_index(inplace=True)
        print(df.shape)

    if start_date:
        df = df[df["time"] >= pd.to_datetime(start_date)]
        print(df.shape)

    if end_date:
        df = df[df["time"] < pd.to_datetime(end_date)]
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
    repeat_array = np.tile(np.arange(4), len(fc) // 4)
    fc['valid_time'] = fc['valid_time'] + pd.to_timedelta(repeat_array * 15, unit='m') - pd.to_timedelta(15, unit='m')

    # remove first row, so the resampling is applied in a way that one forecasted hour
    # covers the timeframe from xx:45 to xx:30
    fc.drop([0], inplace=True)
    empty_row = pd.DataFrame(0, index=range(1), columns=fc.columns)
    print(empty_row)
    print(fc.shape)
    fc = pd.concat([fc, empty_row]).reset_index(drop=True)
    print(fc.shape)

    return fc

def train_test_split(df, shift_steps: int, train_split: float = None, split_date=None) -> pd.DataFrame:
    """
    Generate train and test data from combined data.

    Args:
        shift_steps: Number of time steps to predict into the future
        train_split: Fraction of data to use for training
    """
    assert (train_split is None) != (split_date is None), "Either train_split or split_date must be specified"

    if split_date is None:
        split_date = df.index[int(train_split * len(df))]

    return df[df.index < split_date], df[df.index >= split_date]
