import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable


class ShiftedFeatures:
    def __init__(
        self,
        df: pd.Series,
        shift_features_fn: Callable,
        fixed_features_fn: Callable = None,
        data_pre_fn: Callable = None,
        data_post_fn: Callable = None,
        location: Dict[str, float] = None,
    ):
        """
        Feature generation for time series data. The data is split into fixed and shifted features.
        The fixed features are not shifted in time, while the shifted features are shifted in time.
        The shifted features are shifted by a fixed number of time steps into the past.

        Args:
            df: Series with sorted datetime index and 15min sample frequency
            shift_features_fn: Function to generate shifted features
            fixed_features_fn: Function to generate fixed features
            data_pre_fn: Function to prepare data before feature generation
            data_post_fn: Function to prepare data after feature generation
            location: Dict with fields latitude, longitude [in degree] and altitude [in m]
        """

        # Ensure 15min equal sampling, implies sorted index
        period_vc = (df.index.astype(np.int64) // 1e9).to_series().diff().value_counts()
        assert period_vc.shape[0] == 1, "Expected fixed sample frequency"
        assert period_vc.index[0] == 900, "Expected frequency of 15min"

        # Ensure to not remove rows!
        self.df = df
        if data_pre_fn is not None:
            before_rows = self.df.shape[0]
            self.df = data_pre_fn(self.df)
            assert self.df.shape[0] == before_rows
        assert self.df[self.df.isnull()].shape[0] == 0

        self._data_post_fn = data_post_fn

        self.df_shift = shift_features_fn(self.df, location=location)
        self.df_fixed = (
            fixed_features_fn(self.df, location=location) if fixed_features_fn is not None else None
        )

        self._columns = (
                self.df_shift.columns.tolist() +
                self.df_fixed.columns.tolist() if self.df_fixed is not None else []
        )

    def data(self, shift_steps: int, post: bool = True) -> pd.DataFrame:
        """
        Generate fixed and shifted features before post processing.

        Args:
            shift_steps: Number of time steps to shift the window into the past
            post: Whether to apply post processing function

        Returns:
            DataFrame with fixed and shifted features
        """
        assert shift_steps > 0, "Shift step must be positive to prevent target leakage"
        df_shifted = self.df_shift.shift(-shift_steps)

        if self.df_fixed is None:
            return df_shifted

        df = self.df_fixed.merge(
            df_shifted,
            left_index=True,
            right_index=True,
            how="left",
        )

        if post and self._data_post_fn:
            df = self._data_post_fn(df)

        return df

    @property
    def feature_columns(self) -> List[str]:
        """
        Returns:
            List of feature columns
        """
        return [c for c in self._columns if "target" not in c]


class TrainingData:
    def __init__(
        self,
        feature_data: Dict[str, ShiftedFeatures],
        name: str = None,
    ):
        """
        Training data for time series data. Multiple ShiftedFeatures objects can be combined.

        Args:
            feature_data: Dict of names and ShiftedFeatures objects
            name: Dataset name
        """
        keys = list(feature_data.keys())
        assert all(feature_data[keys[0]].feature_columns == feature_data[k].feature_columns for k in keys[1:])
        self._feature_data = feature_data
        self.feature_columns = feature_data[keys[0]].feature_columns
        self.name = name if name is not None else "".join(feature_data.keys())

    def data(self, shift_steps: int, post: bool = True) -> pd.DataFrame:
        """
        Generate combined data from multiple ShiftedFeatures objects.

        Args:
            shift_steps: Number of time steps to predict into the future
            post: Whether to apply post processing function
        """
        return pd.concat(
            [fd.data(shift_steps=shift_steps, post=post) for fd in self._feature_data.values()],
            axis=0,
            ).sort_index()

    def train_test_split(self, shift_steps: int, train_split: float = None, split_date=None) -> pd.DataFrame:
        """
        Generate train and test data from combined data.

        Args:
            shift_steps: Number of time steps to predict into the future
            train_split: Fraction of data to use for training
        """
        assert (train_split is None) != (split_date is None), "Either train_split or split_date must be specified"

        df = self.data(shift_steps=shift_steps)

        if split_date is None:
            split_date = df.index[int(train_split * len(df))]

        return df[df.index < split_date], df[df.index >= split_date]



def fixed_features(ser: pd.Series, location: Dict = None) -> pd.DataFrame:
    """
    Generate fixed features for time series data.

    Args:
        ser: Series with sorted datetime index

    Returns:
        DataFrame with fixed features
    """
    dfs = [
        ser.to_frame(),
        moving_features(ser),
        daily_features(ser),
        periodic_features(
            ser,
            repeats=[1, 2, 5, 7, 28],
            period_steps=96,
            suffix="_now",
        ),
    ]

    if location:
        dfs.append(irradiance(ser, shift_steps=0, location=location, name="irr_now").to_frame())

    return pd.concat(dfs, axis=1)


def shifted_features_target(ser: pd.Series, location: Dict) -> pd.DataFrame:
    """
    Generate shifted features for time series data.

    Args:
        ser: Series with sorted datetime index

    Returns:
        DataFrame with shifted features
    """

    dfs = [
        time_features(ser, doy_shift=9),
        periodic_features(
            ser,
            repeats=[1, 2, 5, 7, 28],
            period_steps=96,
            suffix="_tar",
        ),
        shift_target(ser, shift_steps=0).to_frame(),
    ]

    if location:
        dfs.append(irradiance(ser, shift_steps=0, location=location, name="irr_tar").to_frame())

    return pd.concat(dfs, axis=1)


def irradiance(df: pd.Series, shift_steps: int, location: Dict, name: str = "irr") -> pd.Series:
    """
    Generate irradiance on horizontal surface plane for index of time series.

    Args:
        df: Series with datetime index
        shift_steps: Number of 15 min time steps to shift the time into the future
        location: Dict with fields latitude, longitude [in degree] and altitude [in m]
    """
    from solarpy.radiation import irradiance_on_plane

    shift_irr_h = pd.Timedelta(
        location["longitude"] / 360 * 24 + shift_steps / 4, unit="h"
    )
    irr = [
        irradiance_on_plane(
            vnorm=np.array([0, 0, -1]),
            h=location["altitude"],
            date=e + shift_irr_h,
            lat=location["latitude"],
        )
        for e in df.index
    ]

    return pd.Series(irr, index=df.index, name=name)


def _periodic_feature_aggregator(
    df: pd.Series,
    offset_steps: int,
    period_steps: int,
    repeat: int,
    metrics: List[str],
    suffix: str = "",
) -> pd.DataFrame:
    """
    Feature aggregation over periodic data points.

    Args:
        df: Series with sorted datetime index
        offset_steps: Number of time steps to shift the window into the past
        period_steps: Number of time steps between reoccuring windows
        repeat: Number of data points to aggregate
        metrics: List of aggregation functions to apply to the window

    Returns:
        DataFrame with aggregated features
    """
    assert len(df.shape) == 1, "Expect Series"
    assert df.index.is_monotonic_increasing, "Index must be sorted"
    assert (
        df.index.to_series().diff().value_counts().iloc[0] + 1 == df.shape[0]
    ), "Expect regular sample intervals"

    # If offset_steps is in the future
    if offset_steps < 0:
        offset_steps = period_steps + offset_steps
    window_size = offset_steps + period_steps * repeat

    col_name = df.name
    return pd.concat(
        [
            df.rolling(
                window=window_size,
                closed="right",
                min_periods=offset_steps + 1,
            )
            .agg(lambda x: x.iloc[-offset_steps - 1 :: -period_steps].apply(m))
            .rename(f"{col_name}_period{repeat}D_{m}{suffix}")
            for m in metrics
        ],
        axis=1,
    )


def periodic_features(
    df: pd.Series,
    period_steps: int,
    repeats: List[int] = [1, 2, 5, 7, 14, 28],
    shift_steps: int = 0,
    metrics: List[str] = ["min", "max", "mean", "count"],
    suffix: str = "",
) -> pd.DataFrame:
    """
    Periodic feature generation for time series data.

    Args:
        df: Series with sorted datetime index
        period_steps: Number of time steps between reoccuring windows
        repeats: Number of re-occuring time stamps per feature
        shift_steps: Number of time steps to shift the window into the future
        metrics: Metrics calculated on selected data
    """
    assert len(df.shape) == 1, "Expect Series"
    assert df.index.is_monotonic_increasing, "Index must be sorted"
    assert (
        df.index.to_series().diff().value_counts().iloc[0] + 1 == df.shape[0]
    ), "Expect regular sample intervals"

    return pd.concat(
        [
            _periodic_feature_aggregator(
                df,
                offset_steps=period_steps - shift_steps,
                period_steps=period_steps,
                repeat=repeat,
                metrics=metrics,
                suffix=suffix,
            )
            for repeat in repeats
        ],
        axis=1,
    )


def moving_features(
    df: pd.Series,
    windows: List[str] = ["1H", "2H", "6H", "24H", "7D", "28D"],
    metrics: List[str] = ["min", "max", "mean", "std", "count"],
) -> pd.DataFrame:
    """
    Moving window feature generation for time series data.

    Args:
        df: Series with sorted datetime index
        windows: List of time windows to aggregate over
        metrics: List of aggregation functions to apply to the window

    Returns:
        DataFrame with aggregated features
    """
    assert len(df.shape) == 1, "Expect Series"
    assert df.index.is_monotonic_increasing, "Index must be sorted"

    col_name = df.name
    return pd.concat(
        [
            df.rolling(win, closed="right").agg(agg).rename(f"{col_name}_{win}_{agg}")
            for win in windows
            for agg in metrics
        ],
        axis=1,
    )


def daily_features(ser: pd.Series) -> pd.Series:
    """
    Daily feature generation for time series data.

    Args:
        df: Series with sorted datetime index

    Returns:
        DataFrame with aggregated features
    """
    assert ser.index.is_monotonic_increasing, "Index must be sorted"
    assert isinstance(ser.index, pd.DatetimeIndex)

    col_name = ser.name
    return (
        ser.groupby(ser.index.strftime("%Y-%m-%d"))
        .cumsum()
        .rename(f"{col_name}_daily_cumsum")
    )


def time_features(ser: pd.Series, doy_shift: float) -> pd.DataFrame:
    """
    Derive time features from datetime index.

    Args:
        df: Series or DataFrame with datetime index
        doy_shift: Shift the period (eg use 9 to solstice 21.12.)

    Returns:
        DataFrame with time features
    """
    assert isinstance(ser.index, pd.DatetimeIndex)
    idx = ser.index

    df = pd.DataFrame(
        data={
            "doy": idx.strftime("%j").astype(int),
            "tod_h": idx.strftime("%H").astype(int),
            "tod_min": idx.strftime("%H").astype(int) * 60
            + idx.strftime("%M").astype(int),
        },
        index=idx,
    )

    df["doy_cos"] = -np.cos((df["doy"] + doy_shift) / 365 * 2 * np.pi)
    df["doy_sin"] = -np.sin((df["doy"] + doy_shift) / 365 * 2 * np.pi)
    df["tod_cos"] = -np.cos(df["tod_min"] / 24 / 60 * 2 * np.pi)
    df["tod_sin"] = -np.sin(df["tod_min"] / 24 / 60 * 2 * np.pi)

    return df


def shift_target(df: pd.Series, shift_steps: int) -> pd.Series:
    """
    Shift the target variable into the future.

    Args:
        df: Series with sorted datetime index
        shift: Number of time steps to shift the target into the future

    Returns:
        Series with shifted values as target
    """
    assert len(df.shape) == 1, "Expect Series"
    assert df.index.is_monotonic_increasing, "Index must be sorted"

    return df.shift(-shift_steps).rename("target")
