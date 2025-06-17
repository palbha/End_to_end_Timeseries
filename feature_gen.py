import pandas as pd
from typing import Optional
import numpy as np

def create_date_features(df, date_col):
    """
    Create basic date-based features from a datetime column.

    Args:
        df (pd.DataFrame): Input dataframe containing the date column.
        date_col (str): Name of the datetime column in df.

    Returns:
        pd.DataFrame: DataFrame with new date features added.
            Features include:
                - year
                - month
                - day
                - day_of_week (0=Monday, 6=Sunday)
                - is_weekend (bool)
                - quarter
                - day_of_year
                - week_of_year
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['day_of_week'] = dt.dt.dayofweek
    df['is_weekend'] = dt.dt.dayofweek >= 5
    df['quarter'] = dt.dt.quarter
    df['day_of_year'] = dt.dt.dayofyear
    df['week_of_year'] = dt.dt.isocalendar().week
    return df



def create_holiday_features(
    df,
    date_col,
    holidays_df=None,
    holiday_flag_col=None,
    use_holiday_library=False,
    country=None,
    state=None,
    prov=None,
):
    """
    Create holiday-related features using either:
    - a holiday DataFrame,
    - a holiday flag column,
    - or the 'holidays' Python library if use_holiday_library=True.

    Additional features include:
        - week_with_holiday (bool): True if the week contains at least one holiday.
        - long_weekend (bool): True if weekend extended by holiday on Friday or Monday.
        - bridge_day (bool): True if holiday is on Tuesday or Thursday (a day bridging weekend).

    Args:
        df (pd.DataFrame): Input dataframe with date column.
        date_col (str): Name of the datetime column in df.
        holidays_df (pd.DataFrame, optional): DataFrame with columns:
            'holiday_date' (datetime),
            'holiday_name' (str, optional),
            'holiday_type' (str, optional).
        holiday_flag_col (str, optional): Column name in df indicating holiday bool flag.
        use_holiday_library (bool): If True, use 'holidays' python library to identify holidays.
        country (str, optional): Country code for holidays library (e.g. 'US', 'CA').
        state/prov (str, optional): State or province code for holidays library (e.g. 'CA' for California).

    Returns:
        pd.DataFrame: DataFrame with holiday features added:
            - is_holiday (bool)
            - holiday_name (str)
            - holiday_type (str)
            - days_to_next_holiday (int)
            - days_since_last_holiday (int)
            - week_with_holiday (bool)
            - long_weekend (bool)
            - bridge_day (bool)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Initialize columns
    df['is_holiday'] = False
    df['holiday_name'] = np.nan
    df['holiday_type'] = np.nan

    # 1) Mark holidays from holidays_df if provided
    if holidays_df is not None:
        holidays_df = holidays_df.copy()
        holidays_df['holiday_date'] = pd.to_datetime(holidays_df['holiday_date'])
        holiday_dates = holidays_df['holiday_date'].sort_values().unique()
        df['is_holiday'] = df[date_col].isin(holiday_dates)

        holiday_map = holidays_df.set_index('holiday_date')[['holiday_name', 'holiday_type']].to_dict('index')

        def append_holiday_name(d):
            if d in holiday_map:
                base_name = holiday_map[d]['holiday_name'] or ""
                # if already marked holiday, append reason with '_'
                if df.loc[df[date_col] == d, 'is_holiday'].any():
                    return base_name
                else:
                    return np.nan
            return np.nan

        # Overwrite holiday_name where is_holiday is True
        df.loc[df['is_holiday'], 'holiday_name'] = df.loc[df['is_holiday'], date_col].map(
            lambda d: holiday_map.get(d, {}).get('holiday_name', np.nan)
        )
        df.loc[df['is_holiday'], 'holiday_type'] = df.loc[df['is_holiday'], date_col].map(
            lambda d: holiday_map.get(d, {}).get('holiday_type', np.nan)
        )

    # 2) Override/add holidays from holiday_flag_col if provided
    if holiday_flag_col is not None and holiday_flag_col in df.columns:
        df.loc[df[holiday_flag_col], 'is_holiday'] = True
        df.loc[df[holiday_flag_col], 'holiday_name'] = df.loc[df[holiday_flag_col], 'holiday_name'].fillna('Flagged_Holiday')

    # 3) Use python holidays library if specified
    if use_holiday_library:
        try:
            import holidays
        except ImportError:
            raise ImportError("The 'holidays' package is required when use_holiday_library=True. Install via 'pip install holidays'.")

        # Prepare holidays object based on inputs
        if country is None:
            raise ValueError("Country must be specified when use_holiday_library=True")
        if state:
            hols = holidays.CountryHoliday(country, prov=state)
        elif prov:
            hols = holidays.CountryHoliday(country, prov=prov)
        else:
            hols = holidays.CountryHoliday(country)

        # Mark holidays from holidays lib
        for idx, dt in df[date_col].iteritems():
            if dt in hols:
                df.at[idx, 'is_holiday'] = True
                old_name = df.at[idx, 'holiday_name']
                hol_name = hols.get(dt)
                # Append with underscore if already a holiday reason exists
                if pd.isna(old_name):
                    df.at[idx, 'holiday_name'] = hol_name
                else:
                    df.at[idx, 'holiday_name'] = f"{old_name}_{hol_name}"

    # Fill any NaNs in holiday_name/type to empty string to avoid issues
    df['holiday_name'] = df['holiday_name'].fillna('')
    df['holiday_type'] = df['holiday_type'].fillna('')

    # Compute days_to_next_holiday and days_since_last_holiday
    holiday_dates_sorted = df.loc[df['is_holiday'], date_col].sort_values().unique()
    if len(holiday_dates_sorted) > 0:
        # Use searchsorted for vectorized next/previous holiday
        next_idx = np.searchsorted(holiday_dates_sorted, df[date_col], side='left')
        prev_idx = np.searchsorted(holiday_dates_sorted, df[date_col], side='right') - 1

        def days_to_next(i, dt):
            if i < len(holiday_dates_sorted):
                return (holiday_dates_sorted[i] - dt).days
            else:
                return np.nan

        def days_since_prev(i, dt):
            if i >= 0:
                return (dt - holiday_dates_sorted[i]).days
            else:
                return np.nan

        df['days_to_next_holiday'] = [days_to_next(i, dt) for i, dt in zip(next_idx, df[date_col])]
        df['days_since_last_holiday'] = [days_since_prev(i, dt) for i, dt in zip(prev_idx, df[date_col])]
    else:
        df['days_to_next_holiday'] = np.nan
        df['days_since_last_holiday'] = np.nan

    # Calculate additional flags:

    # week_with_holiday - True if at least one holiday in the ISO week
    df['week'] = df[date_col].dt.isocalendar().week
    df['year'] = df[date_col].dt.year
    # Group by year+week and check if any holiday present
    week_holiday = df.groupby(['year', 'week'])['is_holiday'].transform('max').astype(bool)
    df['week_with_holiday'] = week_holiday

    # long_weekend - True if weekend + holiday extends weekend
    # Criteria: Sat or Sun is holiday OR (Fri or Mon is holiday adjacent to weekend)
    df['day_of_week'] = df[date_col].dt.dayofweek  # Monday=0, Sunday=6

    # Helper to check if holiday on Friday or Monday
    # Long weekend flags if Sat/Sun is holiday or Fri or Mon is holiday adjacent
    df['long_weekend'] = False
    # Sat=5, Sun=6, Fri=4, Mon=0
    is_holiday = df['is_holiday']
    dow = df['day_of_week']

    # For each row, check if weekend or adjacent holiday to weekend
    for idx, row in df.iterrows():
        if row['is_holiday']:
            # Check if this day is Fri or Mon, or Sat/Sun
            if row['day_of_week'] in [5, 6]:  # Sat or Sun holiday
                df.at[idx, 'long_weekend'] = True
            elif row['day_of_week'] == 4:  # Friday holiday, check if Sat/Sun follows
                next_days = df.loc[(df[date_col] > row[date_col]) & (df[date_col] <= row[date_col] + pd.Timedelta(days=2))]
                if any(next_days['day_of_week'].isin([5,6])):
                    df.at[idx, 'long_weekend'] = True
            elif row['day_of_week'] == 0:  # Monday holiday, check if Fri/Sat/Sun precedes
                prev_days = df.loc[(df[date_col] < row[date_col]) & (df[date_col] >= row[date_col] - pd.Timedelta(days=3))]
                if any(prev_days['day_of_week'].isin([4,5,6])):
                    df.at[idx, 'long_weekend'] = True

    # bridge_day - True if holiday on Tuesday or Thursday
    df['bridge_day'] = df['is_holiday'] & df['day_of_week'].isin([1, 3])  # Tuesday=1, Thursday=3

    # Clean-up helper columns
    df.drop(columns=['week', 'year', 'day_of_week'], inplace=True)

    return df

def create_lag_features(df, target_col, lags):
    """
    Create lag features for a target time series column.

    Args:
        df (pd.DataFrame): Input dataframe sorted by time.
        target_col (str): Name of the target column to create lags for.
        lags (list of int): List of lag periods (in rows) to create.

    Returns:
        pd.DataFrame: DataFrame with lag features added.
            Columns named as '{target_col}_lag_{lag}'.
    """
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df


def create_rolling_features(df, target_col, windows, funcs=['mean', 'sum', 'std', 'min', 'max']):
    """
    Create rolling window features for a target column.

    Args:
        df (pd.DataFrame): Input dataframe sorted by time.
        target_col (str): Name of the target column.
        windows (list of int): List of window sizes (number of rows).
        funcs (list of str): Aggregation functions to apply: 'mean', 'sum', 'std', 'min', 'max'.

    Returns:
        pd.DataFrame: DataFrame with rolling features added.
            Columns named as '{target_col}_roll_{func}_{window}'.
    """
    df = df.copy()
    for window in windows:
        roll = df[target_col].rolling(window=window)
        for func in funcs:
            col_name = f'{target_col}_roll_{func}_{window}'
            if func == 'mean':
                df[col_name] = roll.mean()
            elif func == 'sum':
                df[col_name] = roll.sum()
            elif func == 'std':
                df[col_name] = roll.std()
            elif func == 'min':
                df[col_name] = roll.min()
            elif func == 'max':
                df[col_name] = roll.max()
            else:
                raise ValueError(f"Unsupported rolling function: {func}")
    return df



def create_cyclical_features(df, date_col, periods):
    """
    Create cyclical (sin/cos) features for multiple time periods.

    Args:
        df (pd.DataFrame): Input dataframe.
        date_col (str): Date column (datetime).
        periods (list of tuples): List of (period, prefix) pairs.
            For example: [(7, 'day_of_week'), (12, 'month'), (365, 'day_of_year')]

    Returns:
        pd.DataFrame: DataFrame with '{prefix}_sin' and '{prefix}_cos' features added.
    """
    df = df.copy()

    for period, prefix in periods:
        if period == 7:
            time_vals = df[date_col].dt.dayofweek  # Monday=0 .. Sunday=6
        elif period == 12:
            time_vals = df[date_col].dt.month - 1  # month 1-12 mapped to 0-11
        elif period == 24:
            if hasattr(df[date_col].dt, 'hour'):
                time_vals = df[date_col].dt.hour
            else:
                raise ValueError("Date column has no hour attribute for 24-hour period.")
        elif period == 365:
            time_vals = df[date_col].dt.dayofyear - 1  # day_of_year 1-365 mapped to 0-364
        else:
            # fallback: use modulo of day for custom periods
            time_vals = df[date_col].dt.day % period

        radians = 2 * np.pi * time_vals / period
        df[f'{prefix}_sin'] = np.sin(radians)
        df[f'{prefix}_cos'] = np.cos(radians)

    return df



def create_event_flags(df, date_col, events_df, event_date_col='event_date', event_name_col='event_name'):
    """
    Add boolean flags for special events.

    Args:
        df (pd.DataFrame): Input dataframe with date column.
        date_col (str): Date column in df.
        events_df (pd.DataFrame): DataFrame containing event dates and optional event names.
        event_date_col (str): Column in events_df with event dates.
        event_name_col (str): Column in events_df with event names.

    Returns:
        pd.DataFrame: DataFrame with new boolean columns named after events, indicating if date matches event.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    events_df = events_df.copy()
    events_df[event_date_col] = pd.to_datetime(events_df[event_date_col])

    for event_name, group in events_df.groupby(event_name_col):
        event_dates = group[event_date_col].unique()
        col_name = f'is_event_{event_name.lower().replace(" ", "_")}'
        df[col_name] = df[date_col].isin(event_dates)
    return df


def create_month_quarter_flags(df, date_col):
    """
    Create flags for start/end of month and quarter.

    Args:
        df (pd.DataFrame): Input dataframe with date column.
        date_col (str): Date column name.

    Returns:
        pd.DataFrame: DataFrame with boolean flags:
            - is_start_of_month
            - is_end_of_month
            - is_start_of_quarter
            - is_end_of_quarter
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df['is_start_of_month'] = df[date_col].dt.is_month_start
    df['is_end_of_month'] = df[date_col].dt.is_month_end
    df['is_start_of_quarter'] = df[date_col].dt.is_quarter_start
    df['is_end_of_quarter'] = df[date_col].dt.is_quarter_end
    return df


def create_temporal_historical_avg_feature(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    anchor_type: str = "date",  # options: "date", "week", "month"
    lookback_years: int = 3,
) -> pd.DataFrame:
    """
    Creates historical average features based on calendar anchors (date/week/month).

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of datetime column.
        target_col (str): Target column to compute averages on.
        anchor_type (str): One of ["date", "week", "month"].
        lookback_years (int): How many past years to average.

    Returns:
        pd.DataFrame: DataFrame with new historical average feature added.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year

    if anchor_type == "date":
        df['month_day'] = df[date_col].dt.strftime('%m-%d')
        group_keys = ['month_day', 'year']
        anchor_val_col = 'month_day'
    elif anchor_type == "week":
        df['iso_week'] = df[date_col].dt.isocalendar().week
        group_keys = ['iso_week', 'year']
        anchor_val_col = 'iso_week'
    elif anchor_type == "month":
        df['month'] = df[date_col].dt.month
        group_keys = ['month', 'year']
        anchor_val_col = 'month'
    else:
        raise ValueError(f"Unsupported anchor_type: {anchor_type}")

    grouped = df.groupby(group_keys)[target_col].mean().reset_index(name='mean_target')
    grouped.set_index(group_keys, inplace=True)

    def hist_avg(row):
        anchor = row[anchor_val_col]
        year = row['year']
        past_years = [year - i for i in range(1, lookback_years + 1)]
        vals = []
        for y in past_years:
            try:
                val = grouped.loc[(anchor, y), 'mean_target']
                vals.append(val)
            except KeyError:
                pass
        return np.mean(vals) if vals else pd.NA

    new_col = f"{anchor_type}_historical_avg_{lookback_years}yr"
    df[new_col] = df.apply(hist_avg, axis=1)

    df.drop(columns=[anchor_val_col, 'year'], inplace=True)
    return df
