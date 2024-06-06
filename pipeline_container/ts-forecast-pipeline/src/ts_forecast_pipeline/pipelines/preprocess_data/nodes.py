import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

def load_data(data:pd.DataFrame) -> pd.DataFrame:
    '''
    Retrive data from DB and standarize names of columns.
    Create pivot table to get time series by company ID.

    Args:
        None

    Returns:
        pivot_data [pd.DataFrame]: Dataframe containing time series by company_id.
    '''

    data = pd.read_sql("SELECT Fecha_hoy, ID_empresa, SUM(ing_hab) AS ing_hab FROM Ocupaciones GROUP BY Fecha_hoy, ID_empresa;", conn)
    
    data = data.drop(data.columns[0], axis=1)
    data.columns = ['date', 'company', 'data']
    data['date'] = pd.to_datetime(data['date'])

    pivot_data = data.pivot(index='date', columns='company', values='data')
    pivot_data.columns = [f'company_{col}' for col in pivot_data.columns]
    pivot_data.reset_index(inplace=True)

    return pivot_data



def fill_missing_dates(data:pd.DataFrame) -> pd.DataFrame:
    '''
    Creates empty rows for dates that are missing in the time series.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the date column and a columns for each company ID.
    
    Returns:
        pd.DataFrame: DataFrame with missing dates filled with null values.
    '''

    date_column = 'date'
    
    # Ensure the date column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Set the date column as the index
    data.set_index(date_column, inplace=True)
    
    # Create a complete date range from the min to the max date in the DataFrame
    complete_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    
    # Reindex the DataFrame to this complete date range
    df_reindexed = data.reindex(complete_date_range)
    
    return df_reindexed



def handle_outliers(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Identify and handle outliers.
    Replaces outliers by maximum or minimum value calculated by a distance of 3
    standard deviations from the mean.

    Args:
        dataframe [pd.DataFrame]: Columns for each company ID as independent 
                                  time series, date index.


    Returns:
        [pd.DataFrame]: Same columns, outliers replaced by NaN values.


    '''
    data_no_outliers = []
    for company in data.columns:
        data_by_company = data[company]
        mean = np.mean(data_by_company)
        std_dev = np.std(data_by_company)
        lower_limit = mean - 3 * std_dev
        upper_limit = mean + 3 * std_dev

        data_capped = np.clip(data_by_company, lower_limit, upper_limit)
        data_no_outliers.append(data_capped)

    data_no_outliers = pd.concat(data_no_outliers, axis=1, keys=data.columns)
    data_no_outliers

    return data_no_outliers



def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Fill missing data for each company's time series.

    Args:
        dataframe [pd.DataFrame]: Columns for each company ID as independent 
                                time series, date index.
    Returns:
        [pd.DataFrame]: Same columns, missing data filled.
    '''
    for column in df.columns:
        df_revenue = df[column].to_frame()
        decomposition = seasonal_decompose(df[column].dropna(), period=365)
        df_revenue['trend'] = decomposition._trend
        df_revenue['seasonal'] = decomposition._seasonal
        df_revenue['residual'] = decomposition._resid
        df_revenue['month'] = df_revenue.index.month
        df_revenue['day'] = df_revenue.index.day

    revenue = df[column]
    left = 0
    start = 0
    for i in range(len(revenue)):
        right = i
        if not np.isnan(revenue.iloc[right]) and not np.isnan(revenue.iloc[left]) and np.isnan(revenue.iloc[right - 1]):
            df_revenue.loc[df_revenue.index[left-10:right+10], 'trend'] = df_revenue.loc[df_revenue.index[left-10:right+10], 'trend'].interpolate(method='spline', order=3, limit_direction='both')

            # Calculate residuals and seasonal means
            residuals = df_revenue.groupby(['month', 'day'])['residual'].mean()
            seasonal = df_revenue.groupby(['month', 'day'])['seasonal'].mean()

            # Extract the range of residuals and seasonal values
            start_month_day = (df_revenue['month'].iloc[left+1], df_revenue['day'].iloc[left+1])
            end_month_day = (df_revenue['month'].iloc[right-1], df_revenue['day'].iloc[right-1])

            # Slicing by index from the grouped means
            residual_vals = residuals.loc[start_month_day:end_month_day].values
            seasonal_vals = seasonal.loc[start_month_day:end_month_day].values

            # Calculate new values for the nulls
            trend_vals = df_revenue['trend'].iloc[left+1:right].values
            nulos = residual_vals + seasonal_vals + trend_vals

            revenue.iloc[left+1:right] = nulos
        if left == 0 and not np.isnan(revenue.iloc[right]):
            start = right
        if (not np.isnan(revenue.iloc[right]) and np.isnan(revenue.iloc[left])) or not np.isnan(revenue.iloc[right]):
            left = right

    persistent_nulls = revenue[start:][revenue[start:].isnull()].index.tolist()

    for indexes in persistent_nulls:
        revenue[indexes] = (revenue[indexes + pd.Timedelta(days=1)] + revenue[indexes - pd.Timedelta(days=1)])/2
        df[column] = revenue

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    
    return df