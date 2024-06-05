import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

class HoltWinters:
    def __init__(self, dataframe, aggregation):
        frequencies = {'daily':'D','weekly':'W','biweekly':'2W','monthly':'M'}
        cicle = {'daily':365,'weekly':52,'biweekly':24,'monthly':12}
        
        self.time_series_grouped= dataframe
        self.aggregation_freq = frequencies[aggregation]
        self.cicle = cicle[aggregation]
        self.series = self.time_series_grouped['data']
        self.metrics = None
        
    def get_predictions(self, n_predictions):
        train, test = self.series[0 : len(self.series) - n_predictions], self.series[len(self.series) - n_predictions : len(self.series)]
        model_train = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods = self.cicle, use_boxcox=False, initialization_method="estimated")
        final_model = ExponentialSmoothing(self.series, trend='add', seasonal='add', seasonal_periods = self.cicle, use_boxcox=False, initialization_method="estimated")

        model_train_fit = model_train.fit()
        final_model_fit = final_model.fit()

        test_forecast = model_train_fit.forecast(len(test))
        total_forecast = final_model_fit.forecast(n_predictions)

        last_date = self.time_series_grouped['date'].max()
        next_dates = pd.date_range(start=last_date, periods=n_predictions+1, freq=self.aggregation_freq)[1:]  
        
        self.forecast = pd.DataFrame({'date':next_dates.tolist(), 'forecast':total_forecast})
        self.forecast.set_index('date', inplace=True)

        # Metrics
        MAE = mean_absolute_error(test, test_forecast)
        MAPE = mean_absolute_percentage_error(test, test_forecast)
        RMSE = np.sqrt(mean_squared_error(test, test_forecast))

        self.metrics = {'MAE':MAE, 'MAPE':MAPE, 'RMSE':RMSE}

        return self.forecast
