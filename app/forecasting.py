import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from .models import Transaction
from . import db

def get_expense_forecast(days_to_forecast=30):
    """Get expense forecasts for all categories."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Use last year of data for training
    future_dates = [end_date + timedelta(days=x) for x in range(1, days_to_forecast + 1)]

    # Get historical transactions
    transactions = Transaction.query.filter(
        Transaction.date >= start_date,
        Transaction.date <= end_date
    ).order_by(Transaction.date).all()

    if not transactions:
        return None

    # Prepare data by category
    df = pd.DataFrame([{
        'date': t.date,
        'amount': t.amount,
        'category': t.category
    } for t in transactions])

    forecasts = {}
    for category in df['category'].unique():
        category_data = df[df['category'] == category].copy()
        category_data = category_data.set_index('date')
        category_data = category_data.resample('D').sum().fillna(0)

        transactions_data = category_data.reset_index().values.tolist()
        forecast = get_expense_forecast(transactions_data, forecast_days=days_to_forecast)
        
        forecasts[category] = {
            'dates': forecast['dates'],
            'predictions': forecast['values'],
            'lower_bound': forecast['lower_bound'],
            'upper_bound': forecast['upper_bound']
        }

    return {
        'forecasts': forecasts,
        'forecast_period_days': days_to_forecast,
        'base_date': end_date.strftime('%Y-%m-%d')
    }

def get_expense_forecast(transactions_data, forecast_days=30):
    """
    Generate expense forecasts using Prophet and ARIMA models
    """
    # Convert transactions to time series
    df = pd.DataFrame(transactions_data)
    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df['amount']
    
    # Prophet forecast
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=forecast_days)
    prophet_forecast = m.predict(future)
    
    # ARIMA forecast
    model = ARIMA(df['y'], order=(1,1,1))
    arima_model = model.fit()
    arima_forecast = arima_model.forecast(steps=forecast_days)
    
    # Combine forecasts (simple average)
    final_forecast = (prophet_forecast['yhat'].tail(forecast_days).values + 
                     arima_forecast.values) / 2
    
    forecast_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                     for i in range(forecast_days)]
    
    return {
        'dates': forecast_dates,
        'values': final_forecast.tolist(),
        'lower_bound': prophet_forecast['yhat_lower'].tail(forecast_days).tolist(),
        'upper_bound': prophet_forecast['yhat_upper'].tail(forecast_days).tolist()
    }
