import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from .models import Transaction, BudgetSetting, BudgetAdjustment
from . import db

def get_budget_adjustments():
    """Get budget adjustment recommendations based on spending patterns."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    transactions = Transaction.query.filter(
        Transaction.date.between(start_date, end_date)
    ).all()
    
    if not transactions:
        return {
            'status': 'error',
            'message': 'Not enough transaction data for adjustments'
        }
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': t.date,
        'amount': t.amount,
        'category': t.category
    } for t in transactions])
    
    # Get current budget settings
    budget_settings = BudgetSetting.query.first()
    if not budget_settings:
        return {
            'status': 'error',
            'message': 'No budget settings found'
        }
    
    adjustments = []
    
    # Analyze each category
    for category in df['category'].unique():
        category_data = df[df['category'] == category]
        if len(category_data) < 5:  # Need minimum data points
            continue
        
        # Calculate statistics
        avg_spending = abs(category_data['amount'].mean())
        total_spending = abs(category_data['amount'].sum())
        spending_std = category_data['amount'].std()
        
        # Detect anomalies using Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(category_data[['amount']])
        anomaly_ratio = (anomalies == -1).mean()
        
        # Generate adjustments based on patterns
        if anomaly_ratio > 0.2:  # High number of anomalies
            adjustments.append({
                'category': category,
                'type': 'warning',
                'message': f'Irregular spending patterns detected in {category}. Consider setting a stricter budget.',
                'suggested_limit': avg_spending * 1.2  # 20% buffer
            })
        elif spending_std > avg_spending * 0.5:  # High volatility
            adjustments.append({
                'category': category,
                'type': 'info',
                'message': f'High spending variability in {category}. Consider smoothing out expenses.',
                'suggested_limit': avg_spending * 1.5  # 50% buffer
            })
        elif total_spending > budget_settings.monthly_limit * 0.3:  # High proportion of budget
            adjustments.append({
                'category': category,
                'type': 'alert',
                'message': f'{category} expenses are taking up a large portion of your budget.',
                'suggested_limit': total_spending * 0.8  # 20% reduction
            })
    
    # Store adjustments in database
    for adj in adjustments:
        db.session.add(BudgetAdjustment(
            category=adj['category'],
            type=adj['type'],
            message=adj['message'],
            suggested_limit=adj['suggested_limit'],
            date=datetime.now()
        ))
    db.session.commit()
    
    return {
        'status': 'success',
        'adjustments': adjustments
    }
