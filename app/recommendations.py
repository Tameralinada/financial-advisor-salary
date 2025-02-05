import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from .models import Transaction, BudgetSetting
from . import db

def get_financial_recommendations():
    """Generate personalized financial recommendations based on transaction history."""
    # Get recent transactions
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    transactions = Transaction.query.filter(
        Transaction.date.between(start_date, end_date)
    ).all()
    
    if not transactions:
        return {
            'status': 'error',
            'message': 'Not enough transaction data for recommendations'
        }
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': t.date,
        'amount': t.amount,
        'category': t.category
    } for t in transactions])
    
    # Get budget settings
    budget_settings = BudgetSetting.query.first()
    
    recommendations = []
    insights = analyze_spending_patterns(df)
    
    # Spending pattern recommendations
    for category, insight in insights.items():
        if insight['trend'] == 'increasing':
            recommendations.append({
                'type': 'warning',
                'category': category,
                'message': f'Your spending in {category} has increased by {insight["change_percent"]:.1f}% recently. Consider setting a budget limit.'
            })
        elif insight['trend'] == 'high_variance':
            recommendations.append({
                'type': 'info',
                'category': category,
                'message': f'Your {category} spending is highly variable. Consider planning these expenses better.'
            })
    
    # Savings recommendations
    if budget_settings:
        savings_rate = analyze_savings_rate(df, budget_settings)
        if savings_rate < 0.1:  # Less than 10% savings
            recommendations.append({
                'type': 'warning',
                'category': 'Savings',
                'message': 'Your savings rate is below 10%. Consider reducing non-essential expenses.'
            })
    
    # Debt recommendations
    debt_insights = analyze_debt_payments(df)
    if debt_insights['high_interest_debt']:
        recommendations.append({
            'type': 'alert',
            'category': 'Debt',
            'message': 'Consider prioritizing high-interest debt payments to reduce interest costs.'
        })
    
    return {
        'status': 'success',
        'recommendations': recommendations
    }

def analyze_spending_patterns(df):
    """Analyze spending patterns by category."""
    insights = {}
    
    for category in df['category'].unique():
        category_data = df[df['category'] == category]
        
        if len(category_data) < 2:
            continue
        
        # Calculate trend
        recent_avg = category_data['amount'].tail(10).mean()
        old_avg = category_data['amount'].head(10).mean()
        
        if recent_avg > old_avg:
            trend = 'increasing'
            change_percent = ((recent_avg - old_avg) / old_avg) * 100
        else:
            trend = 'stable'
            change_percent = 0
        
        # Calculate variance
        variance = category_data['amount'].std()
        mean = category_data['amount'].mean()
        cv = variance / mean if mean != 0 else 0
        
        insights[category] = {
            'trend': 'high_variance' if cv > 0.5 else trend,
            'change_percent': change_percent,
            'coefficient_of_variation': cv
        }
    
    return insights

def analyze_savings_rate(df, budget_settings):
    """Calculate savings rate based on income and expenses."""
    income = df[df['amount'] > 0]['amount'].sum()
    expenses = abs(df[df['amount'] < 0]['amount'].sum())
    
    if income > 0:
        return (income - expenses) / income
    return 0

def analyze_debt_payments(df):
    """Analyze debt-related transactions."""
    debt_categories = ['Credit Card', 'Loan Payment', 'Mortgage']
    debt_transactions = df[df['category'].isin(debt_categories)]
    
    return {
        'high_interest_debt': len(debt_transactions) > 0,
        'total_debt_payments': abs(debt_transactions['amount'].sum())
    }
