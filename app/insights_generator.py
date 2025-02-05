import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .models import Transaction, BudgetSetting
from . import db
from collections import defaultdict

class InsightsGenerator:
    def __init__(self):
        self.templates = {
            'spending_increase': "Your spending in {category} has increased by {percent:.1f}% compared to last period.",
            'spending_decrease': "Your spending in {category} has decreased by {percent:.1f}% compared to last period.",
            'unusual_transaction': "Unusual transaction detected in {category}: {amount:.2f} ({description})",
            'frequent_merchant': "You frequently transact with {merchant} ({count} times)",
            'savings_opportunity': "You could save approximately {amount:.2f} by reducing {category} expenses",
            'budget_warning': "You have used {percent:.1f}% of your {category} budget",
            'positive_trend': "Good job! Your {category} spending is {percent:.1f}% below average",
        }
        
    def _prepare_transaction_summary(self, transactions, prev_transactions):
        """Prepare a structured summary of transactions for analysis"""
        current_df = pd.DataFrame([{
            'amount': t.amount,
            'category': t.category,
            'date': t.date,
            'description': t.description
        } for t in transactions])
        
        prev_df = pd.DataFrame([{
            'amount': t.amount,
            'category': t.category,
            'date': t.date,
            'description': t.description
        } for t in prev_transactions])
        
        # Calculate category-wise metrics
        current_by_category = current_df.groupby('category')['amount'].agg(['sum', 'count'])
        prev_by_category = prev_df.groupby('category')['amount'].agg(['sum', 'count'])
        
        insights = []
        
        # Compare spending by category
        for category in current_by_category.index:
            current_spend = current_by_category.loc[category, 'sum']
            prev_spend = prev_by_category.loc[category, 'sum'] if category in prev_by_category.index else 0
            
            if prev_spend > 0:
                change_pct = ((current_spend - prev_spend) / prev_spend) * 100
                if change_pct > 0:
                    insights.append(self.templates['spending_increase'].format(
                        category=category,
                        percent=abs(change_pct)
                    ))
                else:
                    insights.append(self.templates['spending_decrease'].format(
                        category=category,
                        percent=abs(change_pct)
                    ))
        
        # Identify unusual transactions
        category_means = current_df.groupby('category')['amount'].mean()
        category_stds = current_df.groupby('category')['amount'].std()
        
        for category in current_df['category'].unique():
            category_data = current_df[current_df['category'] == category]
            mean = category_means[category]
            std = category_stds[category]
            
            if not pd.isna(std):  # Check if we have enough data points
                unusual_txns = category_data[abs(category_data['amount'] - mean) > 2 * std]
                for _, txn in unusual_txns.iterrows():
                    insights.append(self.templates['unusual_transaction'].format(
                        category=category,
                        amount=abs(txn['amount']),
                        description=txn['description']
                    ))
        
        # Identify frequent merchants
        merchant_counts = current_df['description'].value_counts()
        frequent_merchants = merchant_counts[merchant_counts >= 3]
        for merchant, count in frequent_merchants.items():
            insights.append(self.templates['frequent_merchant'].format(
                merchant=merchant,
                count=count
            ))
        
        return insights
    
    def _analyze_budget_status(self, transactions, budget_settings):
        """Analyze current budget status and generate insights"""
        if not budget_settings:
            return []
        
        insights = []
        df = pd.DataFrame([{
            'amount': t.amount,
            'category': t.category
        } for t in transactions])
        
        category_spending = df.groupby('category')['amount'].sum()
        
        for category, budget in budget_settings.allocations.items():
            if category in category_spending:
                spent = abs(category_spending[category])
                budget_amount = budget * budget_settings.total_budget
                utilization = (spent / budget_amount) * 100
                
                if utilization > 80:
                    insights.append(self.templates['budget_warning'].format(
                        category=category,
                        percent=utilization
                    ))
                elif utilization < 50:
                    insights.append(self.templates['positive_trend'].format(
                        category=category,
                        percent=100 - utilization
                    ))
                
                if utilization > 100:
                    potential_savings = spent - budget_amount
                    insights.append(self.templates['savings_opportunity'].format(
                        amount=potential_savings,
                        category=category
                    ))
        
        return insights
    
    def generate_insights(self, days=30):
        """Generate personalized financial insights."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        prev_start_date = start_date - timedelta(days=days)
        
        # Get current and previous period transactions
        current_transactions = Transaction.query.filter(
            Transaction.date.between(start_date, end_date)
        ).all()
        
        prev_transactions = Transaction.query.filter(
            Transaction.date.between(prev_start_date, start_date)
        ).all()
        
        if not current_transactions:
            return {
                'status': 'error',
                'message': 'No transaction data available for analysis'
            }
        
        # Get budget settings
        budget_settings = BudgetSetting.query.first()
        
        # Generate insights
        transaction_insights = self._prepare_transaction_summary(
            current_transactions,
            prev_transactions
        )
        
        budget_insights = self._analyze_budget_status(
            current_transactions,
            budget_settings
        )
        
        all_insights = transaction_insights + budget_insights
        
        return {
            'status': 'success',
            'insights': all_insights,
            'analysis_period_days': days
        }
