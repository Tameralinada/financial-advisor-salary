import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime, timedelta
from .models import Transaction
from . import db

class BudgetOptimizer:
    def __init__(self):
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.dt_model = DecisionTreeRegressor(random_state=42)
        self.xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, transactions):
        """Convert transactions into features for ML models."""
        df = pd.DataFrame([{
            'amount': t.amount,
            'category': t.category,
            'date': t.date,
            'month': t.date.month,
            'day_of_week': t.date.weekday(),
            'day_of_month': t.date.day
        } for t in transactions])
        
        # Create category-wise spending features
        pivot_df = pd.pivot_table(
            df, 
            values='amount', 
            index='date',
            columns='category',
            aggfunc='sum',
            fill_value=0
        )
        
        # Add time-based features
        pivot_df['month'] = pd.to_datetime(pivot_df.index).month
        pivot_df['day_of_week'] = pd.to_datetime(pivot_df.index).dayofweek
        pivot_df['day_of_month'] = pd.to_datetime(pivot_df.index).day
        
        return pivot_df

    def train_models(self, transactions, target_category):
        """Train ML models to predict spending in a specific category."""
        if not transactions:
            return False
            
        df = self.prepare_features(transactions)
        
        # Prepare features and target
        X = df.drop(columns=[target_category])
        y = df[target_category]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.rf_model.fit(X_train_scaled, y_train)
        self.dt_model.fit(X_train_scaled, y_train)
        self.xgb_model.fit(X_train_scaled, y_train)
        
        return True

    def predict_overspending(self, transactions, category, budget_limit):
        """Predict if a category will exceed its budget."""
        if not transactions:
            return None
            
        df = self.prepare_features(transactions)
        X = df.drop(columns=[category])
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        rf_pred = self.rf_model.predict(X_scaled)[-1]
        dt_pred = self.dt_model.predict(X_scaled)[-1]
        xgb_pred = self.xgb_model.predict(X_scaled)[-1]
        
        # Ensemble prediction (average of all models)
        ensemble_pred = np.mean([rf_pred, dt_pred, xgb_pred])
        
        return {
            'predicted_amount': ensemble_pred,
            'will_exceed_budget': ensemble_pred > budget_limit,
            'predicted_excess': max(0, ensemble_pred - budget_limit),
            'model_predictions': {
                'random_forest': rf_pred,
                'decision_tree': dt_pred,
                'xgboost': xgb_pred
            }
        }

    def get_optimization_suggestions(self, transactions, current_allocations):
        """Generate budget optimization suggestions based on spending patterns."""
        suggestions = []
        total_budget = sum(current_allocations.values())
        
        for category, budget in current_allocations.items():
            prediction = self.predict_overspending(transactions, category, budget)
            
            if prediction and prediction['will_exceed_budget']:
                excess = prediction['predicted_excess']
                
                # Find categories with potential savings
                savings_candidates = {}
                for other_category, other_budget in current_allocations.items():
                    if other_category != category:
                        other_prediction = self.predict_overspending(transactions, other_category, other_budget)
                        if other_prediction and not other_prediction['will_exceed_budget']:
                            potential_saving = other_budget - other_prediction['predicted_amount']
                            if potential_saving > 0:
                                savings_candidates[other_category] = potential_saving
                
                if savings_candidates:
                    suggestion = {
                        'category': category,
                        'predicted_overspend': excess,
                        'reallocation_suggestions': []
                    }
                    
                    remaining_excess = excess
                    for save_category, potential_saving in savings_candidates.items():
                        if remaining_excess <= 0:
                            break
                            
                        amount_to_move = min(potential_saving, remaining_excess)
                        suggestion['reallocation_suggestions'].append({
                            'from_category': save_category,
                            'amount': amount_to_move,
                            'original_budget': current_allocations[save_category],
                            'new_budget': current_allocations[save_category] - amount_to_move
                        })
                        remaining_excess -= amount_to_move
                    
                    suggestions.append(suggestion)
        
        return suggestions

def get_budget_insights(days=30):
    """Get AI-powered budget insights for the specified time period."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get recent transactions
    transactions = Transaction.query.filter(
        Transaction.date >= start_date,
        Transaction.date <= end_date
    ).all()
    
    optimizer = BudgetOptimizer()
    
    # Get current budget allocations from the database
    current_allocations = {
        category: float(amount) 
        for category, amount in db.session.query(
            Transaction.category, 
            db.func.sum(Transaction.amount)
        ).group_by(Transaction.category).all()
    }
    
    # Train models for each category
    for category in current_allocations.keys():
        optimizer.train_models(transactions, category)
    
    # Get optimization suggestions
    suggestions = optimizer.get_optimization_suggestions(transactions, current_allocations)
    
    return {
        'suggestions': suggestions,
        'analysis_period_days': days,
        'total_transactions_analyzed': len(transactions)
    }
