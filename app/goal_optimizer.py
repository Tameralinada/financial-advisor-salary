import numpy as np
from scipy.optimize import linprog
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple
from .models import Transaction, BudgetSetting

class GoalOptimizer:
    def __init__(self):
        # Default constraints
        self.min_category_percent = 0.05  # Minimum 5% of total budget for each category
        self.max_category_percent = 0.50  # Maximum 50% of total budget for each category
        self.essential_categories = ['Housing', 'Utilities', 'Groceries', 'Healthcare']
        self.essential_min_percent = 0.15  # Minimum 15% for essential categories
        
    def _get_historical_spending(self, user_id: int, days: int = 90) -> pd.DataFrame:
        """Get historical spending patterns."""
        start_date = datetime.now() - timedelta(days=days)
        
        # Get transactions from database
        transactions = Transaction.query.filter(
            Transaction.user_id == user_id,
            Transaction.date >= start_date
        ).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': t.date,
            'amount': t.amount,
            'category': t.category
        } for t in transactions])
        
        return df
        
    def _calculate_category_bounds(self, df: pd.DataFrame, total_budget: float) -> Dict[str, Tuple[float, float]]:
        """Calculate min and max bounds for each category based on historical data."""
        category_bounds = {}
        
        # Calculate average monthly spend per category
        monthly_spend = df.groupby('category')['amount'].sum() / 3  # Assuming 90 days of data
        
        for category, avg_spend in monthly_spend.items():
            if category in self.essential_categories:
                # Essential categories have higher minimum bounds
                min_bound = max(avg_spend * 0.8, total_budget * self.essential_min_percent)
                max_bound = max(avg_spend * 1.5, total_budget * self.max_category_percent)
            else:
                # Non-essential categories are more flexible
                min_bound = max(avg_spend * 0.5, total_budget * self.min_category_percent)
                max_bound = min(avg_spend * 1.2, total_budget * self.max_category_percent)
            
            category_bounds[category] = (min_bound, max_bound)
            
        return category_bounds
        
    def optimize_budget(self, user_id: int, total_budget: float, savings_goal: float,
                       investment_goal: float = 0, debt_payment_goal: float = 0) -> Dict[str, float]:
        """
        Optimize budget allocation based on goals and constraints.
        
        Args:
            user_id: User ID
            total_budget: Total monthly budget
            savings_goal: Target monthly savings amount
            investment_goal: Target monthly investment amount
            debt_payment_goal: Target monthly debt payment amount
            
        Returns:
            Dictionary of optimized category allocations
        """
        # Get historical spending data
        df = self._get_historical_spending(user_id)
        if df.empty:
            return None
            
        # Get unique categories
        categories = df['category'].unique()
        n_categories = len(categories)
        
        # Calculate category bounds
        category_bounds = self._calculate_category_bounds(df, total_budget)
        
        # Set up optimization problem
        
        # Objective: Minimize deviation from historical spending patterns
        historical_weights = df.groupby('category')['amount'].mean()
        historical_weights = historical_weights / historical_weights.sum()  # Normalize
        c = np.array([abs(1 - hw) for hw in historical_weights])
        
        # Constraints matrix
        A_ub = []
        b_ub = []
        
        # 1. Category minimum and maximum bounds
        for i, category in enumerate(categories):
            min_bound, max_bound = category_bounds[category]
            
            # Min bound: -x_i ≤ -min_bound
            min_row = np.zeros(n_categories)
            min_row[i] = -1
            A_ub.append(min_row)
            b_ub.append(-min_bound)
            
            # Max bound: x_i ≤ max_bound
            max_row = np.zeros(n_categories)
            max_row[i] = 1
            A_ub.append(max_row)
            b_ub.append(max_bound)
        
        # 2. Total budget constraint
        A_eq = [np.ones(n_categories)]
        b_eq = [total_budget - savings_goal - investment_goal - debt_payment_goal]
        
        # 3. Essential categories minimum allocation
        essential_row = np.zeros(n_categories)
        for i, category in enumerate(categories):
            if category in self.essential_categories:
                essential_row[i] = -1
        A_ub.append(essential_row)
        b_ub.append(-total_budget * len(self.essential_categories) * self.essential_min_percent)
        
        # Convert to numpy arrays
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Solve optimization problem
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            method='highs'
        )
        
        if not result.success:
            return None
            
        # Create optimized budget allocation
        optimized_budget = {}
        for i, category in enumerate(categories):
            optimized_budget[category] = round(result.x[i], 2)
            
        # Add financial goals
        optimized_budget['Savings'] = savings_goal
        if investment_goal > 0:
            optimized_budget['Investments'] = investment_goal
        if debt_payment_goal > 0:
            optimized_budget['Debt Payment'] = debt_payment_goal
            
        return optimized_budget
        
    def get_goal_progress(self, user_id: int, month: int = None, year: int = None) -> Dict[str, float]:
        """Calculate progress towards financial goals for the specified month."""
        if month is None:
            month = datetime.now().month
        if year is None:
            year = datetime.now().year
            
        # Get transactions for the month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
            
        transactions = Transaction.query.filter(
            Transaction.user_id == user_id,
            Transaction.date >= start_date,
            Transaction.date < end_date
        ).all()
        
        # Calculate actual spending by category
        actual_spending = {}
        for t in transactions:
            actual_spending[t.category] = actual_spending.get(t.category, 0) + t.amount
            
        # Get budget settings
        budget_settings = BudgetSetting.query.filter_by(user_id=user_id).first()
        if not budget_settings:
            return None
            
        # Calculate progress
        progress = {
            'total_budget': budget_settings.total_budget,
            'total_spent': sum(actual_spending.values()),
            'savings_goal': budget_settings.savings_goal,
            'actual_savings': budget_settings.total_budget - sum(actual_spending.values()),
            'categories': {
                category: {
                    'budget': budget_settings.allocations.get(category, 0),
                    'spent': actual_spending.get(category, 0)
                } for category in set(list(budget_settings.allocations.keys()) + list(actual_spending.keys()))
            }
        }
        
        # Calculate percentages
        progress['savings_progress'] = (
            (progress['actual_savings'] / progress['savings_goal']) * 100
            if progress['savings_goal'] > 0 else 0
        )
        
        for category in progress['categories']:
            budget = progress['categories'][category]['budget']
            spent = progress['categories'][category]['spent']
            progress['categories'][category]['percentage'] = (
                (spent / budget) * 100 if budget > 0 else 0
            )
            
        return progress
