import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .models import Transaction, Alert
from . import db

class AnomalyDetectionSystem:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    def _prepare_features(self, transactions):
        """Prepare features for anomaly detection"""
        df = pd.DataFrame([{
            'amount': t.amount,
            'category': t.category,
            'date': t.date,
            'description': t.description
        } for t in transactions])
        
        if df.empty:
            return None, None
            
        # Daily aggregations
        daily_stats = df.groupby([df['date'].dt.date, 'category']).agg({
            'amount': ['sum', 'count', 'mean']
        }).reset_index()
        daily_stats.columns = ['date', 'category', 'total_amount', 'transaction_count', 'avg_amount']
        
        # Calculate rolling statistics
        features = []
        for category in df['category'].unique():
            cat_data = daily_stats[daily_stats['category'] == category]
            if not cat_data.empty:
                cat_data = cat_data.sort_values('date')
                cat_data['rolling_mean'] = cat_data['total_amount'].rolling(window=7, min_periods=1).mean()
                cat_data['rolling_std'] = cat_data['total_amount'].rolling(window=7, min_periods=1).std()
                cat_data['amount_vs_mean'] = cat_data['total_amount'] / cat_data['rolling_mean'].replace(0, 1)
                features.append(cat_data)
        
        if not features:
            return None, None
            
        features_df = pd.concat(features, ignore_index=True)
        feature_matrix = features_df[[
            'total_amount', 'transaction_count', 'avg_amount',
            'rolling_mean', 'rolling_std', 'amount_vs_mean'
        ]].fillna(0)
        
        return features_df, feature_matrix
        
    def _generate_alert(self, anomaly_data, alert_type, severity):
        """Generate alert message based on anomaly data"""
        if alert_type == 'category_spike':
            return {
                'type': 'category_spike',
                'severity': severity,
                'category': anomaly_data['category'],
                'message': f"Unusual spending detected in {anomaly_data['category']}: "
                         f"${anomaly_data['total_amount']:.2f} "
                         f"({(anomaly_data['amount_vs_mean'] - 1) * 100:.0f}% above normal)",
                'details': {
                    'amount': float(anomaly_data['total_amount']),
                    'normal_amount': float(anomaly_data['rolling_mean']),
                    'percent_increase': float((anomaly_data['amount_vs_mean'] - 1) * 100)
                }
            }
        elif alert_type == 'frequency_anomaly':
            return {
                'type': 'frequency_anomaly',
                'severity': severity,
                'category': anomaly_data['category'],
                'message': f"Unusual number of transactions in {anomaly_data['category']}: "
                         f"{int(anomaly_data['transaction_count'])} transactions "
                         f"(normally around {int(anomaly_data['rolling_mean'])})",
                'details': {
                    'count': int(anomaly_data['transaction_count']),
                    'normal_count': float(anomaly_data['rolling_mean']),
                    'percent_increase': float((anomaly_data['transaction_count'] / anomaly_data['rolling_mean'] - 1) * 100)
                }
            }
        
    def detect_anomalies(self, user_id, days=30):
        """Detect anomalies in recent transactions"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get recent transactions
        transactions = Transaction.query.filter(
            Transaction.user_id == user_id,
            Transaction.date >= start_date,
            Transaction.date <= end_date
        ).all()
        
        if not transactions:
            return None
            
        # Prepare features
        features_df, feature_matrix = self._prepare_features(transactions)
        if feature_matrix is None:
            return None
            
        # Train and predict with Isolation Forest
        self.isolation_forest.fit(feature_matrix)
        if_predictions = self.isolation_forest.predict(feature_matrix)
        
        # Combine predictions (anomaly if either model detects it)
        combined_anomalies = (if_predictions == -1)
        
        alerts = []
        for idx in np.where(combined_anomalies)[0]:
            row = features_df.iloc[idx]
            
            # Calculate severity based on deviation from normal
            amount_deviation = abs(row['amount_vs_mean'] - 1)
            severity = 'high' if amount_deviation > 0.5 else 'medium' if amount_deviation > 0.3 else 'low'
            
            # Generate category spike alert
            if row['amount_vs_mean'] > 1.2:  # 20% above normal
                alerts.append(self._generate_alert(row, 'category_spike', severity))
            
            # Generate frequency anomaly alert
            if row['transaction_count'] > row['rolling_mean'] * 1.5:  # 50% more transactions than normal
                alerts.append(self._generate_alert(row, 'frequency_anomaly', severity))
        
        # Store alerts in database
        for alert in alerts:
            db_alert = Alert(
                user_id=user_id,
                type=alert['type'],
                severity=alert['severity'],
                message=alert['message'],
                category=alert['category'],
                created_at=datetime.now(),
                details=str(alert['details'])
            )
            db.session.add(db_alert)
        
        try:
            db.session.commit()
        except:
            db.session.rollback()
            raise
        
        return {
            'alerts': alerts,
            'analysis_period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            },
            'total_transactions_analyzed': len(transactions),
            'anomalies_detected': len(alerts)
        }

def detect_anomalies():
    """Detect anomalous transactions using Isolation Forest."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    transactions = Transaction.query.filter(
        Transaction.date.between(start_date, end_date)
    ).all()
    
    if not transactions:
        return {
            'status': 'error',
            'message': 'Not enough transaction data for anomaly detection'
        }
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': t.date,
        'amount': t.amount,
        'category': t.category
    } for t in transactions])
    
    anomalies = []
    
    # Analyze each category separately
    for category in df['category'].unique():
        category_data = df[df['category'] == category]
        if len(category_data) < 5:  # Need minimum data points
            continue
        
        # Prepare features
        features = category_data[['amount']].copy()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Detect anomalies
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(features_scaled)
        
        # Get anomalous transactions
        anomalous_indices = np.where(predictions == -1)[0]
        for idx in anomalous_indices:
            transaction = category_data.iloc[idx]
            anomalies.append({
                'date': transaction['date'].strftime('%Y-%m-%d'),
                'amount': float(transaction['amount']),
                'category': category,
                'confidence': float(abs(features_scaled[idx][0]))  # Higher value = more anomalous
            })
    
    # Sort anomalies by confidence
    anomalies.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'status': 'success',
        'anomalies': anomalies,
        'total_transactions': len(df),
        'anomaly_count': len(anomalies),
        'analysis_period_days': 90
    }
