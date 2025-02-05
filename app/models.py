from . import db
from datetime import datetime
import json
from flask_login import UserMixin

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'created_at': self.created_at.isoformat()
        }

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    tax_rate = db.Column(db.Float, nullable=False)
    tax_amount = db.Column(db.Float, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'description': self.description,
            'amount': self.amount,
            'date': self.date,
            'category': self.category,
            'tax_rate': self.tax_rate,
            'tax_amount': self.tax_amount
        }

class BudgetSetting(db.Model):
    """Model for storing budget settings."""
    id = db.Column(db.Integer, primary_key=True)
    total_budget = db.Column(db.Float, nullable=False)
    allocations = db.Column(db.JSON, nullable=False)  # Category allocations
    savings_goal = db.Column(db.Float, default=0)  # Monthly savings goal
    investment_goal = db.Column(db.Float, default=0)  # Monthly investment goal
    debt_payment_goal = db.Column(db.Float, default=0)  # Monthly debt payment goal
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'total_budget': self.total_budget,
            'allocations': self.allocations,
            'savings_goal': self.savings_goal,
            'investment_goal': self.investment_goal,
            'debt_payment_goal': self.debt_payment_goal,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Alert(db.Model):
    """Model for storing anomaly detection alerts."""
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False)  # category_spike, frequency_anomaly
    severity = db.Column(db.String(20), nullable=False)  # low, medium, high
    message = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    read = db.Column(db.Boolean, nullable=False, default=False)
    details = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'severity': self.severity,
            'message': self.message,
            'category': self.category,
            'created_at': self.created_at.isoformat(),
            'read': self.read,
            'details': self.details
        }

class Bill(db.Model):
    """Model for storing bill information."""
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(50), nullable=False)  # electricity, water, gas, etc.
    amount = db.Column(db.Float)
    due_date = db.Column(db.DateTime)
    frequency = db.Column(db.String(20))  # monthly, quarterly, annual
    provider = db.Column(db.String(100))
    ocr_text = db.Column(db.Text)  # store original OCR text for reference
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'amount': self.amount,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'frequency': self.frequency,
            'provider': self.provider,
            'created_at': self.created_at.isoformat()
        }

class Reminder(db.Model):
    """Model for storing bill reminders."""
    id = db.Column(db.Integer, primary_key=True)
    bill_id = db.Column(db.Integer, db.ForeignKey('bill.id'), nullable=False)
    due_date = db.Column(db.DateTime, nullable=False)
    amount = db.Column(db.Float)
    message = db.Column(db.String(500))
    status = db.Column(db.String(20), default='pending')  # pending, sent, dismissed
    reminder_dates = db.Column(db.JSON)  # list of dates to send reminders
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'bill_id': self.bill_id,
            'due_date': self.due_date.isoformat(),
            'amount': self.amount,
            'message': self.message,
            'status': self.status,
            'reminder_dates': self.reminder_dates,
            'created_at': self.created_at.isoformat()
        }

class BudgetAdjustment(db.Model):
    """Model for storing budget adjustment recommendations."""
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # warning, info, alert
    message = db.Column(db.String(500), nullable=False)
    suggested_limit = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'category': self.category,
            'type': self.type,
            'message': self.message,
            'suggested_limit': self.suggested_limit,
            'date': self.date.isoformat()
        }
