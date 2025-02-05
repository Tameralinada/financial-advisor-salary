import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
from .models import Transaction, Bill, Reminder
from . import db

class BillDetector:
    def __init__(self):
        # Common bill-related keywords
        self.bill_keywords = [
            'invoice', 'bill', 'payment', 'due', 'statement',
            'utility', 'electricity', 'water', 'gas', 'internet',
            'subscription', 'service', 'monthly', 'quarterly', 'annual'
        ]
        
        # Date patterns
        self.date_patterns = [
            r'\d{2}[/-]\d{2}[/-]\d{4}',  # DD/MM/YYYY or DD-MM-YYYY
            r'\d{4}[/-]\d{2}[/-]\d{2}',  # YYYY/MM/DD or YYYY-MM-YYYY
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'  # 15 January 2024
        ]
        
        # Amount patterns
        self.amount_patterns = [
            r'(?:EUR|€)\s*\d+(?:,\d{3})*(?:\.\d{2})?',  # EUR or € followed by amount
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:EUR|€)',  # Amount followed by EUR or €
            r'Total:\s*(?:EUR|€)?\s*\d+(?:,\d{3})*(?:\.\d{2})?'  # Total: amount
        ]
        
    def _extract_dates(self, text):
        """Extract dates from text using regex patterns"""
        dates = []
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    date_str = match.group()
                    # Convert to datetime object (handle different formats)
                    for fmt in ['%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y', '%Y-%m-%d', '%d %B %Y']:
                        try:
                            date = datetime.strptime(date_str, fmt)
                            dates.append(date)
                            break
                        except ValueError:
                            continue
                except Exception:
                    continue
        return dates
        
    def _extract_amounts(self, text):
        """Extract monetary amounts from text"""
        amounts = []
        for pattern in self.amount_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                amount_str = match.group()
                # Clean amount string and convert to float
                amount_str = re.sub(r'[^\d.,]', '', amount_str)
                try:
                    amount = float(amount_str.replace(',', ''))
                    amounts.append(amount)
                except ValueError:
                    continue
        return amounts
        
    def _detect_bill_type(self, text):
        """Detect bill type based on keywords"""
        text_lower = text.lower()
        
        # Check for utility bills
        if any(keyword in text_lower for keyword in ['electricity', 'power', 'energy']):
            return 'Electricity'
        elif any(keyword in text_lower for keyword in ['water', 'hydro']):
            return 'Water'
        elif any(keyword in text_lower for keyword in ['gas', 'natural gas']):
            return 'Gas'
        elif any(keyword in text_lower for keyword in ['internet', 'broadband', 'wifi']):
            return 'Internet'
        elif any(keyword in text_lower for keyword in ['phone', 'mobile', 'cellular']):
            return 'Phone'
        elif any(keyword in text_lower for keyword in ['rent', 'lease']):
            return 'Rent'
        elif any(keyword in text_lower for keyword in ['insurance', 'coverage']):
            return 'Insurance'
            
        return 'Other'
        
    def _detect_frequency(self, text):
        """Detect bill frequency based on keywords"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['monthly', 'per month', 'every month']):
            return 'monthly'
        elif any(word in text_lower for word in ['quarterly', 'every quarter']):
            return 'quarterly'
        elif any(word in text_lower for word in ['annual', 'yearly', 'per year']):
            return 'annual'
        elif any(word in text_lower for word in ['weekly', 'per week']):
            return 'weekly'
        
        return 'monthly'  # default to monthly if no frequency detected
        
    def process_bill(self, text_content, user_id):
        """Process bill text content and extract relevant information"""
        try:
            # Extract dates
            dates = self._extract_dates(text_content)
            due_date = max(dates) if dates else None
            
            # Extract amounts
            amounts = self._extract_amounts(text_content)
            amount = max(amounts) if amounts else None
            
            # Detect bill type
            bill_type = self._detect_bill_type(text_content)
            
            # Detect frequency
            frequency = self._detect_frequency(text_content)
            
            # Create bill record
            bill = Bill(
                user_id=user_id,
                type=bill_type,
                amount=amount,
                due_date=due_date,
                frequency=frequency,
                provider=None,  # Simplified version doesn't detect provider
                ocr_text=text_content,
                created_at=datetime.now()
            )
            
            # Create reminder
            if due_date:
                reminder = Reminder(
                    user_id=user_id,
                    bill_id=bill.id,
                    due_date=due_date,
                    amount=amount,
                    message=f"{bill_type} bill due on {due_date.strftime('%Y-%m-%d')}",
                    status='pending',
                    reminder_dates=[
                        due_date - timedelta(days=7),
                        due_date - timedelta(days=3),
                        due_date - timedelta(days=1)
                    ]
                )
            
            try:
                db.session.add(bill)
                if due_date:
                    db.session.add(reminder)
                db.session.commit()
                
                return {
                    'bill_id': bill.id,
                    'type': bill_type,
                    'amount': amount,
                    'due_date': due_date.strftime('%Y-%m-%d') if due_date else None,
                    'frequency': frequency,
                    'reminder_dates': [d.strftime('%Y-%m-%d') for d in reminder.reminder_dates] if due_date else None
                }
                
            except Exception as e:
                db.session.rollback()
                print(f"Error saving bill: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Error processing bill: {str(e)}")
            return None
            
    def get_upcoming_bills(self, user_id, days=30):
        """Get upcoming bills for the user"""
        end_date = datetime.now() + timedelta(days=days)
        
        upcoming = Bill.query.filter(
            Bill.user_id == user_id,
            Bill.due_date <= end_date,
            Bill.due_date >= datetime.now()
        ).order_by(Bill.due_date).all()
        
        return [{
            'id': bill.id,
            'type': bill.type,
            'amount': bill.amount,
            'due_date': bill.due_date.strftime('%Y-%m-%d'),
            'provider': bill.provider,
            'days_until_due': (bill.due_date - datetime.now()).days
        } for bill in upcoming]
        
    def get_pending_reminders(self, user_id):
        """Get pending reminders for the user"""
        reminders = Reminder.query.filter(
            Reminder.user_id == user_id,
            Reminder.status == 'pending',
            Reminder.due_date >= datetime.now()
        ).order_by(Reminder.due_date).all()
        
        return [{
            'id': reminder.id,
            'bill_id': reminder.bill_id,
            'due_date': reminder.due_date.strftime('%Y-%m-%d'),
            'amount': reminder.amount,
            'message': reminder.message,
            'reminder_dates': [d.strftime('%Y-%m-%d') for d in reminder.reminder_dates]
        } for reminder in reminders]
