from flask import Blueprint, request, jsonify, render_template, send_file
from datetime import datetime, timedelta
import pandas as pd
import io
import csv
from .models import Transaction, BudgetSetting, Alert, Reminder
from . import db
from .config import CATEGORIES, GERMANY_TAX_RATES, BUDGET_CATEGORIES, SOCIAL_SECURITY
from .budget import allocate_budget, get_budget_summary
from .ml_budget import get_budget_insights
from .forecasting import get_expense_forecast
from .recommendations import get_financial_recommendations
from .budget_adjustments import get_budget_adjustments
from .insights_generator import InsightsGenerator
from .anomaly_detector import AnomalyDetectionSystem
from .bill_detector import BillDetector
from .goal_optimizer import GoalOptimizer
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user
import os

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

main = Blueprint('main', __name__)

def simple_classify(description):
    # Simple keyword-based classification
    description = description.lower()
    if any(word in description for word in ['grocery', 'groceries', 'food', 'supermarket', 'lidl', 'aldi', 'edeka', 'rewe']):
        return 'Essential'
    elif any(word in description for word in ['restaurant', 'dinner', 'lunch', 'cinema', 'entertainment', 'cafe', 'bar']):
        return 'Luxury'
    elif any(word in description for word in ['salary', 'wage', 'income', 'payment']):
        return 'Salary'
    elif any(word in description for word in ['investment', 'stock', 'bond', 'dividend']):
        return 'Investments'
    else:
        return 'Services'

def calculate_german_tax(category, amount):
    if category == "Salary" or category == "Investments":
        return amount * GERMANY_TAX_RATES["income"]
    elif category == "Luxury" or category == "Services":
        return amount * GERMANY_TAX_RATES["VAT_standard"]
    else:
        return amount * GERMANY_TAX_RATES["VAT_reduced"]

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename
            }), 200
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/classify_transaction', methods=['POST'])
@login_required
def classify_transaction():
    try:
        print("Received transaction classification request")
        data = request.json
        print(f"Request data: {data}")
        
        description = data['description']
        amount = float(data['amount'])
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Classifying transaction: {description}")
        category = simple_classify(description)
        print(f"Classification result: {category}")
        
        tax_amount = calculate_german_tax(category, amount)
        tax_rate = (tax_amount / amount) * 100 if amount > 0 else 0.0
        
        new_transaction = Transaction(
            description=description,
            amount=amount,
            date=date,
            category=category,
            tax_rate=tax_rate,
            tax_amount=tax_amount
        )
        db.session.add(new_transaction)
        db.session.commit()
        print("Transaction saved successfully")
        
        return jsonify({
            'message': 'Transaction categorized and saved', 
            'category': category, 
            'tax_amount': tax_amount
        })
    except Exception as e:
        print(f"Error in classify_transaction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@main.route('/analytics/spending_by_category', methods=['GET'])
@login_required
def spending_by_category():
    try:
        # Get date range from query parameters
        days = request.args.get('days', default=30, type=int)
        start_date = datetime.now() - timedelta(days=days)
        
        # Query transactions
        transactions = Transaction.query.filter(
            Transaction.date >= start_date.strftime("%Y-%m-%d")
        ).all()
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([{
            'amount': t.amount,
            'category': t.category,
            'date': datetime.strptime(t.date, "%Y-%m-%d %H:%M:%S")
        } for t in transactions])
        
        if df.empty:
            return jsonify({
                'categories': [],
                'amounts': [],
                'percentages': []
            })
        
        # Calculate spending by category
        category_totals = df[df['category'] != 'Salary'].groupby('category')['amount'].sum()
        total_spending = category_totals.sum()
        percentages = (category_totals / total_spending * 100).round(2)
        
        return jsonify({
            'categories': category_totals.index.tolist(),
            'amounts': category_totals.values.tolist(),
            'percentages': percentages.values.tolist()
        })
    except Exception as e:
        print(f"Error in spending_by_category: {str(e)}")
        return jsonify({'error': str(e)}), 400

@main.route('/analytics/monthly_trends', methods=['GET'])
@login_required
def monthly_trends():
    try:
        # Get last 6 months of data
        start_date = datetime.now() - timedelta(days=180)
        
        # Query transactions
        transactions = Transaction.query.filter(
            Transaction.date >= start_date.strftime("%Y-%m-%d")
        ).all()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'amount': t.amount,
            'category': t.category,
            'date': datetime.strptime(t.date, "%Y-%m-%d %H:%M:%S")
        } for t in transactions])
        
        if df.empty:
            return jsonify({
                'labels': [],
                'income': [],
                'expenses': []
            })
        
        # Add month column
        df['month'] = df['date'].dt.strftime('%Y-%m')
        
        # Calculate monthly income and expenses
        monthly_income = df[df['category'] == 'Salary'].groupby('month')['amount'].sum()
        monthly_expenses = df[df['category'] != 'Salary'].groupby('month')['amount'].sum()
        
        # Get all months in range
        all_months = pd.date_range(
            start=df['date'].min(),
            end=df['date'].max(),
            freq='M'
        ).strftime('%Y-%m').tolist()
        
        # Fill in missing months with 0
        monthly_income = monthly_income.reindex(all_months, fill_value=0)
        monthly_expenses = monthly_expenses.reindex(all_months, fill_value=0)
        
        return jsonify({
            'labels': all_months,
            'income': monthly_income.values.tolist(),
            'expenses': monthly_expenses.values.tolist()
        })
    except Exception as e:
        print(f"Error in monthly_trends: {str(e)}")
        return jsonify({'error': str(e)}), 400

@main.route('/transactions', methods=['GET'])
@login_required
def get_transactions():
    try:
        print("Fetching all transactions")
        
        # Get filter parameters
        category = request.args.get('category', default=None)
        start_date = request.args.get('start_date', default=None)
        end_date = request.args.get('end_date', default=None)
        min_amount = request.args.get('min_amount', default=None, type=float)
        max_amount = request.args.get('max_amount', default=None, type=float)
        
        # Start with base query
        query = Transaction.query
        
        # Apply filters
        if category:
            query = query.filter(Transaction.category == category)
        if start_date:
            query = query.filter(Transaction.date >= start_date)
        if end_date:
            query = query.filter(Transaction.date <= end_date)
        if min_amount is not None:
            query = query.filter(Transaction.amount >= min_amount)
        if max_amount is not None:
            query = query.filter(Transaction.amount <= max_amount)
        
        transactions = query.all()
        return jsonify([t.to_dict() for t in transactions])
    except Exception as e:
        print(f"Error in get_transactions: {str(e)}")
        return jsonify({'error': str(e)}), 400

@main.route('/financial_insights', methods=['POST'])
@login_required
def financial_insights():
    try:
        print("Received financial insights request")
        data = request.json
        question = data['question']
        print(f"Question: {question}")
        
        # Simple response without ollama
        response = f"I understand you're asking about: {question}. However, the AI-powered insights feature requires the Ollama package to be properly installed and configured."
        print("Sending simple response")
        
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in financial_insights: {str(e)}")
        return jsonify({'error': str(e)}), 400

@main.route('/budget/set', methods=['POST'])
@login_required
def set_budget():
    try:
        data = request.json
        category = data['category']
        monthly_limit = float(data['monthly_limit'])
        alert_threshold = float(data['alert_threshold'])

        # Update or create budget setting
        budget = BudgetSetting.query.filter_by(category=category).first()
        if budget:
            budget.monthly_limit = monthly_limit
            budget.alert_threshold = alert_threshold
        else:
            budget = BudgetSetting(
                category=category,
                monthly_limit=monthly_limit,
                alert_threshold=alert_threshold
            )
            db.session.add(budget)
        
        db.session.commit()
        return jsonify({'message': f'Budget set for {category}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@main.route('/budget/status', methods=['GET'])
@login_required
def get_budget_status():
    try:
        # Get current month's start date
        today = datetime.now()
        month_start = today.replace(day=1).strftime("%Y-%m-%d")
        
        # Get all budget settings
        budgets = BudgetSetting.query.all()
        status = []
        
        for budget in budgets:
            # Get total spending for the category this month
            transactions = Transaction.query.filter(
                Transaction.category == budget.category,
                Transaction.date >= month_start,
                Transaction.category != 'Salary'
            ).all()
            
            total_spent = sum(t.amount for t in transactions)
            percentage_used = (total_spent / budget.monthly_limit * 100) if budget.monthly_limit > 0 else 0
            
            status.append({
                'category': budget.category,
                'monthly_limit': budget.monthly_limit,
                'spent': total_spent,
                'remaining': budget.monthly_limit - total_spent,
                'percentage_used': percentage_used,
                'alert_threshold': budget.alert_threshold,
                'alert': percentage_used >= budget.alert_threshold
            })
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@main.route('/export/transactions', methods=['GET'])
@login_required
def export_transactions():
    try:
        format = request.args.get('format', 'csv')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Build query
        query = Transaction.query
        if start_date:
            query = query.filter(Transaction.date >= start_date)
        if end_date:
            query = query.filter(Transaction.date <= end_date)
            
        transactions = query.all()
        
        if format == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Date', 'Description', 'Amount', 'Category', 'Tax Rate', 'Tax Amount'])
            
            # Write data
            for t in transactions:
                writer.writerow([
                    t.date,
                    t.description,
                    t.amount,
                    t.category,
                    t.tax_rate,
                    t.tax_amount
                ])
            
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'transactions_{datetime.now().strftime("%Y%m%d")}.csv'
            )
            
        elif format == 'excel':
            df = pd.DataFrame([t.to_dict() for t in transactions])
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Transactions', index=False)
                
                # Get workbook and add formats
                workbook = writer.book
                worksheet = writer.sheets['Transactions']
                
                # Add currency format
                money_fmt = workbook.add_format({'num_format': '€#,##0.00'})
                percent_fmt = workbook.add_format({'num_format': '0.00%'})
                
                # Apply formats to columns
                worksheet.set_column('C:C', 12, money_fmt)  # Amount
                worksheet.set_column('E:E', 10, percent_fmt)  # Tax Rate
                worksheet.set_column('F:F', 12, money_fmt)  # Tax Amount
                
                # Adjust column widths
                for idx, col in enumerate(df.columns):
                    worksheet.set_column(idx, idx, max(len(col) + 2, 12))
            
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'transactions_{datetime.now().strftime("%Y%m%d")}.xlsx'
            )
        
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@main.route('/calculate_income', methods=['POST'])
@login_required
def calculate_income():
    try:
        data = request.json
        gross_income = float(data['gross_income'])
        
        # Calculate social security contributions
        social_security_total = sum(SOCIAL_SECURITY.values())
        social_security_amount = (gross_income * social_security_total) / 100
        
        # Calculate income tax
        taxable_income = gross_income - social_security_amount
        income_tax = taxable_income * GERMANY_TAX_RATES["income"]
        
        # Calculate net income
        net_income = gross_income - social_security_amount - income_tax
        
        # Calculate budget allocations
        budget_allocations = {}
        for category, details in BUDGET_CATEGORIES.items():
            amount = (net_income * details["percentage"]) / 100
            budget_allocations[category] = {
                "amount": round(amount, 2),
                "percentage": details["percentage"],
                "description": details["description"],
                "subcategories": {
                    subcat: round(amount / len(details["subcategories"]), 2)
                    for subcat in details["subcategories"]
                }
            }
        
        return jsonify({
            "gross_income": round(gross_income, 2),
            "deductions": {
                "social_security": {
                    "total": round(social_security_amount, 2),
                    "details": {
                        name: round((gross_income * rate) / 100, 2)
                        for name, rate in SOCIAL_SECURITY.items()
                    }
                },
                "income_tax": round(income_tax, 2)
            },
            "net_income": round(net_income, 2),
            "budget_allocations": budget_allocations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@main.route('/allocate-budget', methods=['POST'])
@login_required
def budget_allocation():
    data = request.get_json()
    total_income = float(data.get('total_income', 0))
    
    if total_income <= 0:
        return jsonify({
            'error': 'Invalid income amount. Please provide a positive number.'
        }), 400
    
    budget = allocate_budget(total_income)
    summary = get_budget_summary(budget)
    
    return jsonify({
        'total_income': total_income,
        'allocations': summary
    })

@main.route('/budget-insights', methods=['GET'])
@login_required
def budget_insights():
    """Get AI-powered budget optimization insights."""
    try:
        # Get the analysis period from query parameters (default to 30 days)
        days = int(request.args.get('days', 30))
        
        # Get insights from the ML models
        insights = get_budget_insights(days)
        
        return jsonify({
            'status': 'success',
            'data': insights,
            'message': 'Budget insights generated successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating budget insights: {str(e)}'
        }), 400

@main.route('/expense-forecast', methods=['GET'])
@login_required
def expense_forecast():
    """Get expense forecasts using ML models."""
    try:
        # Get the forecast period from query parameters (default to 30 days)
        days = int(request.args.get('days', 30))
        
        # Get forecasts from the ML models
        forecasts = get_expense_forecast(days)
        
        if forecasts is None:
            return jsonify({
                'status': 'error',
                'message': 'Insufficient data for forecasting'
            }), 400
        
        return jsonify({
            'status': 'success',
            'data': forecasts,
            'message': 'Expense forecasts generated successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating expense forecasts: {str(e)}'
        }), 400

@main.route('/financial-recommendations', methods=['GET'])
@login_required
def financial_recommendations():
    """Get personalized financial recommendations."""
    try:
        # Get monthly income from query parameters if provided
        monthly_income = request.args.get('monthly_income', type=float)
        
        # Get recommendations from the ML models
        recommendations = get_financial_recommendations(monthly_income)
        
        if recommendations is None:
            return jsonify({
                'status': 'error',
                'message': 'Insufficient data for generating recommendations'
            }), 400
        
        return jsonify({
            'status': 'success',
            'data': recommendations,
            'message': 'Financial recommendations generated successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating recommendations: {str(e)}'
        }), 400

@main.route('/budget-adjustments', methods=['GET'])
@login_required
def budget_adjustments():
    """Get automated budget adjustment recommendations."""
    try:
        # Get the analysis period from query parameters (default to 30 days)
        days = int(request.args.get('days', 30))
        
        # Get adjustments from the ML models
        adjustments = get_budget_adjustments(days)
        
        if adjustments is None:
            return jsonify({
                'status': 'error',
                'message': 'Insufficient data for budget adjustments'
            }), 400
        
        return jsonify({
            'status': 'success',
            'data': adjustments,
            'message': 'Budget adjustments generated successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating budget adjustments: {str(e)}'
        }), 400

@main.route('/personalized-insights', methods=['GET'])
@login_required
def personalized_insights():
    """Get personalized financial insights."""
    try:
        # Initialize insights generator
        generator = InsightsGenerator()
        
        # Generate insights for the current user
        insights = generator.generate_monthly_insights(1)  # TODO: Replace with actual user_id
        
        if insights is None:
            return jsonify({
                'status': 'error',
                'message': 'Insufficient data for generating insights'
            }), 400
        
        return jsonify({
            'status': 'success',
            'data': insights,
            'message': 'Financial insights generated successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating insights: {str(e)}'
        }), 400

@main.route('/anomaly-detection', methods=['GET'])
@login_required
def detect_anomalies():
    """Run anomaly detection and get alerts."""
    try:
        # Get the analysis period from query parameters (default to 30 days)
        days = int(request.args.get('days', 30))
        
        # Initialize anomaly detector
        detector = AnomalyDetectionSystem()
        
        # Run detection for current user
        results = detector.detect_anomalies(1, days)  # TODO: Replace with actual user_id
        
        if results is None:
            return jsonify({
                'status': 'error',
                'message': 'Insufficient data for anomaly detection'
            }), 400
        
        return jsonify({
            'status': 'success',
            'data': results,
            'message': 'Anomaly detection completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error during anomaly detection: {str(e)}'
        }), 400

@main.route('/alerts', methods=['GET'])
@login_required
def get_alerts():
    """Get user's alerts."""
    try:
        # Get unread parameter (optional)
        unread_only = request.args.get('unread', 'false').lower() == 'true'
        
        # Query alerts for current user
        query = Alert.query.filter_by(user_id=1)  # TODO: Replace with actual user_id
        if unread_only:
            query = query.filter_by(read=False)
            
        alerts = query.order_by(Alert.created_at.desc()).all()
        
        return jsonify({
            'status': 'success',
            'data': {
                'alerts': [alert.to_dict() for alert in alerts],
                'total_unread': Alert.query.filter_by(user_id=1, read=False).count()
            },
            'message': 'Alerts retrieved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving alerts: {str(e)}'
        }), 400

@main.route('/alerts/<int:alert_id>/mark-read', methods=['POST'])
@login_required
def mark_alert_read(alert_id):
    """Mark an alert as read."""
    try:
        alert = Alert.query.get_or_404(alert_id)
        
        # Verify alert belongs to current user
        if alert.user_id != 1:  # TODO: Replace with actual user_id
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized access to alert'
            }), 403
            
        alert.read = True
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Alert marked as read'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': f'Error marking alert as read: {str(e)}'
        }), 400

@main.route('/process-bill', methods=['POST'])
@login_required
def process_bill():
    """Process bill text content."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No text content provided'
            }), 400
            
        # Process bill text
        detector = BillDetector()
        result = detector.process_bill(data['text'], 1)  # TODO: Replace with actual user_id
        
        if result:
            return jsonify({
                'status': 'success',
                'data': result,
                'message': 'Bill processed successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to process bill'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing bill: {str(e)}'
        }), 400

@main.route('/upcoming-bills', methods=['GET'])
@login_required
def get_upcoming_bills():
    """Get upcoming bills for the user."""
    try:
        days = int(request.args.get('days', 30))
        detector = BillDetector()
        bills = detector.get_upcoming_bills(1, days)  # TODO: Replace with actual user_id
        
        return jsonify({
            'status': 'success',
            'data': bills,
            'message': 'Upcoming bills retrieved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving upcoming bills: {str(e)}'
        }), 400

@main.route('/reminders', methods=['GET'])
@login_required
def get_reminders():
    """Get pending reminders for the user."""
    try:
        detector = BillDetector()
        reminders = detector.get_pending_reminders(1)  # TODO: Replace with actual user_id
        
        return jsonify({
            'status': 'success',
            'data': reminders,
            'message': 'Reminders retrieved successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving reminders: {str(e)}'
        }), 400

@main.route('/reminders/<int:reminder_id>/dismiss', methods=['POST'])
@login_required
def dismiss_reminder(reminder_id):
    """Dismiss a reminder."""
    try:
        reminder = Reminder.query.get_or_404(reminder_id)
        
        # Verify reminder belongs to current user
        if reminder.user_id != 1:  # TODO: Replace with actual user_id
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized access to reminder'
            }), 403
            
        reminder.status = 'dismissed'
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Reminder dismissed successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': f'Error dismissing reminder: {str(e)}'
        }), 400

@main.route('/update-goals', methods=['POST'])
@login_required
def update_goals():
    """Update financial goals and optimize budget."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
            
        # Get budget settings
        budget_settings = BudgetSetting.query.filter_by(user_id=1).first()  # TODO: Replace with actual user_id
        if not budget_settings:
            return jsonify({
                'status': 'error',
                'message': 'Budget settings not found'
            }), 404
            
        # Update goals
        budget_settings.savings_goal = data.get('savings_goal', budget_settings.savings_goal)
        budget_settings.investment_goal = data.get('investment_goal', budget_settings.investment_goal)
        budget_settings.debt_payment_goal = data.get('debt_payment_goal', budget_settings.debt_payment_goal)
        
        # Optimize budget based on new goals
        optimizer = GoalOptimizer()
        optimized_budget = optimizer.optimize_budget(
            user_id=1,  # TODO: Replace with actual user_id
            total_budget=budget_settings.total_budget,
            savings_goal=budget_settings.savings_goal,
            investment_goal=budget_settings.investment_goal,
            debt_payment_goal=budget_settings.debt_payment_goal
        )
        
        if optimized_budget:
            budget_settings.allocations = optimized_budget
            db.session.commit()
            
            return jsonify({
                'status': 'success',
                'data': {
                    'goals': {
                        'savings_goal': budget_settings.savings_goal,
                        'investment_goal': budget_settings.investment_goal,
                        'debt_payment_goal': budget_settings.debt_payment_goal
                    },
                    'allocations': optimized_budget
                },
                'message': 'Goals updated and budget optimized successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to optimize budget with new goals'
            }), 400
            
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': f'Error updating goals: {str(e)}'
        }), 400

@main.route('/goal-progress', methods=['GET'])
@login_required
def get_goal_progress():
    """Get progress towards financial goals."""
    try:
        month = request.args.get('month', type=int)
        year = request.args.get('year', type=int)
        
        optimizer = GoalOptimizer()
        progress = optimizer.get_goal_progress(1, month, year)  # TODO: Replace with actual user_id
        
        if progress:
            return jsonify({
                'status': 'success',
                'data': progress,
                'message': 'Goal progress retrieved successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to retrieve goal progress'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving goal progress: {str(e)}'
        }), 400

@main.route('/optimize-budget', methods=['POST'])
@login_required
def optimize_budget():
    """Optimize budget allocation based on current goals."""
    try:
        budget_settings = BudgetSetting.query.filter_by(user_id=1).first()  # TODO: Replace with actual user_id
        if not budget_settings:
            return jsonify({
                'status': 'error',
                'message': 'Budget settings not found'
            }), 404
            
        optimizer = GoalOptimizer()
        optimized_budget = optimizer.optimize_budget(
            user_id=1,  # TODO: Replace with actual user_id
            total_budget=budget_settings.total_budget,
            savings_goal=budget_settings.savings_goal,
            investment_goal=budget_settings.investment_goal,
            debt_payment_goal=budget_settings.debt_payment_goal
        )
        
        if optimized_budget:
            budget_settings.allocations = optimized_budget
            db.session.commit()
            
            return jsonify({
                'status': 'success',
                'data': {
                    'allocations': optimized_budget
                },
                'message': 'Budget optimized successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to optimize budget'
            }), 400
            
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': f'Error optimizing budget: {str(e)}'
        }), 400

@main.route('/api/budget-summary', methods=['GET'])
@login_required
def get_budget_summary():
    try:
        # Get current month's transactions
        start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        transactions = Transaction.query.filter(Transaction.date >= start_date.strftime("%Y-%m-%d")).all()
        
        # Calculate totals
        income = sum(t.amount for t in transactions if t.category == 'Salary')
        expenses = abs(sum(t.amount for t in transactions if t.category != 'Salary'))
        net_savings = income - expenses
        
        # Get last month's data for comparison
        last_month_start = (start_date - timedelta(days=start_date.day)).replace(day=1)
        last_month_transactions = Transaction.query.filter(
            Transaction.date >= last_month_start.strftime("%Y-%m-%d"),
            Transaction.date < start_date.strftime("%Y-%m-%d")
        ).all()
        
        last_month_income = sum(t.amount for t in last_month_transactions if t.category == 'Salary')
        last_month_expenses = abs(sum(t.amount for t in last_month_transactions if t.category != 'Salary'))
        
        # Calculate changes
        income_change = ((income - last_month_income) / last_month_income * 100) if last_month_income else 0
        expense_change = ((expenses - last_month_expenses) / last_month_expenses * 100) if last_month_expenses else 0
        
        # Calculate budget progress
        budget_settings = BudgetSetting.query.first()
        total_budget = budget_settings.total_budget if budget_settings else 5000
        budget_progress = (expenses / total_budget * 100) if total_budget else 0
        
        # Calculate savings rate
        savings_rate = (net_savings / income * 100) if income > 0 else 0
        
        # Calculate days left in current month
        today = datetime.now()
        next_month = today.replace(day=28) + timedelta(days=4)
        last_day = next_month - timedelta(days=next_month.day)
        days_left = (last_day - today).days + 1
        
        return jsonify({
            "totalIncome": income,
            "totalExpenses": expenses,
            "netSavings": net_savings,
            "budgetProgress": round(budget_progress, 1),
            "incomeChange": round(income_change, 1),
            "expenseChange": round(expense_change, 1),
            "savingsRate": round(savings_rate, 1),
            "daysLeft": days_left
        })
    except Exception as e:
        print(f"Error in get_budget_summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@main.route('/api/transactions', methods=['GET', 'POST'])
@login_required
def transactions_api():
    if request.method == 'POST':
        try:
            data = request.json
            description = data.get('description')
            amount = float(data.get('amount'))
            category = data.get('category') or simple_classify(description)
            date = data.get('date') or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            tax_amount = calculate_german_tax(category, amount)
            tax_rate = (tax_amount / amount) * 100 if amount > 0 else 0.0
            
            transaction = Transaction(
                description=description,
                amount=amount,
                date=date,
                category=category,
                tax_rate=tax_rate,
                tax_amount=tax_amount
            )
            
            db.session.add(transaction)
            db.session.commit()
            
            return jsonify(transaction.to_dict()), 201
        except Exception as e:
            print(f"Error adding transaction: {str(e)}")
            return jsonify({'error': str(e)}), 400
    else:
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 10, type=int)
            category = request.args.get('category')
            date_range = request.args.get('dateRange', 30, type=int)
            min_amount = request.args.get('minAmount', type=float)
            max_amount = request.args.get('maxAmount', type=float)
            sort_by = request.args.get('sortBy', 'date_desc')
            
            query = Transaction.query
            
            if category:
                query = query.filter(Transaction.category == category)
            
            if date_range:
                start_date = datetime.now() - timedelta(days=date_range)
                query = query.filter(Transaction.date >= start_date.strftime("%Y-%m-%d"))
            
            if min_amount is not None:
                query = query.filter(Transaction.amount >= min_amount)
            
            if max_amount is not None:
                query = query.filter(Transaction.amount <= max_amount)
            
            if sort_by == 'date_desc':
                query = query.order_by(Transaction.date.desc())
            elif sort_by == 'date_asc':
                query = query.order_by(Transaction.date.asc())
            elif sort_by == 'amount_desc':
                query = query.order_by(Transaction.amount.desc())
            elif sort_by == 'amount_asc':
                query = query.order_by(Transaction.amount.asc())
            
            paginated = query.paginate(page=page, per_page=per_page, error_out=False)
            
            return jsonify({
                'transactions': [t.to_dict() for t in paginated.items],
                'total': paginated.total,
                'pages': paginated.pages,
                'current_page': paginated.page
            })
        except Exception as e:
            print(f"Error fetching transactions: {str(e)}")
            return jsonify({'error': str(e)}), 400

@main.route('/api/analytics', methods=['GET'])
@login_required
def analytics_api():
    try:
        # Get date range from query parameters
        days = request.args.get('days', 180, type=int)
        start_date = datetime.now() - timedelta(days=days)
        
        # Get transactions within date range
        transactions = Transaction.query.filter(
            Transaction.date >= start_date.strftime("%Y-%m-%d")
        ).all()
        
        # Calculate monthly totals
        monthly_data = {}
        category_totals = {}
        
        for transaction in transactions:
            month = transaction.date[:7]  # Get YYYY-MM
            amount = transaction.amount
            category = transaction.category
            
            if month not in monthly_data:
                monthly_data[month] = {'income': 0, 'expenses': 0}
            
            if category not in category_totals:
                category_totals[category] = 0
                
            if category == 'Salary':
                monthly_data[month]['income'] += amount
            else:
                monthly_data[month]['expenses'] += abs(amount)
                category_totals[category] += abs(amount)
        
        # Sort months
        sorted_months = sorted(monthly_data.keys())
        
        return jsonify({
            'months': sorted_months,
            'income': [monthly_data[m]['income'] for m in sorted_months],
            'expenses': [monthly_data[m]['expenses'] for m in sorted_months],
            'categories': list(category_totals.keys()),
            'categoryAmounts': list(category_totals.values())
        })
    except Exception as e:
        print(f"Error in analytics: {str(e)}")
        return jsonify({'error': str(e)}), 500
