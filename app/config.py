CATEGORIES = ['Essential', 'Luxury', 'Salary', 'Investments', 'Services']

# German tax rates (simplified for demonstration)
GERMANY_TAX_RATES = {
    "income": 0.42,  # 42% income tax rate
    "VAT_standard": 0.19,  # 19% standard VAT
    "VAT_reduced": 0.07,  # 7% reduced VAT
}

# Budget category allocations (percentage of after-tax income)
BUDGET_CATEGORIES = {
    "Basic Needs": {
        "percentage": 50,
        "description": "Essential expenses like food, housing, utilities",
        "subcategories": ["Food", "Housing", "Utilities", "Transportation"]
    },
    "Wants": {
        "percentage": 20,
        "description": "Non-essential spending and entertainment",
        "subcategories": ["Entertainment", "Shopping", "Dining Out", "Hobbies"]
    },
    "Savings": {
        "percentage": 10,
        "description": "Emergency fund and general savings",
        "subcategories": ["Emergency Fund", "General Savings"]
    },
    "Investments": {
        "percentage": 10,
        "description": "Long-term investments and growth",
        "subcategories": ["Stocks", "Bonds", "Real Estate"]
    },
    "Dreams": {
        "percentage": 5,
        "description": "Future aspirations and goals",
        "subcategories": ["Travel", "Major Purchases"]
    },
    "Education": {
        "percentage": 3,
        "description": "Learning and skill development",
        "subcategories": ["Courses", "Books", "Training"]
    },
    "Donations": {
        "percentage": 2,
        "description": "Charitable giving and support",
        "subcategories": ["Charity", "Community Support"]
    }
}

# Social security contributions (percentage of gross income)
SOCIAL_SECURITY = {
    "health_insurance": 7.3,  # Public health insurance
    "pension_insurance": 9.3,  # Pension insurance
    "unemployment_insurance": 1.2,  # Unemployment insurance
    "nursing_care_insurance": 1.525  # Nursing care insurance
}
