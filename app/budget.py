def allocate_budget(total_income):
    """
    Allocate the total income into different budget categories based on predefined percentages.
    
    Args:
        total_income (float): Total monthly income
        
    Returns:
        dict: Dictionary containing budget allocations for each category
    """
    allocations = {
        "Basic Needs": 0.50,
        "Wants": 0.20,
        "Savings": 0.10,
        "Investments": 0.10,
        "Dreams": 0.05,
        "Education": 0.03,
        "Donations": 0.02
    }

    budget = {category: amount * total_income for category, amount in allocations.items()}
    return budget

def get_budget_summary(budget):
    """
    Create a formatted summary of the budget allocations.
    
    Args:
        budget (dict): Dictionary containing budget allocations
        
    Returns:
        list: List of dictionaries containing category and formatted amount
    """
    return [
        {
            "category": category,
            "amount": f"${amount:.2f}",
            "amount_raw": amount
        }
        for category, amount in budget.items()
    ]
