# Finance Manager

A comprehensive Flask-based financial management application with AI-powered features for transaction categorization, budget optimization, and financial insights.

## Features
- 🤖 AI-powered transaction categorization
- 💰 German tax calculation and management
- 📊 Financial insights and analytics using LLM
- 📈 Budget tracking and optimization
- 🔔 Bill detection and reminders
- 📱 RESTful API for integration
- 🔒 Secure user authentication
- 📝 Transaction history tracking
- 🗄️ SQLite database storage

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd finance-manager
   ```
2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit .env with your settings
   # Make sure to change the SECRET_KEY
   ```
5. Initialize the database:
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

## Running the Application
```bash
python run.py
```
The application will be available at `http://localhost:5000`

## API Endpoints

### Authentication
- **POST** `/auth/register`: Register a new user
- **POST** `/auth/login`: Login user
- **POST** `/auth/logout`: Logout user
- **GET** `/auth/profile`: Get user profile

### Transactions
- **POST** `/classify_transaction`: Classify and save a new transaction
- **GET** `/transactions`: List all transactions
- **GET** `/analytics/spending_by_category`: Get spending analytics
- **GET** `/analytics/monthly_trends`: Get monthly spending trends

### Budget Management
- **POST** `/budget/set`: Set budget preferences
- **GET** `/budget/status`: Get current budget status
- **GET** `/budget-insights`: Get AI-powered budget insights
- **POST** `/optimize-budget`: Get budget optimization recommendations

### Bills and Reminders
- **POST** `/process-bill`: Process and analyze bills
- **GET** `/upcoming-bills`: Get list of upcoming bills
- **GET** `/reminders`: Get payment reminders

### File Management
- **POST** `/upload`: Upload supporting documents

## Security Features
- Secure user authentication with Flask-Login
- Environment variable configuration
- CORS protection
- File upload validation
- XSS protection headers
- SQL injection prevention

## Development
To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Testing
```bash
# Run tests
python -m pytest tests/
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Support
If you encounter any problems or have suggestions, please open an issue in the GitHub repository.

