from app import create_app, db

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")
        print("Server starting on http://localhost:5000")
    app.run(debug=True, host='localhost', port=5000)
