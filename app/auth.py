from flask import Blueprint, request, jsonify, url_for
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from .models import User

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')

        if not email or not password or not name:
            return jsonify({'error': 'Missing required fields'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400

        user = User(
            email=email,
            name=name,
            password=generate_password_hash(password, method='sha256')
        )

        db.session.add(user)
        db.session.commit()

        return jsonify({'message': 'Registration successful'}), 201
    except Exception as e:
        return jsonify({'error': 'Registration failed'}), 500

@auth.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Missing required fields'}), 400

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            return jsonify({'error': 'Invalid credentials'}), 401

        login_user(user)
        return jsonify({'message': 'Login successful'}), 200
    except Exception as e:
        return jsonify({'error': 'Login failed'}), 500

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logout successful'}), 200

@auth.route('/profile')
@login_required
def profile():
    return jsonify({
        'id': current_user.id,
        'email': current_user.email,
        'name': current_user.name
    })
