# app.py
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, session
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import sqlite3
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import random
import re

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Change this in production

# Database setup
def init_db():
    conn = sqlite3.connect('agriculture.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE,
                 password TEXT,
                 role TEXT,
                 name TEXT,
                 email TEXT,
                 phone TEXT,
                 address TEXT,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS crops
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT,
                 description TEXT,
                 price REAL,
                 farmer_id INTEGER,
                 quantity REAL,
                 available BOOLEAN,
                 image_url TEXT,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY(farmer_id) REFERENCES users(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS orders
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 customer_id INTEGER,
                 crop_id INTEGER,
                 quantity REAL,
                 total_price REAL,
                 status TEXT,
                 order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY(customer_id) REFERENCES users(id),
                 FOREIGN KEY(crop_id) REFERENCES crops(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS weather_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 location TEXT,
                 date DATE,
                 temperature REAL,
                 humidity REAL,
                 rainfall REAL,
                 recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

init_db()

# Load ML models (in a real app, these would be pre-trained)
def load_models():
    # Create a simple RandomForestClassifier for demonstration
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create some dummy training data
    X = np.random.rand(100, 7)  # 7 features
    y = np.random.choice(['rice', 'wheat', 'maize', 'cotton', 'sugarcane'], 100)  # 5 crop types
    
    # Train the model
    model.fit(X, y)
    
    return {
        'crop_recommendation': model,
        'disease_prediction': None,
        'fertilizer_recommendation': None,
        'rainfall_prediction': None
    }

models = load_models()

# Helper functions
def get_db_connection():
    conn = sqlite3.connect('agriculture.db')
    conn.row_factory = sqlite3.Row
    return conn

# Authentication decorator
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes for HTML pages
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/crop-recommendation')
@login_required
def crop_recommendation_page():
    return render_template('crop_recommendation.html')

@app.route('/rain-prediction')
@login_required
def rain_prediction_page():
    return render_template('rain_prediction.html')

@app.route('/crop-details')
@login_required
def crop_details_page():
    return render_template('crop_details.html')

@app.route('/disease-detection')
@login_required
def disease_detection_page():
    return render_template('disease_detection.html')

@app.route('/cost-estimator')
@login_required
def cost_estimator_page():
    return render_template('cost_estimator.html')

@app.route('/plant-health')
@login_required
def plant_health_page():
    return render_template('plant_health.html')

@app.route('/logout')
def logout():
    # Clear session data
    session.clear()
    return redirect(url_for('login_page'))

# API endpoints
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['username', 'password', 'name', 'email', 'phone', 'address']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, data['email']):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate phone number (basic validation)
        phone_pattern = r'^\+?1?\d{9,15}$'
        if not re.match(phone_pattern, data['phone']):
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        # Check if username already exists
        conn = get_db_connection()
        existing_user = conn.execute('SELECT username FROM users WHERE username = ?', 
                                   (data['username'],)).fetchone()
        
        if existing_user:
            conn.close()
            return jsonify({'error': 'Username already exists'}), 400
        
        # Insert new user with default role as 'farmer'
        try:
            conn.execute('''
                INSERT INTO users (username, password, role, name, email, phone, address) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['username'],
                data['password'],  # In production, hash this password
                'farmer',  # Default role
                data['name'],
                data['email'],
                data['phone'],
                data['address']
            ))
            conn.commit()
            
            # Get the newly created user
            user = conn.execute('SELECT * FROM users WHERE username = ?', 
                              (data['username'],)).fetchone()
            
            return jsonify({
                'message': 'Registration successful',
                'user': dict(user)
            }), 201
            
        except sqlite3.Error as e:
            return jsonify({'error': f'Database error: {str(e)}'}), 500
        finally:
            conn.close()
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', 
                       (username, password)).fetchone()
    conn.close()
    
    if user:
        # Store user info in session
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['user_role'] = user['role']
        session['user_name'] = user['name']
        
        return jsonify({
            'message': 'Login successful',
            'user': dict(user)
        }), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/recommend-crop', methods=['POST'])
def recommend_crop():
    try:
        data = request.get_json()
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Validate all required fields are present and are numbers
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            try:
                float(data[field])
            except (TypeError, ValueError):
                return jsonify({'error': f'Invalid value for {field}. Must be a number.'}), 400
        
        # Convert input data to numpy array
        features = np.array([[
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]])
        
        # Get prediction
        prediction = models['crop_recommendation'].predict(features)[0]
        
        # Determine nutrient status
        nitrogen_status = "High" if data['nitrogen'] > 90 else "Medium" if data['nitrogen'] > 50 else "Low"
        phosphorus_status = "High" if data['phosphorus'] > 90 else "Medium" if data['phosphorus'] > 50 else "Low"
        potassium_status = "High" if data['potassium'] > 90 else "Medium" if data['potassium'] > 50 else "Low"
        
        # Get additional recommendations based on soil parameters
        recommendations = [str(prediction)]
        if data['nitrogen'] < 50:
            recommendations.append("Consider adding nitrogen-rich fertilizers")
        if data['phosphorus'] < 50:
            recommendations.append("Consider adding phosphorus-rich fertilizers")
        if data['potassium'] < 50:
            recommendations.append("Consider adding potassium-rich fertilizers")
        if data['ph'] < 6.0:
            recommendations.append("Consider adding lime to raise soil pH")
        elif data['ph'] > 7.5:
            recommendations.append("Consider adding sulfur to lower soil pH")
        
        return jsonify({
            'recommendations': recommendations,
            'nitrogen_status': nitrogen_status,
            'phosphorus_status': phosphorus_status,
            'potassium_status': potassium_status,
            'input_parameters': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crops', methods=['GET'])
def list_crops():
    conn = get_db_connection()
    crops = conn.execute('SELECT crops.*, users.name as farmer_name FROM crops JOIN users ON crops.farmer_id = users.id WHERE available = 1').fetchall()
    conn.close()
    return jsonify([dict(crop) for crop in crops])

@app.route('/api/farmer/crops', methods=['GET'])
def farmer_crops():
    farmer_id = request.args.get('farmer_id')
    conn = get_db_connection()
    crops = conn.execute('SELECT * FROM crops WHERE farmer_id = ?', (farmer_id,)).fetchall()
    conn.close()
    return jsonify([dict(crop) for crop in crops])

@app.route('/api/add-crop', methods=['POST'])
def add_crop():
    try:
        farmer_id = request.form.get('farmer_id')
        name = request.form.get('name')
        description = request.form.get('description')
        price = float(request.form.get('price'))
        quantity = float(request.form.get('quantity'))
        
        # Handle image upload
        image_url = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Save the file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_url = f'/uploads/{filename}'
        
        conn = get_db_connection()
        try:
            conn.execute('''
                INSERT INTO crops (name, description, price, farmer_id, quantity, available, image_url) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, description, price, farmer_id, quantity, True, image_url))
            conn.commit()
            return jsonify({'message': 'Crop added successfully'}), 201
        except sqlite3.Error as e:
            return jsonify({'error': f'Database error: {str(e)}'}), 500
        finally:
            conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-disease', methods=['POST'])
def detect_disease():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid image file. Please upload a PNG, JPG, or JPEG file.'}), 400
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            # Save the uploaded image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            image_url = f'/uploads/{filename}'
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400
        diseases = {
            'healthy': {
                'description': 'The leaf appears to be healthy with no signs of disease.',
                'treatment': 'No treatment required. Continue regular plant care.',
                'prevention': 'Maintain good plant hygiene and regular monitoring.',
                'probability': 0.85,
                'color': '#4CAF50'
            },
            'leaf_blight': {
                'description': 'Leaf blight is a fungal disease that causes brown spots and leaf death.',
                'treatment': 'Apply fungicide and remove affected leaves.',
                'prevention': 'Ensure proper air circulation and avoid overhead watering.',
                'probability': 0.75,
                'color': '#FF5722'
            },
            'powdery_mildew': {
                'description': 'Powdery mildew appears as white powdery spots on leaves.',
                'treatment': 'Apply sulfur-based fungicide and improve air circulation.',
                'prevention': 'Maintain proper spacing between plants and avoid overhead watering.',
                'probability': 0.70,
                'color': '#9C27B0'
            },
            'rust': {
                'description': 'Rust disease shows as orange or brown powdery spots on leaves.',
                'treatment': 'Apply copper-based fungicide and remove infected leaves.',
                'prevention': 'Use resistant varieties and maintain proper plant spacing.',
                'probability': 0.65,
                'color': '#FFC107'
            },
            'bacterial_spot': {
                'description': 'Bacterial spot causes dark, water-soaked lesions on leaves.',
                'treatment': 'Apply copper-based bactericide and improve air circulation.',
                'prevention': 'Avoid overhead watering and maintain plant hygiene.',
                'probability': 0.60,
                'color': '#2196F3'
            }
        }
        disease = random.choice(list(diseases.keys()))
        confidence = random.uniform(0.7, 0.95)
        pie_chart_data = {
            'labels': list(diseases.keys()),
            'datasets': [{
                'data': [diseases[d]['probability'] for d in diseases.keys()],
                'backgroundColor': [diseases[d]['color'] for d in diseases.keys()]
            }]
        }
        return jsonify({
            'disease_name': disease.replace('_', ' ').title(),
            'confidence': confidence,
            'description': diseases[disease]['description'],
            'treatment': diseases[disease]['treatment'],
            'prevention': diseases[disease]['prevention'],
            'pie_chart_data': pie_chart_data,
            'image_url': image_url,
            'detailed_analysis': {
                'leaf_health': f"The leaf shows {disease.replace('_', ' ')} characteristics with {confidence:.0%} confidence.",
                'severity': random.choice(['Mild', 'Moderate', 'Severe']),
                'affected_area': f"{random.randint(5, 30)}% of the leaf surface",
                'recommended_actions': [
                    diseases[disease]['treatment'],
                    diseases[disease]['prevention'],
                    'Monitor plant health regularly',
                    'Maintain proper plant spacing',
                    'Ensure adequate air circulation'
                ]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Admin endpoints
@app.route('/admin/users', methods=['GET'])
def list_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return jsonify([dict(user) for user in users])

@app.route('/admin/orders', methods=['GET'])
def list_orders():
    conn = get_db_connection()
    orders = conn.execute('''SELECT orders.*, users.name as customer_name, crops.name as crop_name 
                           FROM orders 
                           JOIN users ON orders.customer_id = users.id
                           JOIN crops ON orders.crop_id = crops.id''').fetchall()
    conn.close()
    return jsonify([dict(order) for order in orders])

# Add this after the load_models function
def get_crop_recommendations(season, soil_type, expected_rainfall):
    # This is a simplified version. In a real application, this would use a more sophisticated algorithm
    crop_database = {
        'kharif': {
            'clay': ['Rice', 'Jute', 'Sugarcane'],
            'sandy': ['Maize', 'Groundnut', 'Cotton'],
            'loamy': ['Rice', 'Maize', 'Soybean'],
            'black': ['Cotton', 'Sugarcane', 'Soybean'],
            'red': ['Groundnut', 'Maize', 'Cotton']
        },
        'rabi': {
            'clay': ['Wheat', 'Mustard', 'Gram'],
            'sandy': ['Barley', 'Chickpea', 'Lentil'],
            'loamy': ['Wheat', 'Mustard', 'Peas'],
            'black': ['Wheat', 'Gram', 'Mustard'],
            'red': ['Wheat', 'Chickpea', 'Mustard']
        },
        'zaid': {
            'clay': ['Watermelon', 'Cucumber', 'Pumpkin'],
            'sandy': ['Watermelon', 'Muskmelon', 'Cucumber'],
            'loamy': ['Watermelon', 'Cucumber', 'Pumpkin'],
            'black': ['Watermelon', 'Cucumber', 'Pumpkin'],
            'red': ['Watermelon', 'Muskmelon', 'Cucumber']
        }
    }
    
    recommended_crops = crop_database.get(season, {}).get(soil_type, [])
    return [{'name': crop, 'description': f'Best suited for {soil_type} soil in {season} season'} for crop in recommended_crops]

def get_additional_tips(season, expected_rainfall):
    tips = []
    
    if season == 'kharif':
        if expected_rainfall > 1000:
            tips.append("Consider water drainage systems due to heavy rainfall")
        elif expected_rainfall < 500:
            tips.append("Plan for irrigation systems due to low rainfall")
    elif season == 'rabi':
        if expected_rainfall > 300:
            tips.append("Monitor for waterlogging in wheat fields")
        elif expected_rainfall < 100:
            tips.append("Ensure proper irrigation for winter crops")
    elif season == 'zaid':
        if expected_rainfall > 200:
            tips.append("Protect summer crops from heavy rainfall")
        elif expected_rainfall < 50:
            tips.append("Focus on drought-resistant varieties")
    
    return " ".join(tips) if tips else "Monitor weather forecasts regularly and adjust farming practices accordingly."

@app.route('/api/predict-rain', methods=['POST'])
def predict_rain():
    try:
        data = request.get_json()
        district = data.get('district')
        season = data.get('season')
        soil_type = data.get('soil_type')
        
        if not all([district, season, soil_type]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # In a real application, this would use historical data and weather models
        # For demonstration, we'll generate some realistic values
        base_rainfall = {
            'kharif': random.uniform(800, 1200),
            'rabi': random.uniform(100, 300),
            'zaid': random.uniform(50, 200)
        }
        
        expected_rainfall = base_rainfall.get(season, 0)
        rain_probability = random.uniform(0.6, 0.9)
        
        # Get crop recommendations based on the prediction
        recommended_crops = get_crop_recommendations(season, soil_type, expected_rainfall)
        
        # Get additional tips
        additional_tips = get_additional_tips(season, expected_rainfall)
        
        return jsonify({
            'expected_rainfall': round(expected_rainfall, 2),
            'rain_probability': rain_probability,
            'recommended_crops': recommended_crops,
            'additional_tips': additional_tips
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_crop_details(crop_name):
    # This is a simplified version. In a real application, this would use a database
    crop_database = {
        'rice': {
            'growing_season': 'Kharif (June-October)',
            'growing_duration': '120-150 days',
            'optimal_temperature': '20-35°C',
            'water_requirements': 'High (1200-1500mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '120-150 kg',
                    'description': 'Apply in 3 splits: 1/3 at transplanting, 1/3 at tillering, 1/3 at panicle initiation'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '60-80 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '60-80 kg',
                    'description': 'Apply in 2 splits: 1/2 as basal, 1/2 at panicle initiation'
                }
            ],
            'growing_tips': [
                'Maintain water level at 2-3 inches during vegetative phase',
                'Practice proper weed management',
                'Use certified seeds for better yield',
                'Monitor for diseases like blast and bacterial blight',
                'Practice crop rotation to maintain soil health'
            ],
            'pest_control': [
                {
                    'name': 'Stem Borer',
                    'description': 'Use pheromone traps and apply recommended insecticides'
                },
                {
                    'name': 'Leaf Folder',
                    'description': 'Monitor regularly and spray when damage threshold is reached'
                },
                {
                    'name': 'Brown Plant Hopper',
                    'description': 'Use resistant varieties and apply recommended insecticides'
                }
            ],
            'market_price': '25-30',
            'yield_per_acre': '2.5-3.5 tons'
        },
        'wheat': {
            'growing_season': 'Rabi (October-March)',
            'growing_duration': '120-140 days',
            'optimal_temperature': '15-25°C',
            'water_requirements': 'Moderate (500-700mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '100-120 kg',
                    'description': 'Apply in 3 splits: 1/3 at sowing, 1/3 at tillering, 1/3 at flowering'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '50-60 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '40-50 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Ensure proper seed bed preparation',
                'Use certified seeds',
                'Practice proper weed management',
                'Monitor for rust diseases',
                'Irrigate at critical growth stages'
            ],
            'pest_control': [
                {
                    'name': 'Aphids',
                    'description': 'Monitor regularly and apply recommended insecticides'
                },
                {
                    'name': 'Termites',
                    'description': 'Treat seeds with recommended insecticides'
                },
                {
                    'name': 'Rust Diseases',
                    'description': 'Use resistant varieties and apply recommended fungicides'
                }
            ],
            'market_price': '20-25',
            'yield_per_acre': '2-3 tons'
        },
        'cotton': {
            'growing_season': 'Kharif (June-October)',
            'growing_duration': '150-180 days',
            'optimal_temperature': '20-30°C',
            'water_requirements': 'Moderate (700-900mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '80-100 kg',
                    'description': 'Apply in 3 splits: 1/3 at sowing, 1/3 at squaring, 1/3 at flowering'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '40-50 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '40-50 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Use Bt cotton varieties for better pest resistance',
                'Practice proper spacing',
                'Monitor for bollworms',
                'Practice proper weed management',
                'Irrigate at critical growth stages'
            ],
            'pest_control': [
                {
                    'name': 'Bollworms',
                    'description': 'Use Bt cotton varieties and apply recommended insecticides'
                },
                {
                    'name': 'Sucking Pests',
                    'description': 'Monitor regularly and apply recommended insecticides'
                },
                {
                    'name': 'Leaf Curl Virus',
                    'description': 'Use resistant varieties and control whitefly population'
                }
            ],
            'market_price': '60-70',
            'yield_per_acre': '2-3 quintals'
        },
        'sugarcane': {
            'growing_season': 'Year-round in tropical areas, Kharif/Rabi in others',
            'growing_duration': '10-18 months',
            'optimal_temperature': '20-32°C',
            'water_requirements': 'High (1500-2500mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '150-200 kg',
                    'description': 'Apply in 3 splits: at planting, at tillering, and after last earthing up'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '80-100 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '100-120 kg',
                    'description': 'Apply in 2 splits: 50% basal, 50% at tillering'
                }
            ],
            'growing_tips': [
                'Use healthy setts for planting',
                'Ensure proper drainage',
                'Practice timely weeding and earthing up',
                'Monitor for red rot disease',
                'Manage sugarcane pyrilla and borers'
            ],
            'pest_control': [
                {
                    'name': 'Sugarcane Pyrilla',
                    'description': 'Spray recommended insecticides or use biological control agents'
                },
                {
                    'name': 'Stem Borers',
                    'description': 'Remove affected shoots and apply recommended insecticides'
                },
                {
                    'name': 'Red Rot',
                    'description': 'Use resistant varieties and follow proper sanitation practices'
                }
            ],
            'market_price': '3-4',
            'yield_per_acre': '70-80 tons'
        },
        'maize': {
            'growing_season': 'Kharif and Rabi',
            'growing_duration': '70-110 days (depending on variety)',
            'optimal_temperature': '21-30°C',
            'water_requirements': 'Moderate (500-800mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '100-120 kg',
                    'description': 'Apply in 3 splits: 1/4 at sowing, 1/2 at knee high, 1/4 at tasseling'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '50-60 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '40-50 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Use hybrid varieties for higher yield',
                'Maintain proper plant population',
                'Practice timely weeding',
                'Monitor for fall armyworm',
                'Ensure adequate drainage'
            ],
            'pest_control': [
                {
                    'name': 'Fall Armyworm',
                    'description': 'Monitor regularly and apply recommended insecticides, especially in the whorl'
                },
                {
                    'name': 'Stem Borer',
                    'description': 'Apply insecticides in the funnels of the plants'
                },
                {
                    'name': 'Maize Stalk Rot',
                    'description': 'Use resistant varieties and practice crop rotation'
                }
            ],
            'market_price': '18-22',
            'yield_per_acre': '2.5-3.5 tons'
        },
        'barley': {
            'growing_season': 'Rabi (October-March)',
            'growing_duration': '90-120 days',
            'optimal_temperature': '15-20°C',
            'water_requirements': 'Low to Moderate (400-600mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '60-80 kg',
                    'description': 'Apply in 2 splits: 1/2 at sowing, 1/2 at first irrigation'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '30-40 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '20-30 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Suitable for dry areas',
                'Requires well-drained soil',
                'Monitor for rust and smuts',
                'Ensure proper seed treatment'
            ],
            'pest_control': [
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Armyworms',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Rust Diseases',
                    'description': 'Use resistant varieties and apply fungicides'
                }
            ],
            'market_price': '15-20',
            'yield_per_acre': '1.5-2.5 tons'
        },
        'tomato': {
            'growing_season': 'Year-round in suitable climates',
            'growing_duration': '90-150 days',
            'optimal_temperature': '20-25°C',
            'water_requirements': 'Moderate (600-800mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '80-100 kg',
                    'description': 'Apply in splits: at transplanting, flowering, and fruit development'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '60-80 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '80-100 kg',
                    'description': 'Apply in splits: at transplanting and fruit development'
                }
            ],
            'growing_tips': [
                'Requires staking or support',
                'Prune regularly for better yield',
                'Monitor for early and late blight',
                'Ensure consistent watering'
            ],
            'pest_control': [
                {
                    'name': 'Tomato Fruit Borer',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Whiteflies',
                    'description': 'Use yellow sticky traps and apply insecticides'
                },
                {
                    'name': 'Early and Late Blight',
                    'description': 'Apply fungicides and remove infected leaves'
                }
            ],
            'market_price': '10-20',
            'yield_per_acre': '20-30 tons'
        },
        'onion': {
            'growing_season': 'Rabi (October-March) and Kharif (June-October)',
            'growing_duration': '90-150 days',
            'optimal_temperature': '15-25°C',
            'water_requirements': 'Moderate (350-550mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '100-120 kg',
                    'description': 'Apply in 2-3 splits during vegetative growth'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '50-60 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '50-60 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Requires well-drained soil',
                'Ensure proper spacing',
                'Monitor for thrips and fungal diseases',
                'Harvest when tops fall over'
            ],
            'pest_control': [
                {
                    'name': 'Thrips',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Onion Maggot',
                    'description': 'Apply insecticides to the soil'
                },
                {
                    'name': 'Downy Mildew',
                    'description': 'Apply fungicides and ensure good air circulation'
                }
            ],
            'market_price': '15-30',
            'yield_per_acre': '15-25 tons'
        },
        'cabbage': {
            'growing_season': 'winter (October-March) and early rainy',
            'growing_duration': '90-120 days',
            'optimal_temperature': '15-20°C',
            'water_requirements': 'Moderate (300-500mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '120-150 kg',
                    'description': 'Apply in 2 splits: at transplanting and head formation'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '60-80 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '80-100 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Requires well-drained soil',
                'Ensure proper spacing',
                'Monitor for diamondback moth and clubroot',
                'Ensure consistent moisture'
            ],
            'pest_control': [
                {
                    'name': 'Diamondback Moth',
                    'description': 'Use biological control agents or recommended insecticides'
                },
                {
                    'name': 'Cabbage Worms',
                    'description': 'Apply Bt insecticide'
                },
                {
                    'name': 'Clubroot',
                    'description': 'Use resistant varieties and manage soil pH'
                }
            ],
            'market_price': '5-15',
            'yield_per_acre': '20-40 tons'
        },
        'sunflower': {
            'growing_season': 'rainy'and 'winter',
            'growing_duration': '90-120 days',
            'optimal_temperature': '20-25°C',
            'water_requirements': 'Low to Moderate (400-600mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '60-80 kg',
                    'description': 'Apply in 2 splits: at sowing and flowering'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '40-60 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '40-60 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Requires full sun',
                'Ensure proper spacing',
                'Monitor for head rot and rust',
                'Pollination is important for seed set'
            ],
            'pest_control': [
                {
                    'name': 'Sunflower Moth',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Cutworms',
                    'description': 'Apply insecticides to the soil surface'
                },
                {
                    'name': 'Head Rot',
                    'description': 'Use resistant varieties and ensure good drainage'
                }
            ],
            'market_price': '30-50',
            'yield_per_acre': '1-2 tons'
        },
        'ragi': {
            'growing_season': 'rainy' (June-October),
            'growing_duration': '90-110 days',
            'optimal_temperature': '20-30°C',
            'water_requirements': 'Low to Moderate (400-500mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '40-60 kg',
                    'description': 'Apply in 2 splits: at sowing and tillering'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '20-30 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '20-30 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Drought tolerant crop',
                'Suitable for marginal lands',
                'Requires good seed bed preparation',
                'Monitor for blast and leaf spot diseases'
            ],
            'pest_control': [
                {
                    'name': 'Stem Borer',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Leaf Folder',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Blast Disease',
                    'description': 'Use resistant varieties and apply fungicides'
                }
            ],
            'market_price': '25-35',
            'yield_per_acre': '1.5-2 tons'
        },
        'cauliflower': {
            'growing_season': 'Rabi (October-March)',
            'growing_duration': '90-120 days',
            'optimal_temperature': '15-20°C',
            'water_requirements': 'Moderate (300-500mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '120-150 kg',
                    'description': 'Apply in 2 splits: at transplanting and curd formation'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '60-80 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '80-100 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Requires cool weather',
                'Ensure proper spacing',
                'Blanching may be required for white curds',
                'Monitor for diamondback moth and clubroot'
            ],
            'pest_control': [
                {
                    'name': 'Diamondback Moth',
                    'description': 'Use biological control agents or recommended insecticides'
                },
                {
                    'name': 'Cabbage Worms',
                    'description': 'Apply Bt insecticide'
                },
                {
                    'name': 'Clubroot',
                    'description': 'Use resistant varieties and manage soil pH'
                }
            ],
            'market_price': '10-20',
            'yield_per_acre': '15-25 tons'
        },
        'millet': {
            'growing_season': 'Kharif (June-October)',
            'growing_duration': '70-100 days',
            'optimal_temperature': '25-35°C',
            'water_requirements': 'Low (300-400mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '40-60 kg',
                    'description': 'Apply in 2 splits: at sowing and tillering'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '20-30 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '20-30 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Drought tolerant crop',
                'Suitable for poor soils',
                'Requires minimal inputs',
                'Monitor for blast and smut diseases'
            ],
            'pest_control': [
                {
                    'name': 'Stem Borer',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Shoot Fly',
                    'description': 'Use treated seeds and apply insecticides'
                },
                {
                    'name': 'Smut Diseases',
                    'description': 'Use resistant varieties and practice seed treatment'
                }
            ],
            'market_price': '20-30',
            'yield_per_acre': '1-1.5 tons'
        },
        'groundnut': {
            'growing_season': 'Kharif (June-October)',
            'growing_duration': '90-120 days',
            'optimal_temperature': '25-30°C',
            'water_requirements': 'Moderate (500-700mm)',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '20-30 kg',
                    'description': 'Apply as basal dose'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '40-60 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '30-40 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Calcium (Ca)',
                    'amount': '200-300 kg gypsum',
                    'description': 'Apply at flowering for better pod development'
                }
            ],
            'growing_tips': [
                'Requires sandy loam soil',
                'Ensure proper spacing for pegging',
                'Monitor for leaf spot and rust diseases',
                'Timely harvesting is crucial'
            ],
            'pest_control': [
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Leaf Miner',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Leaf Spot and Rust',
                    'description': 'Apply fungicides'
                }
            ],
            'market_price': '40-60',
            'yield_per_acre': '1.5-2.5 tons'
        },
        'mango': {
            'growing_season': 'Year-round in tropical and subtropical climates',
            'growing_duration': 'Many years (perennial)',
            'optimal_temperature': '24-30°C',
            'water_requirements': 'Moderate, less during flowering',
            'fertilizer_recommendations': [
                {
                    'name': 'NPK',
                    'amount': 'Varies with age and size',
                    'description': 'Apply balanced NPK and micronutrients regularly'
                }
            ],
            'growing_tips': [
                'Requires warm climate',
                'Proper pruning is essential',
                'Monitor for anthracnose and powdery mildew',
                'Irrigate during fruit development'
            ],
            'pest_control': [
                {
                    'name': 'Mango Hopper',
                    'description': 'Monitor and apply recommended insecticides during flowering'
                },
                {
                    'name': 'Fruit Fly',
                    'description': 'Use fruit fly traps and apply insecticides'
                },
                {
                    'name': 'Anthracnose',
                    'description': 'Apply fungicides regularly, especially during rainy season'
                }
            ],
            'market_price': '50-100',
            'yield_per_acre': '8-12 tons (mature tree)'
        },
        'coffee': {
            'growing_season': 'Year-round in suitable climates',
            'growing_duration': 'Many years (perennial)',
            'optimal_temperature': '18-25°C',
            'water_requirements': 'High, especially during flowering and fruit development',
            'fertilizer_recommendations': [
                {
                    'name': 'NPK',
                    'amount': 'Varies with age and yield',
                    'description': 'Apply balanced NPK and micronutrients in split doses'
                }
            ],
            'growing_tips': [
                'Requires shade and well-drained soil',
                'Pruning is essential for shape and yield',
                'Monitor for coffee berry borer and leaf rust',
                'Harvest ripe berries carefully'
            ],
            'pest_control': [
                {
                    'name': 'Coffee Berry Borer',
                    'description': 'Use traps and insecticides as recommended'
                },
                {
                    'name': 'Leaf Rust',
                    'description': 'Use resistant varieties and apply fungicides'
                },
                {
                    'name': 'Mealybugs',
                    'description': 'Use biological control or insecticides'
                }
            ],
            'market_price': '200-400', # per kg of beans
            'yield_per_acre': '0.5-1 ton (of beans)'
        },
        'watermelon': {
            'growing_season': 'Zaid (March-June) and summer',
            'growing_duration': '70-90 days',
            'optimal_temperature': '25-35°C',
            'water_requirements': 'High, especially during fruit development',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '60-80 kg',
                    'description': 'Apply in splits: at planting and vining'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '60-80 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '60-80 kg',
                    'description': 'Apply in splits: at planting and fruit set'
                }
            ],
            'growing_tips': [
                'Requires warm weather and full sun',
                'Provide ample space for vines',
                'Monitor for downy mildew and aphids',
                'Ensure consistent watering'
            ],
            'pest_control': [
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Squash Bugs',
                    'description': 'Remove manually or apply insecticides'
                },
                {
                    'name': 'Downy Mildew',
                    'description': 'Apply fungicides and ensure good air circulation'
                }
            ],
            'market_price': '10-20', # per kg
            'yield_per_acre': '20-40 tons'
        },
        'cucumber': {
            'growing_season': 'Summer and Zaid',
            'growing_duration': '50-70 days',
            'optimal_temperature': '20-30°C',
            'water_requirements': 'High and consistent',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '80-100 kg',
                    'description': 'Apply in splits: at planting and every 2-3 weeks'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '50-70 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '60-80 kg',
                    'description': 'Apply in splits: at planting and fruit development'
                }
            ],
            'growing_tips': [
                'Requires warm weather and full sun',
                'Can be grown on trellises or on the ground',
                'Monitor for powdery mildew and cucumber beetles',
                'Harvest regularly to encourage more fruiting'
            ],
            'pest_control': [
                {
                    'name': 'Cucumber Beetles',
                    'description': 'Use row covers and apply insecticides'
                },
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Powdery Mildew',
                    'description': 'Apply fungicides and ensure good air circulation'
                }
            ],
            'market_price': '15-25', # per kg
            'yield_per_acre': '10-20 tons'
        },
        'pumpkin': {
            'growing_season': 'Kharif and Zaid',
            'growing_duration': '90-120 days',
            'optimal_temperature': '20-30°C',
            'water_requirements': 'Moderate to High',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '80-100 kg',
                    'description': 'Apply in splits: at planting and vining'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '60-80 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '60-80 kg',
                    'description': 'Apply in splits: at planting and fruit set'
                }
            ],
            'growing_tips': [
                'Requires warm weather and ample space',
                'Pollination is important for fruit set',
                'Monitor for vine borer and powdery mildew',
                'Protect developing fruits from pests'
            ],
            'pest_control': [
                {
                    'name': 'Squash Vine Borer',
                    'description': 'Apply insecticides to the base of plants'
                },
                {
                    'name': 'Cucumber Beetles',
                    'description': 'Use row covers and apply insecticides'
                },
                {
                    'name': 'Powdery Mildew',
                    'description': 'Apply fungicides and ensure good air circulation'
                }
            ],
            'market_price': '5-10', # per kg
            'yield_per_acre': '15-30 tons'
        },
        'banana': {
            'growing_season': 'Year-round in tropical climates',
            'growing_duration': '9-12 months (per bunch)',
            'optimal_temperature': '25-30°C',
            'water_requirements': 'High and consistent',
            'fertilizer_recommendations': [
                {
                    'name': 'NPK',
                    'amount': 'Varies with variety and soil',
                    'description': 'Apply balanced NPK and micronutrients regularly'
                }
            ],
            'growing_tips': [
                'Requires well-drained soil and protection from wind',
                'Remove suckers to maintain plant population',
                'Monitor for Panama disease and Sigatoka leaf spot',
                'Support bunches to prevent toppling'
            ],
            'pest_control': [
                {
                    'name': 'Banana Weevil',
                    'description': 'Use traps and insecticides'
                },
                {
                    'name': 'Nematodes',
                    'description': 'Use resistant varieties and soil treatment'
                },
                {
                    'name': 'Sigatoka Leaf Spot',
                    'description': 'Apply fungicides and remove affected leaves'
                }
            ],
            'market_price': '10-30', # per dozen
            'yield_per_acre': '30-60 tons'
        },
        'grapes': {
            'growing_season': 'Depends on variety and climate',
            'growing_duration': 'Many years (perennial)',
            'optimal_temperature': '15-30°C',
            'water_requirements': 'Moderate, less during ripening',
            'fertilizer_recommendations': [
                {
                    'name': 'NPK',
                    'amount': 'Varies with variety and soil',
                    'description': 'Apply balanced NPK and micronutrients based on soil test'
                }
            ],
            'growing_tips': [
                'Requires support structures (trellises)',
                'Pruning is crucial for fruit production',
                'Monitor for powdery mildew and downy mildew',
                'Ensure good air circulation'
            ],
            'pest_control': [
                {
                    'name': 'Grapevine Moth',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Mealybugs',
                    'description': 'Use biological control or insecticides'
                },
                {
                    'name': 'Powdery and Downy Mildew',
                    'description': 'Apply fungicides regularly'
                }
            ],
            'market_price': '40-80', # per kg
            'yield_per_acre': '10-20 tons'
        },
        'lady fingers': {
            'growing_season': 'Kharif and Zaid',
            'growing_duration': '50-70 days',
            'optimal_temperature': '25-35°C',
            'water_requirements': 'Moderate and consistent',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '60-80 kg',
                    'description': 'Apply in 2 splits: at sowing and flowering'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '40-60 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '40-60 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Requires warm weather and full sun',
                'Harvest pods when young and tender',
                'Monitor for yellow vein mosaic virus and fruit borer',
                'Ensure proper spacing'
            ],
            'pest_control': [
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Fruit Borer',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Yellow Vein Mosaic Virus',
                    'description': 'Use resistant varieties and control whiteflies'
                }
            ],
            'market_price': '20-30', # per kg
            'yield_per_acre': '8-15 tons'
        },
        'capsicum': {
            'growing_season': 'Year-round in suitable climates',
            'growing_duration': '90-120 days',
            'optimal_temperature': '20-25°C',
            'water_requirements': 'Moderate and consistent',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '100-120 kg',
                    'description': 'Apply in splits: at transplanting and flowering'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '60-80 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '80-100 kg',
                    'description': 'Apply in splits: at transplanting and fruit development'
                }
            ],
            'growing_tips': [
                'Requires well-drained soil and full sun',
                'Provide support for plants',
                'Monitor for thrips and bacterial spot',
                'Ensure consistent watering'
            ],
            'pest_control': [
                {
                    'name': 'Thrips',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Bacterial Spot',
                    'description': 'Apply copper-based bactericides and remove infected leaves'
                }
            ],
            'market_price': '30-50', # per kg
            'yield_per_acre': '20-30 tons'
        },
        'potato': {
            'growing_season': 'Rabi (October-March)',
            'growing_duration': '70-120 days',
            'optimal_temperature': '15-20°C',
            'water_requirements': 'Moderate and consistent',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '100-120 kg',
                    'description': 'Apply in 2 splits: at planting and earthing up'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '60-80 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '100-120 kg',
                    'description': 'Apply entire dose as basal dressing'
                }
            ],
            'growing_tips': [
                'Requires well-drained soil',
                'Earthing up is important',
                'Monitor for late blight and potato tuber moth',
                'Use certified seed potatoes'
            ],
            'pest_control': [
                {
                    'name': 'Potato Tuber Moth',
                    'description': 'Use pheromone traps and insecticides'
                },
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Late Blight',
                    'description': 'Apply fungicides regularly'
                }
            ],
            'market_price': '10-20', # per kg
            'yield_per_acre': '20-30 tons'
        },
        'radish': {
            'growing_season': 'Year-round',
            'growing_duration': '25-40 days',
            'optimal_temperature': '10-20°C',
            'water_requirements': 'Moderate and consistent',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '40-60 kg',
                    'description': 'Apply as basal dose'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '30-40 kg',
                    'description': 'Apply as basal dose'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '30-40 kg',
                    'description': 'Apply as basal dose'
                }
            ],
            'growing_tips': [
                'Fast-growing crop',
                'Requires well-drained soil',
                'Monitor for flea beetles and aphids',
                'Harvest promptly to avoid woodiness'
            ],
            'pest_control': [
                {
                    'name': 'Flea Beetles',
                    'description': 'Use row covers or apply insecticides'
                },
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Root Maggots',
                    'description': 'Practice crop rotation and soil treatment'
                }
            ],
            'market_price': '10-20', # per kg
            'yield_per_acre': '10-15 tons'
        },
        'brinjal': {
            'growing_season': 'Year-round in suitable climates',
            'growing_duration': '100-150 days',
            'optimal_temperature': '20-30°C',
            'water_requirements': 'Moderate and consistent',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '100-120 kg',
                    'description': 'Apply in splits: at transplanting, flowering, and fruiting'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '60-80 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '80-100 kg',
                    'description': 'Apply in splits: at transplanting and fruiting'
                }
            ],
            'growing_tips': [
                'Requires well-drained soil and full sun',
                'Provide support for plants',
                'Monitor for fruit and shoot borer and little leaf disease',
                'Harvest fruits when young and glossy'
            ],
            'pest_control': [
                {
                    'name': 'Fruit and Shoot Borer',
                    'description': 'Remove affected parts and apply recommended insecticides'
                },
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Little Leaf',
                    'description': 'Remove diseased plants and control insect vectors'
                }
            ],
            'market_price': '20-30', # per kg
            'yield_per_acre': '20-30 tons'
        },
        'chilly': {
            'growing_season': 'Year-round in suitable climates',
            'growing_duration': '100-150 days',
            'optimal_temperature': '20-25°C',
            'water_requirements': 'Moderate and consistent',
            'fertilizer_recommendations': [
                {
                    'name': 'Nitrogen (N)',
                    'amount': '80-100 kg',
                    'description': 'Apply in splits: at transplanting, flowering, and fruiting'
                },
                {
                    'name': 'Phosphorus (P2O5)',
                    'amount': '50-70 kg',
                    'description': 'Apply entire dose as basal dressing'
                },
                {
                    'name': 'Potassium (K2O)',
                    'amount': '60-80 kg',
                    'description': 'Apply in splits: at transplanting and fruiting'
                }
            ],
            'growing_tips': [
                'Requires well-drained soil and full sun',
                'Provide support for plants',
                'Monitor for thrips and powdery mildew',
                'Ensure consistent watering'
            ],
            'pest_control': [
                {
                    'name': 'Thrips',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Aphids',
                    'description': 'Monitor and apply recommended insecticides'
                },
                {
                    'name': 'Powdery Mildew',
                    'description': 'Apply fungicides and ensure good air circulation'
                }
            ],
            'market_price': '40-80', # per kg
            'yield_per_acre': '10-15 tons'
        }
    }
    
    return crop_database.get(crop_name.lower())

@app.route('/api/crop-details', methods=['POST'])
def crop_details():
    try:
        data = request.get_json()
        crop_name = data.get('crop_name')
        
        if not crop_name:
            return jsonify({'error': 'Crop name is required'}), 400
        
        crop_details = get_crop_details(crop_name)
        if not crop_details:
            return jsonify({'error': f'Details not found for crop: {crop_name}'}), 404
        
        return jsonify({
            'crop_name': crop_name.title(),
            **crop_details
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/estimate-cost', methods=['POST'])
def estimate_cost():
    try:
        data = request.get_json()
        land_area = float(data.get('land_area', 0))
        crop_type = data.get('crop_type')
        
        if not land_area or not crop_type:
            return jsonify({'error': 'Land area and crop type are required'}), 400
        
        # Crop-specific requirements and costs (per acre)
        crop_data = {
            'rice': {
                'seed_quantity': 40,  # kg per acre
                'seed_cost_per_kg': 50,
                'fertilizer_quantity': 200,  # kg per acre
                'fertilizer_cost_per_kg': 30,
                'labor_days': 25,  # days per acre
                'labor_cost_per_day': 500,
                'irrigation_cost': 5000,
                'pesticide_quantity': 5,  # liters per acre
                'pesticide_cost_per_liter': 800,
                'machinery_cost': 3000,
                'expected_yield': '2.5-3.5',  # tons per acre
                'market_price': 25  # per kg
            },
            'wheat': {
                'seed_quantity': 100,
                'seed_cost_per_kg': 40,
                'fertilizer_quantity': 150,
                'fertilizer_cost_per_kg': 30,
                'labor_days': 20,
                'labor_cost_per_day': 500,
                'irrigation_cost': 4000,
                'pesticide_quantity': 4,
                'pesticide_cost_per_liter': 800,
                'machinery_cost': 2500,
                'expected_yield': '2-3',
                'market_price': 20
            },
            'cotton': {
                'seed_quantity': 5,
                'seed_cost_per_kg': 200,
                'fertilizer_quantity': 180,
                'fertilizer_cost_per_kg': 30,
                'labor_days': 30,
                'labor_cost_per_day': 500,
                'irrigation_cost': 4500,
                'pesticide_quantity': 6,
                'pesticide_cost_per_liter': 800,
                'machinery_cost': 3500,
                'expected_yield': '2-3',
                'market_price': 60
            },
            'sugarcane': {
                'seed_quantity': 40000,  # setts per acre
                'seed_cost_per_kg': 5,
                'fertilizer_quantity': 250,
                'fertilizer_cost_per_kg': 30,
                'labor_days': 35,
                'labor_cost_per_day': 500,
                'irrigation_cost': 6000,
                'pesticide_quantity': 7,
                'pesticide_cost_per_liter': 800,
                'machinery_cost': 4000,
                'expected_yield': '70-80',
                'market_price': 3
            },
            'maize': {
                'seed_quantity': 20,
                'seed_cost_per_kg': 100,
                'fertilizer_quantity': 160,
                'fertilizer_cost_per_kg': 30,
                'labor_days': 22,
                'labor_cost_per_day': 500,
                'irrigation_cost': 4500,
                'pesticide_quantity': 5,
                'pesticide_cost_per_liter': 800,
                'machinery_cost': 3000,
                'expected_yield': '2.5-3.5',
                'market_price': 18
            },
            'barley': {
                'seed_quantity': 100, 'seed_cost_per_kg': 35,
                'fertilizer_quantity': 120, 'fertilizer_cost_per_kg': 30,
                'labor_days': 18, 'labor_cost_per_day': 500,
                'irrigation_cost': 3500,
                'pesticide_quantity': 3, 'pesticide_cost_per_liter': 700,
                'machinery_cost': 2000,
                'expected_yield': '1.5-2.5', 'market_price': 18
            },
            'tomato': {
                'seed_quantity': 0.5, 'seed_cost_per_kg': 3000,
                'fertilizer_quantity': 150, 'fertilizer_cost_per_kg': 35,
                'labor_days': 40, 'labor_cost_per_day': 500,
                'irrigation_cost': 6000,
                'pesticide_quantity': 8, 'pesticide_cost_per_liter': 900,
                'machinery_cost': 4000,
                'expected_yield': '20-30', 'market_price': 15
            },
            'onion': {
                'seed_quantity': 5, 'seed_cost_per_kg': 1000,
                'fertilizer_quantity': 120, 'fertilizer_cost_per_kg': 30,
                'labor_days': 30, 'labor_cost_per_day': 500,
                'irrigation_cost': 4000,
                'pesticide_quantity': 4, 'pesticide_cost_per_liter': 700,
                'machinery_cost': 2500,
                'expected_yield': '15-25', 'market_price': 20
            },
            'cabbage': {
                'seed_quantity': 0.2, 'seed_cost_per_kg': 4000,
                'fertilizer_quantity': 180, 'fertilizer_cost_per_kg': 30,
                'labor_days': 25, 'labor_cost_per_day': 500,
                'irrigation_cost': 4500,
                'pesticide_quantity': 5, 'pesticide_cost_per_liter': 800,
                'machinery_cost': 3000,
                'expected_yield': '20-40', 'market_price': 10
            },
            'sunflower': {
                'seed_quantity': 6, 'seed_cost_per_kg': 300,
                'fertilizer_quantity': 100, 'fertilizer_cost_per_kg': 30,
                'labor_days': 15, 'labor_cost_per_day': 500,
                'irrigation_cost': 3000,
                'pesticide_quantity': 3, 'pesticide_cost_per_liter': 700,
                'machinery_cost': 2000,
                'expected_yield': '1-2', 'market_price': 40
            },
            'ragi': {
                'seed_quantity': 10, 'seed_cost_per_kg': 80,
                'fertilizer_quantity': 80, 'fertilizer_cost_per_kg': 25,
                'labor_days': 15, 'labor_cost_per_day': 500,
                'irrigation_cost': 2500,
                'pesticide_quantity': 2, 'pesticide_cost_per_liter': 600,
                'machinery_cost': 1500,
                'expected_yield': '1.5-2', 'market_price': 30
            },
            'cauliflower': {
                'seed_quantity': 0.1, 'seed_cost_per_kg': 5000,
                'fertilizer_quantity': 180, 'fertilizer_cost_per_kg': 30,
                'labor_days': 28, 'labor_cost_per_day': 500,
                'irrigation_cost': 4800,
                'pesticide_quantity': 6, 'pesticide_cost_per_liter': 800,
                'machinery_cost': 3200,
                'expected_yield': '15-25', 'market_price': 15
            },
            'millet': {
                'seed_quantity': 8, 'seed_cost_per_kg': 60,
                'fertilizer_quantity': 70, 'fertilizer_cost_per_kg': 25,
                'labor_days': 14, 'labor_cost_per_day': 500,
                'irrigation_cost': 2000,
                'pesticide_quantity': 2, 'pesticide_cost_per_liter': 500,
                'machinery_cost': 1200,
                'expected_yield': '1-1.5', 'market_price': 25
            },
            'groundnut': {
                'seed_quantity': 100, 'seed_cost_per_kg': 70,
                'fertilizer_quantity': 100, 'fertilizer_cost_per_kg': 30,
                'labor_days': 20, 'labor_cost_per_day': 500,
                'irrigation_cost': 4000,
                'pesticide_quantity': 4, 'pesticide_cost_per_liter': 700,
                'machinery_cost': 2800,
                'expected_yield': '1.5-2.5', 'market_price': 50
            },
            'mango': {
                'seed_quantity': 10, 'seed_cost_per_kg': 200,
                'fertilizer_quantity': 500, 'fertilizer_cost_per_kg': 20,
                'labor_days': 50, 'labor_cost_per_day': 500,
                'irrigation_cost': 7000,
                'pesticide_quantity': 10, 'pesticide_cost_per_liter': 1000,
                'machinery_cost': 5000,
                'expected_yield': '8-12', 'market_price': 70
            },
            'coffee': {
                'seed_quantity': 50, 'seed_cost_per_kg': 100,
                'fertilizer_quantity': 300, 'fertilizer_cost_per_kg': 25,
                'labor_days': 60, 'labor_cost_per_day': 500,
                'irrigation_cost': 8000,
                'pesticide_quantity': 12, 'pesticide_cost_per_liter': 900,
                'machinery_cost': 6000,
                'expected_yield': '0.5-1', 'market_price': 300
            },
            'watermelon': {
                'seed_quantity': 0.3, 'seed_cost_per_kg': 2500,
                'fertilizer_quantity': 150, 'fertilizer_cost_per_kg': 30,
                'labor_days': 30, 'labor_cost_per_day': 500,
                'irrigation_cost': 5000,
                'pesticide_quantity': 6, 'pesticide_cost_per_liter': 800,
                'machinery_cost': 3500,
                'expected_yield': '20-40', 'market_price': 15
            },
            'cucumber': {
                'seed_quantity': 0.2, 'seed_cost_per_kg': 3000,
                'fertilizer_quantity': 120, 'fertilizer_cost_per_kg': 30,
                'labor_days': 25, 'labor_cost_per_day': 500,
                'irrigation_cost': 4000,
                'pesticide_quantity': 5, 'pesticide_cost_per_liter': 700,
                'machinery_cost': 2800,
                'expected_yield': '10-20', 'market_price': 20
            },
            'pumpkin': {
                'seed_quantity': 0.5, 'seed_cost_per_kg': 1500,
                'fertilizer_quantity': 130, 'fertilizer_cost_per_kg': 30,
                'labor_days': 28, 'labor_cost_per_day': 500,
                'irrigation_cost': 4500,
                'pesticide_quantity': 6, 'pesticide_cost_per_liter': 750,
                'machinery_cost': 3000,
                'expected_yield': '15-30', 'market_price': 8
            },
            'banana': {
                'seed_quantity': 200, 'seed_cost_per_kg': 50,
                'fertilizer_quantity': 400, 'fertilizer_cost_per_kg': 25,
                'labor_days': 70, 'labor_cost_per_day': 500,
                'irrigation_cost': 9000,
                'pesticide_quantity': 15, 'pesticide_cost_per_liter': 800,
                'machinery_cost': 7000,
                'expected_yield': '30-60', 'market_price': 20
            },
            'grapes': {
                'seed_quantity': 100, 'seed_cost_per_kg': 80,
                'fertilizer_quantity': 300, 'fertilizer_cost_per_kg': 30,
                'labor_days': 50, 'labor_cost_per_day': 500,
                'irrigation_cost': 7000,
                'pesticide_quantity': 8, 'pesticide_cost_per_liter': 900,
                'machinery_cost': 5000,
                'expected_yield': '10-20', 'market_price': 60
            },
            'lady fingers': {
                'seed_quantity': 8, 'seed_cost_per_kg': 150,
                'fertilizer_quantity': 100, 'fertilizer_cost_per_kg': 30,
                'labor_days': 20, 'labor_cost_per_day': 500,
                'irrigation_cost': 3000,
                'pesticide_quantity': 4, 'pesticide_cost_per_liter': 600,
                'machinery_cost': 2000,
                'expected_yield': '8-15', 'market_price': 25
            },
            'capsicum': {
                'seed_quantity': 0.1, 'seed_cost_per_kg': 6000,
                'fertilizer_quantity': 150, 'fertilizer_cost_per_kg': 35,
                'labor_days': 35, 'labor_cost_per_day': 500,
                'irrigation_cost': 5500,
                'pesticide_quantity': 7, 'pesticide_cost_per_liter': 850,
                'machinery_cost': 3800,
                'expected_yield': '20-30', 'market_price': 40
            },
            'potato': {
                'seed_quantity': 1500, 'seed_cost_per_kg': 20,
                'fertilizer_quantity': 150, 'fertilizer_cost_per_kg': 30,
                'labor_days': 30, 'labor_cost_per_day': 500,
                'irrigation_cost': 4500,
                'pesticide_quantity': 5, 'pesticide_cost_per_liter': 700,
                'machinery_cost': 3000,
                'expected_yield': '20-30', 'market_price': 15
            },
            'radish': {
                'seed_quantity': 10, 'seed_cost_per_kg': 50,
                'fertilizer_quantity': 80, 'fertilizer_cost_per_kg': 25,
                'labor_days': 10, 'labor_cost_per_day': 500,
                'irrigation_cost': 2000,
                'pesticide_quantity': 2, 'pesticide_cost_per_liter': 500,
                'machinery_cost': 1000,
                'expected_yield': '10-15', 'market_price': 15
            },
            'brinjal': {
                'seed_quantity': 0.4, 'seed_cost_per_kg': 3500,
                'fertilizer_quantity': 120, 'fertilizer_cost_per_kg': 30,
                'labor_days': 30, 'labor_cost_per_day': 500,
                'irrigation_cost': 4000,
                'pesticide_quantity': 6, 'pesticide_cost_per_liter': 700,
                'machinery_cost': 2800,
                'expected_yield': '20-30', 'market_price': 25
            },
            'chilly': {
                'seed_quantity': 0.3, 'seed_cost_per_kg': 4000,
                'fertilizer_quantity': 100, 'fertilizer_cost_per_kg': 30,
                'labor_days': 35, 'labor_cost_per_day': 500,
                'irrigation_cost': 4500,
                'pesticide_quantity': 7, 'pesticide_cost_per_liter': 800,
                'machinery_cost': 3000,
                'expected_yield': '10-15', 'market_price': 60
            }
        }
        
        if crop_type not in crop_data:
            return jsonify({'error': f'Invalid crop type: {crop_type}'}), 400
        
        crop = crop_data[crop_type]
        
        # Calculate quantities
        seed_quantity = crop['seed_quantity'] * land_area
        fertilizer_quantity = crop['fertilizer_quantity'] * land_area
        labor_days = crop['labor_days'] * land_area
        pesticide_quantity = crop['pesticide_quantity'] * land_area
        
        # Calculate costs
        seed_cost = seed_quantity * crop['seed_cost_per_kg']
        fertilizer_cost = fertilizer_quantity * crop['fertilizer_cost_per_kg']
        labor_cost = labor_days * crop['labor_cost_per_day']
        irrigation_cost = crop['irrigation_cost'] * land_area
        pesticide_cost = pesticide_quantity * crop['pesticide_cost_per_liter']
        machinery_cost = crop['machinery_cost'] * land_area
        
        # Calculate miscellaneous costs (10% of total direct costs)
        direct_costs = seed_cost + fertilizer_cost + labor_cost + irrigation_cost + pesticide_cost + machinery_cost
        miscellaneous_cost = direct_costs * 0.1
        
        # Calculate total cost
        total_cost = direct_costs + miscellaneous_cost
        cost_per_acre = total_cost / land_area
        
        # Calculate expected returns
        expected_yield = crop['expected_yield']
        market_price = crop['market_price']
        min_yield = float(expected_yield.split('-')[0])
        expected_revenue = min_yield * land_area * market_price * 1000  # Convert tons to kg
        expected_profit = expected_revenue - total_cost
        profit_margin = (expected_profit / total_cost) * 100 if total_cost > 0 else 0
        
        # Generate additional information
        additional_info = get_crop_specific_info(crop_type, total_cost, land_area)
        
        return jsonify({
            'crop_type': crop_type,
            'seed_quantity': round(seed_quantity, 2),
            'fertilizer_quantity': round(fertilizer_quantity, 2),
            'labor_days': round(labor_days, 2),
            'pesticide_quantity': round(pesticide_quantity, 2),
            'seed_cost': seed_cost,
            'fertilizer_cost': fertilizer_cost,
            'labor_cost': labor_cost,
            'irrigation_cost': irrigation_cost,
            'pesticide_cost': pesticide_cost,
            'machinery_cost': machinery_cost,
            'miscellaneous_cost': miscellaneous_cost,
            'total_cost': total_cost,
            'cost_per_acre': cost_per_acre,
            'expected_yield': expected_yield,
            'expected_revenue': expected_revenue,
            'expected_profit': expected_profit,
            'profit_margin': profit_margin,
            'additional_info': additional_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_crop_specific_info(crop_type, total_cost, land_area):
    # This function provides crop-specific information and recommendations
    crop_info = {
        'rice': {
            'avg_yield': '2.5-3.5 tons per acre',
            'market_price': '₹25-30 per kg',
            'profit_margin': '30-40%',
            'tips': [
                'Ensure proper water management',
                'Use certified seeds for better yield',
                'Monitor for diseases like blast and bacterial blight'
            ]
        },
        'wheat': {
            'avg_yield': '2-3 tons per acre',
            'market_price': '₹20-25 per kg',
            'profit_margin': '25-35%',
            'tips': [
                'Ensure proper seed bed preparation',
                'Monitor for rust diseases',
                'Irrigate at critical growth stages'
            ]
        },
        'cotton': {
            'avg_yield': '2-3 quintals per acre',
            'market_price': '₹60-70 per kg',
            'profit_margin': '35-45%',
            'tips': [
                'Use Bt cotton varieties for better pest resistance',
                'Monitor for bollworms',
                'Practice proper spacing'
            ]
        },
        'sugarcane': {
            'avg_yield': '70-80 tons per acre',
            'market_price': '₹3-4 per kg',
            'profit_margin': '40-50%',
            'tips': [
                'Ensure proper irrigation',
                'Monitor for red rot disease',
                'Use recommended spacing'
            ]
        },
        'maize': {
            'avg_yield': '2.5-3.5 tons per acre',
            'market_price': '₹18-22 per kg',
            'profit_margin': '30-40%',
            'tips': [
                'Ensure proper weed management',
                'Monitor for stem borer',
                'Use recommended spacing'
            ]
        }
    }
    
    info = crop_info.get(crop_type, {})
    if not info:
        return "No specific information available for this crop type."
    
    estimated_yield = float(info['avg_yield'].split('-')[0]) * land_area
    market_price = float(info['market_price'].split('-')[0].replace('₹', ''))
    estimated_revenue = estimated_yield * market_price
    estimated_profit = estimated_revenue - total_cost
    profit_margin = (estimated_profit / total_cost) * 100 if total_cost > 0 else 0
    
    return f"""
    Based on your inputs and current market conditions:
    • Estimated Yield: {estimated_yield:.2f} tons
    • Estimated Revenue: ₹{estimated_revenue:.2f}
    • Estimated Profit: ₹{estimated_profit:.2f}
    • Expected Profit Margin: {profit_margin:.1f}%
    
    Tips for {crop_type.title()} cultivation:
    {chr(10).join('• ' + tip for tip in info['tips'])}
    
    Note: These are approximate values. Actual results may vary based on various factors like weather conditions, 
    soil quality, and farming practices.
    """

@app.route('/api/analyze-plant', methods=['POST'])
def analyze_plant():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Read the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # In a real application, you would:
        # 1. Preprocess the image
        # 2. Pass it through your trained model
        # 3. Get predictions for plant health and nutrient deficiencies
        
        # For demonstration, we'll return some sample recommendations
        # In a real app, this would come from your ML model
        health_status = random.choice(['Healthy', 'Slightly Stressed', 'Moderately Stressed'])
        detected_issues = random.choice([
            'Slight nitrogen deficiency',
            'Mild phosphorus deficiency',
            'Potassium deficiency',
            'No major issues detected',
            'Slight water stress'
        ])
        
        # Sample fertilizer recommendations based on detected issues
        fertilizer_recommendations = {
            'nitrogen deficiency': [
                {
                    'name': 'Urea',
                    'amount': '50-60 kg per acre',
                    'application': 'Apply in 2-3 splits during the growing season'
                },
                {
                    'name': 'Ammonium Sulfate',
                    'amount': '40-50 kg per acre',
                    'application': 'Apply as basal dose before planting'
                }
            ],
            'phosphorus deficiency': [
                {
                    'name': 'DAP (Di-Ammonium Phosphate)',
                    'amount': '60-70 kg per acre',
                    'application': 'Apply as basal dose before planting'
                },
                {
                    'name': 'SSP (Single Super Phosphate)',
                    'amount': '100-120 kg per acre',
                    'application': 'Apply as basal dose before planting'
                }
            ],
            'potassium deficiency': [
                {
                    'name': 'MOP (Muriate of Potash)',
                    'amount': '30-40 kg per acre',
                    'application': 'Apply in 2 splits: 50% basal, 50% at flowering'
                },
                {
                    'name': 'SOP (Sulfate of Potash)',
                    'amount': '35-45 kg per acre',
                    'application': 'Apply in 2 splits: 50% basal, 50% at flowering'
                }
            ],
            'water stress': [
                {
                    'name': 'Water-Soluble NPK (19-19-19)',
                    'amount': '5-7 kg per acre',
                    'application': 'Apply through drip irrigation or foliar spray'
                }
            ],
            'no issues': [
                {
                    'name': 'Balanced NPK (14-14-14)',
                    'amount': '40-50 kg per acre',
                    'application': 'Apply as basal dose before planting'
                }
            ]
        }
        
        # Select appropriate fertilizers based on detected issues
        if 'nitrogen' in detected_issues.lower():
            recommended_fertilizers = fertilizer_recommendations['nitrogen deficiency']
        elif 'phosphorus' in detected_issues.lower():
            recommended_fertilizers = fertilizer_recommendations['phosphorus deficiency']
        elif 'potassium' in detected_issues.lower():
            recommended_fertilizers = fertilizer_recommendations['potassium deficiency']
        elif 'water' in detected_issues.lower():
            recommended_fertilizers = fertilizer_recommendations['water stress']
        else:
            recommended_fertilizers = fertilizer_recommendations['no issues']
        
        # Additional recommendations based on health status
        additional_recommendations = {
            'Healthy': 'Continue current management practices. Monitor plant health regularly.',
            'Slightly Stressed': 'Increase monitoring frequency. Consider slight adjustments to irrigation schedule.',
            'Moderately Stressed': 'Implement immediate corrective measures. Consider soil testing for detailed analysis.'
        }
        
        return jsonify({
            'health_status': health_status,
            'detected_issues': detected_issues,
            'recommended_fertilizers': recommended_fertilizers,
            'additional_recommendations': additional_recommendations[health_status]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cost-estimator-crops', methods=['GET'])
def list_cost_estimator_crops():
    # Crop-specific requirements and costs (per acre)
    # This dictionary is duplicated from estimate_cost, consider refactoring if possible
    crop_data = {
        'rice': {
            'seed_quantity': 40,  # kg per acre
            'seed_cost_per_kg': 50,
            'fertilizer_quantity': 200,  # kg per acre
            'fertilizer_cost_per_kg': 30,
            'labor_days': 25,  # days per acre
            'labor_cost_per_day': 500,
            'irrigation_cost': 5000,
            'pesticide_quantity': 5,  # liters per acre
            'pesticide_cost_per_liter': 800,
            'machinery_cost': 3000,
            'expected_yield': '2.5-3.5',  # tons per acre
            'market_price': 25  # per kg
        },
        'wheat': {
            'seed_quantity': 100,
            'seed_cost_per_kg': 40,
            'fertilizer_quantity': 150,
            'fertilizer_cost_per_kg': 30,
            'labor_days': 20,
            'labor_cost_per_day': 500,
            'irrigation_cost': 4000,
            'pesticide_quantity': 4,
            'pesticide_cost_per_liter': 800,
            'machinery_cost': 2500,
            'expected_yield': '2-3',
            'market_price': 20
        },
        'cotton': {
            'seed_quantity': 5,
            'seed_cost_per_kg': 200,
            'fertilizer_quantity': 180,
            'fertilizer_cost_per_kg': 30,
            'labor_days': 30,
            'labor_cost_per_day': 500,
            'irrigation_cost': 4500,
            'pesticide_quantity': 6,
            'pesticide_cost_per_liter': 800,
            'machinery_cost': 3500,
            'expected_yield': '2-3',
            'market_price': 60
        },
        'sugarcane': {
            'seed_quantity': 40000,  # setts per acre
            'seed_cost_per_kg': 5,
            'fertilizer_quantity': 250,
            'fertilizer_cost_per_kg': 30,
            'labor_days': 35,
            'labor_cost_per_day': 500,
            'irrigation_cost': 6000,
            'pesticide_quantity': 7,
            'pesticide_cost_per_liter': 800,
            'machinery_cost': 4000,
            'expected_yield': '70-80',
            'market_price': 3
        },
        'maize': {
            'seed_quantity': 20,
            'seed_cost_per_kg': 100,
            'fertilizer_quantity': 160,
            'fertilizer_cost_per_kg': 30,
            'labor_days': 22,
            'labor_cost_per_day': 500,
            'irrigation_cost': 4500,
            'pesticide_quantity': 5,
            'pesticide_cost_per_liter': 800,
            'machinery_cost': 3000,
            'expected_yield': '2.5-3.5',
            'market_price': 18
        },
        'barley': {
            'seed_quantity': 100, 'seed_cost_per_kg': 35,
            'fertilizer_quantity': 120, 'fertilizer_cost_per_kg': 30,
            'labor_days': 18, 'labor_cost_per_day': 500,
            'irrigation_cost': 3500,
            'pesticide_quantity': 3, 'pesticide_cost_per_liter': 700,
            'machinery_cost': 2000,
            'expected_yield': '1.5-2.5', 'market_price': 18
        },
        'tomato': {
            'seed_quantity': 0.5, 'seed_cost_per_kg': 3000,
            'fertilizer_quantity': 150, 'fertilizer_cost_per_kg': 35,
            'labor_days': 40, 'labor_cost_per_day': 500,
            'irrigation_cost': 6000,
            'pesticide_quantity': 8, 'pesticide_cost_per_liter': 900,
            'machinery_cost': 4000,
            'expected_yield': '20-30', 'market_price': 15
        },
        'onion': {
            'seed_quantity': 5, 'seed_cost_per_kg': 1000,
            'fertilizer_quantity': 120, 'fertilizer_cost_per_kg': 30,
            'labor_days': 30, 'labor_cost_per_day': 500,
            'irrigation_cost': 4000,
            'pesticide_quantity': 4, 'pesticide_cost_per_liter': 700,
            'machinery_cost': 2500,
            'expected_yield': '15-25', 'market_price': 20
        },
        'cabbage': {
            'seed_quantity': 0.2, 'seed_cost_per_kg': 4000,
            'fertilizer_quantity': 180, 'fertilizer_cost_per_kg': 30,
            'labor_days': 25, 'labor_cost_per_day': 500,
            'irrigation_cost': 4500,
            'pesticide_quantity': 5, 'pesticide_cost_per_liter': 800,
            'machinery_cost': 3000,
            'expected_yield': '20-40', 'market_price': 10
        },
        'sunflower': {
            'seed_quantity': 6, 'seed_cost_per_kg': 300,
            'fertilizer_quantity': 100, 'fertilizer_cost_per_kg': 30,
            'labor_days': 15, 'labor_cost_per_day': 500,
            'irrigation_cost': 3000,
            'pesticide_quantity': 3, 'pesticide_cost_per_liter': 700,
            'machinery_cost': 2000,
            'expected_yield': '1-2', 'market_price': 40
        },
        'ragi': {
            'seed_quantity': 10, 'seed_cost_per_kg': 80,
            'fertilizer_quantity': 80, 'fertilizer_cost_per_kg': 25,
            'labor_days': 15, 'labor_cost_per_day': 500,
            'irrigation_cost': 2500,
            'pesticide_quantity': 2, 'pesticide_cost_per_liter': 600,
            'machinery_cost': 1500,
            'expected_yield': '1.5-2', 'market_price': 30
        },
        'cauliflower': {
            'seed_quantity': 0.1, 'seed_cost_per_kg': 5000,
            'fertilizer_quantity': 180, 'fertilizer_cost_per_kg': 30,
            'labor_days': 28, 'labor_cost_per_day': 500,
            'irrigation_cost': 4800,
            'pesticide_quantity': 6, 'pesticide_cost_per_liter': 800,
            'machinery_cost': 3200,
            'expected_yield': '15-25', 'market_price': 15
        },
        'millet': {
            'seed_quantity': 8, 'seed_cost_per_kg': 60,
            'fertilizer_quantity': 70, 'fertilizer_cost_per_kg': 25,
            'labor_days': 14, 'labor_cost_per_day': 500,
            'irrigation_cost': 2000,
            'pesticide_quantity': 2, 'pesticide_cost_per_liter': 500,
            'machinery_cost': 1200,
            'expected_yield': '1-1.5', 'market_price': 25
        },
        'groundnut': {
            'seed_quantity': 100, 'seed_cost_per_kg': 70,
            'fertilizer_quantity': 100, 'fertilizer_cost_per_kg': 30,
            'labor_days': 20, 'labor_cost_per_day': 500,
            'irrigation_cost': 4000,
            'pesticide_quantity': 4, 'pesticide_cost_per_liter': 700,
            'machinery_cost': 2800,
            'expected_yield': '1.5-2.5', 'market_price': 50
        },
        'mango': {
            'seed_quantity': 10, 'seed_cost_per_kg': 200,
            'fertilizer_quantity': 500, 'fertilizer_cost_per_kg': 20,
            'labor_days': 50, 'labor_cost_per_day': 500,
            'irrigation_cost': 7000,
            'pesticide_quantity': 10, 'pesticide_cost_per_liter': 1000,
            'machinery_cost': 5000,
            'expected_yield': '8-12', 'market_price': 70
        },
        'coffee': {
            'seed_quantity': 50, 'seed_cost_per_kg': 100,
            'fertilizer_quantity': 300, 'fertilizer_cost_per_kg': 25,
            'labor_days': 60, 'labor_cost_per_day': 500,
            'irrigation_cost': 8000,
            'pesticide_quantity': 12, 'pesticide_cost_per_liter': 900,
            'machinery_cost': 6000,
            'expected_yield': '0.5-1', 'market_price': 300
        },
        'watermelon': {
            'seed_quantity': 0.3, 'seed_cost_per_kg': 2500,
            'fertilizer_quantity': 150, 'fertilizer_cost_per_kg': 30,
            'labor_days': 30, 'labor_cost_per_day': 500,
            'irrigation_cost': 5000,
            'pesticide_quantity': 6, 'pesticide_cost_per_liter': 800,
            'machinery_cost': 3500,
            'expected_yield': '20-40', 'market_price': 15
        },
        'cucumber': {
            'seed_quantity': 0.2, 'seed_cost_per_kg': 3000,
            'fertilizer_quantity': 120, 'fertilizer_cost_per_kg': 30,
            'labor_days': 25, 'labor_cost_per_day': 500,
            'irrigation_cost': 4000,
            'pesticide_quantity': 5, 'pesticide_cost_per_liter': 700,
            'machinery_cost': 2800,
            'expected_yield': '10-20', 'market_price': 20
        },
        'pumpkin': {
            'seed_quantity': 0.5, 'seed_cost_per_kg': 1500,
            'fertilizer_quantity': 130, 'fertilizer_cost_per_kg': 30,
            'labor_days': 28, 'labor_cost_per_day': 500,
            'irrigation_cost': 4500,
            'pesticide_quantity': 6, 'pesticide_cost_per_liter': 750,
            'machinery_cost': 3000,
            'expected_yield': '15-30', 'market_price': 8
        },
        'banana': {
            'seed_quantity': 200, 'seed_cost_per_kg': 50,
            'fertilizer_quantity': 400, 'fertilizer_cost_per_kg': 25,
            'labor_days': 70, 'labor_cost_per_day': 500,
            'irrigation_cost': 9000,
            'pesticide_quantity': 15, 'pesticide_cost_per_liter': 800,
            'machinery_cost': 7000,
            'expected_yield': '30-60', 'market_price': 20
        },
        'grapes': {
            'seed_quantity': 100, 'seed_cost_per_kg': 80,
            'fertilizer_quantity': 300, 'fertilizer_cost_per_kg': 30,
            'labor_days': 50, 'labor_cost_per_day': 500,
            'irrigation_cost': 7000,
            'pesticide_quantity': 8, 'pesticide_cost_per_liter': 900,
            'machinery_cost': 5000,
            'expected_yield': '10-20', 'market_price': 60
        },
        'lady fingers': {
            'seed_quantity': 8, 'seed_cost_per_kg': 150,
            'fertilizer_quantity': 100, 'fertilizer_cost_per_kg': 30,
            'labor_days': 20, 'labor_cost_per_day': 500,
            'irrigation_cost': 3000,
            'pesticide_quantity': 4, 'pesticide_cost_per_liter': 600,
            'machinery_cost': 2000,
            'expected_yield': '8-15', 'market_price': 25
        },
        'capsicum': {
            'seed_quantity': 0.1, 'seed_cost_per_kg': 6000,
            'fertilizer_quantity': 150, 'fertilizer_cost_per_kg': 35,
            'labor_days': 35, 'labor_cost_per_day': 500,
            'irrigation_cost': 5500,
            'pesticide_quantity': 7, 'pesticide_cost_per_liter': 850,
            'machinery_cost': 3800,
            'expected_yield': '20-30', 'market_price': 40
        },
        'potato': {
            'seed_quantity': 1500, 'seed_cost_per_kg': 20,
            'fertilizer_quantity': 150, 'fertilizer_cost_per_kg': 30,
            'labor_days': 30, 'labor_cost_per_day': 500,
            'irrigation_cost': 4500,
            'pesticide_quantity': 5, 'pesticide_cost_per_liter': 700,
            'machinery_cost': 3000,
            'expected_yield': '20-30', 'market_price': 15
        },
        'radish': {
            'seed_quantity': 10, 'seed_cost_per_kg': 50,
            'fertilizer_quantity': 80, 'fertilizer_cost_per_kg': 25,
            'labor_days': 10, 'labor_cost_per_day': 500,
            'irrigation_cost': 2000,
            'pesticide_quantity': 2, 'pesticide_cost_per_liter': 500,
            'machinery_cost': 1000,
            'expected_yield': '10-15', 'market_price': 15
        },
        'brinjal': {
            'seed_quantity': 0.4, 'seed_cost_per_kg': 3500,
            'fertilizer_quantity': 120, 'fertilizer_cost_per_kg': 30,
            'labor_days': 30, 'labor_cost_per_day': 500,
            'irrigation_cost': 4000,
            'pesticide_quantity': 6, 'pesticide_cost_per_liter': 700,
            'machinery_cost': 2800,
            'expected_yield': '20-30', 'market_price': 25
        },
        'chilly': {
            'seed_quantity': 0.3, 'seed_cost_per_kg': 4000,
            'fertilizer_quantity': 100, 'fertilizer_cost_per_kg': 30,
            'labor_days': 35, 'labor_cost_per_day': 500,
            'irrigation_cost': 4500,
            'pesticide_quantity': 7, 'pesticide_cost_per_liter': 800,
            'machinery_cost': 3000,
            'expected_yield': '10-15', 'market_price': 60
        }
    }
    return jsonify(list(crop_data.keys()))

if __name__ == '__main__':
    app.run(debug=True)