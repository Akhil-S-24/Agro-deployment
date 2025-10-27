# Agro-deployment
A Flask-based smart agriculture platform that integrates crop recommendation, rainfall prediction, disease detection, and cost estimation using ML and SQLite. It supports user authentication, farmer dashboards, image uploads, and detailed crop data for precision farming.

📝 Project Description :

The Smart Agriculture System is a web-based intelligent farming platform that helps farmers make data-driven decisions.
It includes modules such as Crop Recommendation, Rain Prediction, Crop Details, Disease Detection, Plant Health Monitoring, and Cost Estimation.
By analyzing soil nutrients (N, P, K), temperature, and humidity, it recommends the best crops and estimates farming costs for better productivity and profitability.

⚙️ Key Features

🌾 Crop Recommendation using soil and weather data

🌧️ Rain Prediction for better planning

🪴 Plant Health Monitoring

💰 Cost Estimator based on land area and crop type

🌿 Disease Detection through image-based input

📊 Crop Details Dashboard

🔒 User-friendly interface built with Flask and HTML/CSS

🧰 Technologies Used

Frontend: HTML, CSS, JavaScript

Backend: Python (Flask Framework)

Machine Learning Models: Crop & Disease Prediction

Database (optional): MongoDB / SQLite

Tools: Jupyter Notebook, Flask Server

📁 Folder Structure (recommended before upload)
Smart-Agriculture-System/
│
├── app.py
├── requirements.txt
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
│   ├── dashboard.html
│   ├── crop_recommendation.html
│   ├── cost_estimator.html
│   └── index.html
├── models/
│   ├── crop_model.pkl
│   └── rain_model.pkl
└── README.md

🚀 How to Run
# Clone the repository
git clone https://github.com/<your-username>/Smart-Agriculture-System.git
cd Smart-Agriculture-System

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py


Then open http://127.0.0.1:5000/
 in your browser.
