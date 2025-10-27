# train_models.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Example for crop recommendation (you would use real agricultural data)
def train_crop_recommendation_model():
    # Mock data - in reality, use a proper dataset
    data = {
        'N': [20, 30, 40, 50, 60],
        'P': [10, 20, 30, 40, 50],
        'K': [5, 15, 25, 35, 45],
        'temperature': [20, 22, 24, 26, 28],
        'humidity': [60, 65, 70, 75, 80],
        'ph': [6.0, 6.5, 7.0, 7.5, 8.0],
        'rainfall': [100, 150, 200, 250, 300],
        'label': ['wheat', 'rice', 'corn', 'sugarcane', 'cotton']
    }
    
    df = pd.DataFrame(data)
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Save the model
    with open('crop_recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained with accuracy: {model.score(X_test, y_test):.2f}")

if __name__ == '__main__':
    train_crop_recommendation_model()
    # Add similar functions for other models