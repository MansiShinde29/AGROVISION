import os
import numpy as np
from PIL import Image
import joblib

def extract_features(image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))  # Ensure the size is the same as during training
    image = np.array(image)
    image = image.flatten()  # Convert to a 1D array
    return image

def predict_disease(image_path):
    # Load the trained model
    clf = joblib.load('home/models/disease_model.pkl')
    
    # Extract features from the input image
    features = extract_features(image_path)
    
    # Make prediction
    prediction = clf.predict([features])[0]
    return prediction

if __name__ == "__main__":
    test_image_path = "home/static/disease_detection"
    
    if os.path.exists(test_image_path):
        result = predict_disease(test_image_path)
        print(f"✅ Predicted Disease: {result}")
    else:
        print(f"🚨 Test image not found at: {test_image_path}")
