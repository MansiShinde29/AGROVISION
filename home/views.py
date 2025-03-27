import json
from django.shortcuts import render,redirect
import pickle
import numpy as np
import pandas as pd
import os
import cv2
from .train_disease_model import extract_features
import joblib
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.models import User
from django.http import HttpResponse
from .models import build_and_train_crop_model
# import openai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Contact
from django.contrib import messages


# Define the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct model directory
model_dir = os.path.join(BASE_DIR, "home", "models")

# Define model paths
crop_model_path = os.path.join(model_dir, "crop_models.pkl")
fertilizer_model_path = os.path.join(model_dir, "fertilizer_model.pkl")
disease_model_path = os.path.join(model_dir, "disease_model.pkl")
state_encoder_path = os.path.join(BASE_DIR, 'home', 'state_encoder.pkl')
city_encoder_path = os.path.join(BASE_DIR, 'home', 'city_encoder.pkl')

# Fertilizer recommendation dictionary
fertilizer_info = {
    "Urea": "Urea is a nitrogen-rich fertilizer used to enhance plant growth and increase crop yield.",
    "DAP": "DAP (Diammonium Phosphate) provides nitrogen and phosphorus, promoting root development.",
    "MOP": "MOP (Muriate of Potash) provides potassium, essential for disease resistance and water regulation.",
    "14-35-14": "A balanced fertilizer suitable for general crop growth, providing essential N-P-K nutrients.",
    "28-28": "Specially formulated to provide equal nitrogen and phosphorus, promoting healthy growth.",
    "17-17-17": "A balanced fertilizer that supports overall plant health and productivity.",
    "20-20": "An effective fertilizer providing equal amounts of nitrogen and phosphorus.",
    "10-26-26": "Formulated to provide phosphorus and potassium for fruit and seed formation.",
    "19-19-19": "Balanced fertilizer providing nitrogen, phosphorus, and potassium for overall plant growth."
}

# Function to safely load models
def load_model(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"🚨 Model file missing or empty: {file_path}")
        return None

# Load trained models
crop_model = load_model(crop_model_path)
fertilizer_model = load_model(fertilizer_model_path)
disease_model = load_model(disease_model_path)
state_encoder = load_model(state_encoder_path)
city_encoder = load_model(city_encoder_path)


def train_crop_model(request):
    build_and_train_crop_model()
    return render(request, 'home/train_success.html')

def crop_recommend(request):
    if request.method == 'POST':
        nitrogen = float(request.POST['nitrogen'])
        phosphorus = float(request.POST['phosphorus'])
        potassium = float(request.POST['potassium'])
        temperature = float(request.POST['temperature'])
        humidity = float(request.POST['humidity'])
        ph = float(request.POST['ph'])
        rainfall = float(request.POST['rainfall'])
        
        # Add these two
        state = request.POST['state']
        city = request.POST['city']
        
        # Assuming you encode state and city into numeric values (You can use Label Encoding)
        # For now, we just convert them to ASCII sum as a placeholder
        state_value = sum([ord(char) for char in state.lower()])
        city_value = sum([ord(char) for char in city.lower()])

        features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, state_value, city_value]
        
        try:
            model = joblib.load('home/models/crop_models.pkl')
            prediction = model.predict([features])[0]
            return render(request, 'home/crop_recommend.html', {'prediction': prediction})
        except Exception as e:
            return HttpResponse(f"Error: {e}")
        
    return render(request, 'home/crop_recommend.html')

def fertilizer_recommendation(request):
    if request.method == "POST":
        try:
            nitrogen = float(request.POST['nitrogen'])
            phosphorus = float(request.POST['phosphorus'])
            potassium = float(request.POST['potassium'])

            # Define normal ranges for soil nutrients (these can be adjusted if needed)
            normal_nitrogen = (80, 120)
            normal_phosphorus = (40, 60)
            normal_potassium = (40, 80)

            # Suggestions based on nutrient levels
            suggestions = []

            if nitrogen < normal_nitrogen[0]:
                suggestions.append("The nitrogen level of your soil is low. Consider using Urea or other nitrogen-rich fertilizers.")
            elif nitrogen > normal_nitrogen[1]:
                suggestions.append("The nitrogen level of your soil is high. Avoid using nitrogen-rich fertilizers.")

            if phosphorus < normal_phosphorus[0]:
                suggestions.append("The phosphorus level of your soil is low. Consider using DAP or organic phosphates.")
            elif phosphorus > normal_phosphorus[1]:
                suggestions.append("The phosphorus level of your soil is high. Avoid using phosphorus-based fertilizers.")

            if potassium < normal_potassium[0]:
                suggestions.append("The potassium level of your soil is low. Try using MOP or natural potash sources like banana peels.")
            elif potassium > normal_potassium[1]:
                suggestions.append("The potassium level of your soil is high. Avoid using potassium-rich fertilizers.")

            # If no issues found, proceed with model prediction
            if not suggestions and fertilizer_model:
                prediction = fertilizer_model.predict([[nitrogen, phosphorus, potassium]])[0]
                fertilizer_description = fertilizer_info.get(prediction, "No information available for this fertilizer.")
            else:
                prediction = "Soil Condition Analysis Complete"
                fertilizer_description = "\n".join(suggestions)

        except Exception as e:
            prediction = f"🚨 Error processing request: {str(e)}"
            fertilizer_description = ""

        return render(request, "fertilizer.html", {
            "recommendation": prediction,
            "description": fertilizer_description
        })

    return render(request, "fertilizer.html")

# Load disease model
disease_model_path = os.path.join(BASE_DIR, "home", "models", "disease_model.pkl")

try:
    with open(disease_model_path, "rb") as f:
        disease_model = pickle.load(f)
        
    # Ensure the model file contains the correct objects
    if isinstance(disease_model, tuple) and len(disease_model) == 2:
        clf, label_encoder = disease_model  # Define clf and label_encoder properly
    else:
        raise ValueError("🚨 The model file does not contain the expected classifier and label encoder.")
except FileNotFoundError:
    raise FileNotFoundError(f"🚨 The model file 'disease_model.pkl' was not found at {disease_model_path}.")
except Exception as e:
    raise Exception(f"🚨 An error occurred while loading the model: {str(e)}")


def disease_detection(request):
    disease_info = {
        "tomato_rust": {
            "description": "Tomato rust causes yellow or brown spots on leaves and stems, resulting in poor fruit production.",
            "cure": "Remove infected plants, use fungicides, and plant resistant varieties."
        },
        "tomato_spot": {
            "description": "Tomato spot causes dark spots on fruits and leaves, leading to stunted growth.",
            "cure": "Remove infected plants and use appropriate fungicides."
        },
        "mulberry_rust": {
            "description": "Mulberry rust causes orange rust pustules on leaves, leading to premature leaf drop.",
            "cure": "Apply fungicides and remove infected leaves."
        },
        "Basil_downy mildew": {
            "description": "Basil downy mildew causes yellowing leaves, dark sporulation on the underside, and distorted growth.",
            "cure": "Use resistant varieties, ensure good air circulation, and apply fungicides if needed."
        },
        "banana_rust": {
            "description": "Banana rust causes reddish-brown streaks and spots on leaves, leading to reduced photosynthesis and fruit yield.",
            "cure": "Apply appropriate fungicides and practice proper field sanitation."
        },
        "Apple_black_rot": {
            "description": "Apple black rot results in dark, sunken lesions on fruit, leaves, and branches, often leading to fruit rot and tree decline.",
            "cure": "Prune infected areas, apply fungicides, and maintain orchard cleanliness."
        },

          "Tomato_Early_Blight": {
           "description": "Tomato early blight causes dark, concentric spots on older leaves, leading to yellowing and defoliation.",
           "cure": "Apply fungicides, remove infected leaves, and rotate crops."
        },
         "Tomato_Late_Blight": {
          "description": "Tomato late blight causes dark, water-soaked spots on leaves and fruits, rapidly killing the plant.",
          "cure": "Use resistant varieties, apply fungicides, and destroy infected plants."
        },
         "Tomato_Fusarium_Wilt": {
          "description": "Tomato fusarium wilt causes yellowing and wilting of leaves, starting from the lower leaves upwards.",
          "cure": "Use resistant varieties and ensure proper soil drainage."
        },
         "Orange_Greasy_Spot": {
          "description": "Orange greasy spot causes yellow blister-like lesions on leaves, resulting in premature leaf drop.",
           "cure": "Apply copper-based fungicides and ensure proper air circulation."
            },
        "Orange_Citrus_Canker": {
            "description": "Orange citrus canker causes lesions on leaves, stems, and fruits, resulting in reduced yield and quality.",
            "cure": "Remove infected parts, use copper-based fungicides, and plant resistant varieties."
        },
        "Orange_Melanose": {
            "description": "Orange melanose causes small, raised, dark spots on leaves and fruits, making them unsightly.",
            "cure": "Prune dead wood and apply appropriate fungicides."
        },
        "Orange_Black_Spot": {
            "description": "Orange black spot causes hard, raised, dark spots on fruit rinds, reducing fruit quality.",
            "cure": "Use fungicides, remove infected fruits, and avoid overhead irrigation."
        },
        "Basil_Root_Rot": {
            "description": "Basil root rot causes wilting and yellowing of leaves due to poor root health.",
            "cure": "Ensure proper drainage, avoid overwatering, and use healthy soil."
        },
        "Basil_Leaf_Blight": {
            "description": "Basil leaf blight causes dark spots and blighting on leaves, reducing leaf quality.",
            "cure": "Use fungicides and remove infected leaves."
        },
        "Basil_Fusarium_Wilt": {
            "description": "Basil fusarium wilt causes yellowing and wilting of leaves, leading to stunted growth.",
            "cure": "Use resistant varieties and ensure proper soil drainage."
        },
        "Basil_Bacterial_leaf_Spot": {
            "description": "Basil bacterial leaf spot causes black, water-soaked spots on leaves, leading to decay.",
            "cure": "Remove infected leaves and use copper-based bactericides."
        },
        "Corn_Southern_Blight": {
            "description": "Corn southern blight causes white, cottony fungal growth at the base of the plant, leading to decay.",
            "cure": "Remove infected plants, use fungicides, and practice crop rotation."
        },
        
        "Corn_Gray_Leaf_Spot": {
            "description": "Corn gray leaf spot causes narrow, rectangular lesions on leaves, leading to leaf necrosis.",
            "cure": "Use resistant varieties, apply fungicides, and practice crop rotation."
        },
        "Corn_Anthracnose_Leaf_Blight": {
            "description": "Corn anthracnose leaf blight causes irregular, water-soaked lesions on leaves, resulting in reduced yield.",
            "cure": "Use resistant varieties, apply fungicides, and remove infected debris."
        },
        "Corn_Southern_blight": {
            "description": "Corn southern blight causes white, cottony fungal growth at the base.",
            "cure": "Remove infected plants, use fungicides, and practice crop rotation."
            
        },
        "Corn_Northern_Corn_Blight": {
            "description": "Corn northern corn blight causes long, grayish lesions on leaves, resulting.",
            "cure": "Use resistant varieties, apply fungicides, and rotate crops."
        },

        "Grape_Powdery_Mildew": {
            "description": "Grape powdery mildew causes white, powdery fungal growth on leaves, stems, and fruit.",
            "cure": "Use fungicides and ensure proper air circulation."
        },
        "Grape_Gray_Mold": {
            "description": "Grape gray mold causes gray, fuzzy growth on fruits and leaves, especially in wet conditions.",
            "cure": "Apply fungicides and ensure proper spacing between plants."
        },
        "Grape_Downy_Mildew": {
            "description": "Grape downy mildew causes yellowish patches on leaves with white, downy growth on the underside.",
            "cure": "Use fungicides and ensure good air circulation."
        },
        "Grape_Anthracnose": {
            "description": "Grape anthracnose causes dark, sunken lesions on leaves, stems, and fruits, resulting in reduced yield.",
            "cure": "Prune infected parts and apply fungicides."
        },
        "Mulberry_Root_Rot": {
            "description": "Mulberry root rot causes root decay, wilting, and stunted growth.",
            "cure": "Improve soil drainage and apply fungicides."
        },
        "Mulberry_Powdery_Mildew": {
            "description": "Mulberry powdery mildew causes white, powdery growth on leaves, leading to leaf drop.",
            "cure": "Apply fungicides and improve air circulation."
        },
        "Mulberry_Leaf_Spot": {
            "description": "Mulberry leaf spot causes small, dark spots on leaves, leading to premature leaf drop.",
            "cure": "Remove infected leaves and apply fungicides."
        },
        "Mulberry_Stem_Canker": {
            "description": "Mulberry stem canker causes dark, sunken lesions on stems, reducing plant vigor.",
            "cure": "Prune infected stems and use appropriate fungicides."
        },
        "Banana_Streak_Virus": {
            "description": "Banana streak virus causes yellow streaks and mottling on leaves, leading to reduced growth.",
            "cure": "Use virus-free planting material and destroy infected plants."
        },
        "Banana_Panama_Disease": {
            "description": "Banana Panama disease causes wilting and yellowing of leaves due to fungal infection.",
            "cure": "Use resistant varieties and ensure proper soil drainage."
        },
        "Banana_Moko_Disease": {
            "description": "Banana moko disease causes wilting, leaf yellowing, and internal discoloration of fruits.",
            "cure": "Destroy infected plants and use disease-free planting material."
        },
        "Banana_Black_Sigatoka": {
            "description": "Banana black sigatoka causes dark streaks on leaves, reducing photosynthesis and yield.",
            "cure": "Use resistant varieties and apply fungicides regularly."
        },
        "Apple_Scab": {
            "description": "Apple scab causes olive-green or black spots on leaves and fruit, reducing fruit quality.",
            "cure": "Use fungicides and plant resistant varieties."
        },
        "Apple_Powdery_Mildew": {
            "description": "Apple powdery mildew causes white, powdery patches on leaves, shoots, and blossoms.",
            "cure": "Prune infected parts and apply fungicides."
        },
        "Apple_fire_Blight": {
            "description": "Apple fire blight causes blackened, wilted leaves and twigs, resembling burnt areas.",
            "cure": "Prune infected parts and apply copper-based bactericides."
        },
        "Apple_Cedar_Rust": {
            "description": "Apple cedar rust causes bright orange or yellow spots on leaves, reducing fruit yield.",
            "cure": "Use resistant varieties and apply fungicides as needed."
        },
        "Apple_scab": {
            "description": "Apple scab causes olive-green or black spots on leaves and fruit, reducing.",
            "cure": "Use fungicides and plant resistant varieties.",
        },
        
      "banana_healthy": {
            "description": "Plant is healthy.",
            "cure": "No cure needed."
        },
        "Apple_healthy": {
            "description": "Plant is healthy.",
            "cure": "No cure needed."
        },
        "mulberry_healthy": {
            "description": "Plant is healthy.",
            "cure": "No cure needed."
        },
        "tomato_healthy": {
            "description": "Plant is healthy.",
            "cure": "No cure needed."
        },
        "orange_healthy": {
            "description": "Plant is healthy.",
            "cure": "No cure needed."
        },
        "basil_healthy": {
            "description": "Plant is healthy.",
            "cure": "No cure needed."
        },
        "corn_healthy": {
            "description": "Plant is healthy.",
            "cure": "No cure needed."
        },
        "grape_healthy": {
            "description": "Plant is healthy.",
            "cure": "No cure needed."
        },
        "grape_rott": {
            "description": "Grape rot causes fruit decay, mold growth, and brown discoloration of the grapes, affecting yield and quality.",
            "cure": "Implement proper pruning, fungicide application, and avoid overhead watering."
        },
        "Orange_rust": {
            "description": "Orange rust causes orange powdery spots on leaves, which eventually distort and cause early leaf drop.",
            "cure": "Remove infected plants, apply fungicides, and ensure good air circulation."
        },
        "corn_rust": {
            "description": "Corn rust results in small, reddish-brown pustules on leaves, reducing photosynthesis and overall yield.",
            "cure": "Use resistant varieties, apply fungicides, and maintain crop rotation practices."
        },
        "unknown": {
            "description": "This disease is not recognized. Please consult an agricultural expert.",
            "cure": "No cure information available."
        }
    }

    context = {}

    if request.method == "POST":
        if 'leaf_image' not in request.FILES:
            context["disease"] = "🚨 No image uploaded!"
            return render(request, "disease.html", context)

        try:
            leaf_image = request.FILES['leaf_image']
            upload_path = os.path.join("home/static/disease_detection", leaf_image.name)

            with open(upload_path, "wb") as f:
                for chunk in leaf_image.chunks():
                    f.write(chunk)

            image = cv2.imread(upload_path)
            if image is None:
                raise ValueError("🚨 Unable to read the uploaded image.")

            features = extract_features(image).reshape(1, -1)
            prediction = clf.predict(features)[0]
            predicted_disease = label_encoder.inverse_transform([prediction])[0]

            # Get disease details
            disease_details = disease_info.get(predicted_disease, {
                "description": "Description not available.",
                "cure": "Cure information not available."
            })

            # Add the URL of the uploaded image to the context
            context = {
                "disease": predicted_disease,
                "description": disease_details["description"],
                "cure": disease_details["cure"],
                "uploaded_image_url": f"/static/disease_detection/{leaf_image.name}"
            }

        except Exception as e:
            context = {"disease": f"🚨 Error processing request: {str(e)}"}

    return render(request, "disease.html", context)
# Simple Rule-Based Chatbot Functionality

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '').lower()

        if 'crop' in user_message:
            response = "Please provide your soil type, temperature, and humidity."
        elif 'fertilizer' in user_message:
            response = "Let me know your crop type and soil condition."
        elif 'disease' in user_message:
            response = "Upload an image of the crop and I'll detect any diseases."
        else:
            response = "I'm sorry, I didn't understand that. Can you try asking differently?"
        
        return JsonResponse({'reply': response})
    
def chatbot(request):
     return render(request, 'home/chatbot.html')

# Home Page
def index(request):
    return render(request, "home/index.html")

# Signup Function
def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return redirect('signup')
        
        user = User.objects.create_user(username=username, password=password, email=email)
        user.save()
        messages.success(request, 'Signup successful. Please login.')
        login(request, user)  # Automatically log the user in after signup
        return redirect('index')  # Redirect to main website page

    return render(request, 'home/signup.html')

# Login Function
# home/views.py

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, "You have successfully logged in!")
            request.session['welcome_message'] = "Welcome to AgroVision!"
            return redirect('index')
        else:
            messages.error(request, "Invalid username or password.")
        return redirect('index')
    return render(request, 'home/login.html')

def logout_view(request):
    logout(request)
    messages.success(request, "You have successfully logged out!")
    return redirect('index')

#ContactUS
def index(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')  # Use 'message' as in the model

        if name and email and message:
            contact = Contact(name=name, email=email, message=message)  # Make sure this is 'message'
            contact.save()
            messages.success(request, "Your query has been submitted successfully!")
            return redirect('index')  # Adjust this URL name if needed

    return render(request, 'home/index.html')