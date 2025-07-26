from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import pandas as pd
import joblib
import numpy as np
from integmodel import load_model, predict_image
from prediction_code import predict  # Growth analysis function
from pest_detection import detect_pests  # Pest detection function
from weed_classifier import load_model_and_mapping, predict_weed  # Weed classification functions
from yield_prediction import predict_yield, predict_rice_yield
from weatheralerts import IndianLocations, WeatherMonitor
import openai
import os
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key'

locations = IndianLocations()
weather_monitor = WeatherMonitor(locations)

# Simulate a user database (replace with a real database in production)
users = {}  # Format: email: password

# Agricultural knowledge base for chatbot
AGRI_KNOWLEDGE = {
    "crops": {
        "rice": {
            "planting_season": "Kharif (June-July) and Rabi (November-December)",
            "water_requirement": "High - 1200-1800mm annually",
            "soil_type": "Clay loam with good water retention",
            "common_diseases": ["Blast", "Brown spot", "Bacterial blight"],
            "fertilizers": "NPK 120:60:40 kg/ha",
            "pests": ["Stem borer", "Brown planthopper", "Rice hispa"]
        },
        "wheat": {
            "planting_season": "Rabi (October-December)",
            "water_requirement": "Medium - 450-650mm",
            "soil_type": "Well-drained loamy soil",
            "common_diseases": ["Rust", "Smut", "Bunt"],
            "fertilizers": "NPK 120:60:40 kg/ha",
            "pests": ["Aphids", "Army worm", "Termites"]
        },
        "cotton": {
            "planting_season": "Kharif (April-May)",
            "water_requirement": "Medium to high - 700-1300mm",
            "soil_type": "Deep, well-drained black cotton soil",
            "common_diseases": ["Bollworm", "Whitefly", "Aphids"],
            "fertilizers": "NPK 120:60:60 kg/ha",
            "pests": ["Bollworm", "Jassids", "Thrips"]
        },
        "tomato": {
            "planting_season": "Year-round in greenhouse, Rabi season in open field",
            "water_requirement": "Medium - 400-600mm",
            "soil_type": "Well-drained loamy soil with pH 6.0-6.8",
            "common_diseases": ["Blight", "Wilt", "Leaf curl"],
            "fertilizers": "NPK 100:50:50 kg/ha",
            "pests": ["Whitefly", "Aphids", "Fruit borer"]
        },
        "maize": {
            "planting_season": "Kharif (June-July) and Rabi (October-November)",
            "water_requirement": "Medium - 500-800mm",
            "soil_type": "Well-drained fertile loamy soil",
            "common_diseases": ["Blight", "Rust", "Downy mildew"],
            "fertilizers": "NPK 150:75:60 kg/ha",
            "pests": ["Stem borer", "Fall armyworm", "Aphids"]
        }
    },
    "fertilizers": {
        "nitrogen": "Promotes leaf growth and green color. Sources: Urea, Ammonium sulphate",
        "phosphorus": "Essential for root development and flowering. Sources: DAP, SSP",
        "potassium": "Improves disease resistance and fruit quality. Sources: MOP, SOP",
        "organic": "Compost, FYM, vermicompost improve soil health naturally",
        "micronutrients": "Zinc, Iron, Boron are essential in small quantities"
    },
    "pest_control": {
        "integrated_pest_management": "Combine biological, cultural, and chemical methods",
        "biological_control": "Use beneficial insects like ladybugs, parasitic wasps",
        "cultural_practices": "Crop rotation, intercropping, proper spacing",
        "organic_pesticides": "Neem oil, BT spray, pheromone traps"
    },
    "general_tips": [
        "Soil testing should be done every 2-3 years",
        "Crop rotation helps maintain soil fertility",
        "Integrated pest management reduces chemical usage",
        "Weather monitoring is crucial for farming decisions",
        "Proper drainage prevents waterlogging",
        "Mulching conserves soil moisture"
    ]
}

class AgriChatbot:
    def __init__(self):
        # Try to use OpenAI if available, otherwise use rule-based responses
        self.use_openai = bool(os.getenv('OPENAI_API_KEY'))
        if self.use_openai:
            openai.api_key = os.getenv('OPENAI_API_KEY')
    
    def is_agriculture_related(self, message):
        """Check if the message is related to agriculture"""
        agri_keywords = [
            'crop', 'farming', 'agriculture', 'soil', 'fertilizer', 'pesticide',
            'irrigation', 'harvest', 'seed', 'plant', 'cultivation', 'farm',
            'rice', 'wheat', 'cotton', 'corn', 'maize', 'tomato', 'vegetable', 'fruit',
            'weather', 'season', 'disease', 'pest', 'yield', 'organic', 'growth',
            'weed', 'herbicide', 'nitrogen', 'phosphorus', 'potassium', 'compost',
            'rotation', 'intercropping', 'greenhouse', 'kharif', 'rabi', 'monsoon'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in agri_keywords)
    
    def get_rule_based_response(self, message):
        """Generate response using rule-based system"""
        message_lower = message.lower()
        
        # Greetings
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
            return "Hello! I'm your agricultural assistant. I can help you with farming questions, crop information, soil management, pest control, fertilizers, and more. What would you like to know?"
        
        # Crop-specific queries
        for crop_name, crop_info in AGRI_KNOWLEDGE['crops'].items():
            if crop_name in message_lower:
                if 'season' in message_lower or 'when' in message_lower or 'time' in message_lower:
                    return f"For {crop_name.title()}: Best planting season is {crop_info['planting_season']}"
                elif 'water' in message_lower or 'irrigation' in message_lower:
                    return f"For {crop_name.title()}: Water requirement is {crop_info['water_requirement']}"
                elif 'soil' in message_lower:
                    return f"For {crop_name.title()}: Suitable soil type is {crop_info['soil_type']}"
                elif 'disease' in message_lower:
                    diseases = ', '.join(crop_info['common_diseases'])
                    return f"Common diseases for {crop_name.title()}: {diseases}"
                elif 'pest' in message_lower and 'pests' in crop_info:
                    pests = ', '.join(crop_info['pests'])
                    return f"Common pests for {crop_name.title()}: {pests}"
                elif 'fertilizer' in message_lower:
                    return f"For {crop_name.title()}: Recommended fertilizer is {crop_info['fertilizers']}"
                else:
                    return f"Here's information about {crop_name.title()}:\n" + \
                           f"Planting season: {crop_info['planting_season']}\n" + \
                           f"Water requirement: {crop_info['water_requirement']}\n" + \
                           f"Soil type: {crop_info['soil_type']}"
        
        # Fertilizer queries
        if 'fertilizer' in message_lower or 'nutrient' in message_lower:
            if 'nitrogen' in message_lower:
                return f"Nitrogen: {AGRI_KNOWLEDGE['fertilizers']['nitrogen']}"
            elif 'phosphorus' in message_lower:
                return f"Phosphorus: {AGRI_KNOWLEDGE['fertilizers']['phosphorus']}"
            elif 'potassium' in message_lower:
                return f"Potassium: {AGRI_KNOWLEDGE['fertilizers']['potassium']}"
            elif 'organic' in message_lower:
                return f"Organic fertilizers: {AGRI_KNOWLEDGE['fertilizers']['organic']}"
            else:
                return "The main fertilizers are NPK (Nitrogen, Phosphorus, Potassium). Nitrogen promotes leaf growth, Phosphorus helps root development, and Potassium improves disease resistance."
        
        # Pest control queries
        if 'pest control' in message_lower or 'pesticide' in message_lower:
            if 'integrated' in message_lower or 'ipm' in message_lower:
                return f"Integrated Pest Management: {AGRI_KNOWLEDGE['pest_control']['integrated_pest_management']}"
            elif 'biological' in message_lower:
                return f"Biological control: {AGRI_KNOWLEDGE['pest_control']['biological_control']}"
            elif 'organic' in message_lower:
                return f"Organic pesticides: {AGRI_KNOWLEDGE['pest_control']['organic_pesticides']}"
            else:
                return "For effective pest control, use Integrated Pest Management (IPM) combining biological, cultural, and chemical methods. This reduces chemical usage and is environmentally friendly."
        
        # General farming queries
        if 'soil test' in message_lower:
            return "Soil testing should be done every 2-3 years to check pH, nutrient levels, and organic matter content. This helps in making informed fertilizer decisions."
        
        if 'crop rotation' in message_lower:
            return "Crop rotation is beneficial for maintaining soil fertility, breaking pest cycles, and improving yields. Common rotations include cereal-legume-cereal patterns."
        
        if 'organic farming' in message_lower:
            return "Organic farming uses natural methods like compost, bio-fertilizers, and integrated pest management. It's sustainable but requires careful planning and patience."
        
        if 'irrigation' in message_lower:
            return "Proper irrigation is crucial for crop growth. Methods include drip irrigation (water-efficient), sprinkler irrigation (good for large areas), and flood irrigation (traditional method)."
        
        if 'weather' in message_lower:
            return "Weather monitoring helps in planning farming activities. Monitor temperature, rainfall, humidity, and wind speed for better crop management decisions."
        
        if 'yield' in message_lower:
            return "To increase crop yield: use quality seeds, proper fertilization, adequate irrigation, pest control, and follow recommended spacing and planting time."
        
        # Seasonal queries
        if 'kharif' in message_lower:
            return "Kharif season (June-October): Crops grown during monsoon. Examples: Rice, Cotton, Sugarcane, Maize. Requires good drainage and pest management."
        
        if 'rabi' in message_lower:
            return "Rabi season (November-April): Winter crops. Examples: Wheat, Barley, Peas, Gram. Requires irrigation as there's less rainfall."
        
        # Default agriculture response
        return "I can help you with information about crops (rice, wheat, cotton, tomato, maize), soil management, fertilizers, irrigation, pest control, and general farming practices. Please ask me a specific question!"
    
    def get_openai_response(self, message):
        """Generate response using OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert agricultural assistant specialized in Indian farming practices. Answer only agriculture-related questions with practical, accurate information relevant to Indian farmers. If the question is not related to agriculture, politely decline and redirect to agricultural topics. Keep responses concise and actionable."
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self.get_rule_based_response(message)
    
    def get_response(self, message):
        """Main method to get chatbot response"""
        if not self.is_agriculture_related(message):
            return "I'm specialized in agricultural topics. Please ask me questions about farming, crops, soil, irrigation, fertilizers, pest control, or other agriculture-related subjects."
        
        if self.use_openai:
            return self.get_openai_response(message)
        else:
            return self.get_rule_based_response(message)

# Initialize chatbot
chatbot = AgriChatbot()

@app.route('/', methods=['GET', 'POST'])
def register_and_login():
    """Single page for registration and login."""
    if request.method == 'POST':
        # Fetch form data
        name = request.form.get('name')
        mobile = request.form.get('mobile')
        email = request.form['email']
        password = request.form['password']
        city = request.form.get('city')

        # Handle registration
        if name and mobile and city:
            if email not in users:
                users[email] = {'password': password, 'name': name, 'mobile': mobile, 'city': city}
                session['user'] = email
                session['location'] = city
                return redirect(url_for('index'))
            return "User already exists. Please log in!"

        # Handle login
        if email in users and users[email]['password'] == password:
            session['user'] = email
            session['location'] = users[email]['city']
            return redirect(url_for('index'))
        return "Invalid credentials or registration required!"

    # Fetch cities from dataset
    cities = [(city.title(), info['state']) for city, info in locations.cities.items()]
    return render_template('register_and_login.html', cities=cities)

RESULTS_TEXT_FILE = 'static/analysis_results.txt'

def read_results_from_text_file():
    """Read analysis results from the text file."""
    if not os.path.exists(RESULTS_TEXT_FILE):
        return []

    results = []
    with open(RESULTS_TEXT_FILE, 'r') as file:
        current_result = {}
        for line in file:
            line = line.strip()
            if line.startswith("Analysis"):
                if current_result:
                    results.append(current_result)
                current_result = {"ID": len(results) + 1}
            elif line.startswith("Growth Health Analysis:"):
                current_result["Growth Health Analysis"] = line.split(":", 1)[1].strip()
            elif line.startswith("Pest Detection:"):
                current_result["Pest Detection"] = line.split(":", 1)[1].strip()
            elif line.startswith("Weed Severity Classification:"):
                current_result["Weed Severity Classification"] = line.split(":", 1)[1].strip()
        if current_result:
            results.append(current_result)
    return results

@app.route('/dashboard')
def dashboard():
    """Display saved analysis results in a table format."""
    analysis_results = read_results_from_text_file()
    return render_template('dashboard.html', analysis_results=analysis_results)

@app.route('/index')
def index():
    """Main index page displaying weather and analysis options."""
    if 'user' not in session or 'location' not in session:
        return redirect(url_for('register_and_login'))
    
    location = session['location']
    city_info = locations.search_location(location.lower())[1]

    # Fetch weather data
    weather_data = weather_monitor.get_weather_data(*city_info['coords'])
    if not weather_data or 'current' not in weather_data:
        weather_data = {
            'current': {
                'temperature_2m': 'N/A',
                'relative_humidity_2m': 'N/A',
                'wind_speed_10m': 'N/A'
            }
        }
        alerts = ["Weather data unavailable."]
        recommendations = ["Please check back later for recommendations."]
    else:
        # Generate alerts and recommendations
        alerts = weather_monitor.generate_alerts(weather_data)
        recommendations = weather_monitor.generate_recommendations(weather_data)

    return render_template(
        'index.html',
        location=location,
        weather=weather_data,
        alerts=alerts,
        recommendations=recommendations,
    )

@app.route('/logout')
def logout():
    """Logout route."""
    session.clear()
    return redirect(url_for('register_and_login'))

# Load Weed Severity Classification Model
WEED_MODEL_PATH = r'C:\Users\sivak\nit\project\models\best_model.pth'
WEED_SPLIT_INFO_PATH = r'C:\Users\sivak\nit\project\models\split_info.json'

# Load encoders and models
state_encoder = joblib.load(r'C:\Users\sivak\nit\project\models\state_encoder.pkl')
district_encoder = joblib.load(r'C:\Users\sivak\nit\project\models\district_encoder.pkl')
season_encoder = joblib.load(r'C:\Users\sivak\nit\project\models\season_encoder.pkl')

# Ensure the model and mappings are loaded correctly
try:
    weed_model, weed_idx_to_label, weed_device = load_model_and_mapping(WEED_MODEL_PATH, WEED_SPLIT_INFO_PATH)
except Exception as e:
    print(f"Error loading weed model: {e}")
    weed_model, weed_idx_to_label, weed_device = None, None, None

model = load_model(r'C:\Users\sivak\nit\project\best_model_checkpoint5.pth')

def analyze_image(image_path):
    """Run integmodel.py first to determine which analysis to perform."""
    class_names = ['growth', 'pest', 'weed']
    predicted_class = predict_image(model, image_path)
    predicted_label = class_names[predicted_class]
    
    print(f"[INFO] Integmodel Prediction: {predicted_label}")
    return predicted_label

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        image_path = os.path.join('static/uploads', image.filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)

        # Determine which model to run
        predicted_label = analyze_image(image_path)

        # Default results
        growth_results = {"Predicted Label": "No results", "Confidence": "No results"}
        pest_results = {"No pests detected": ["No results"]}
        weed_results = {"top_prediction": "No results", "all_predictions": []}

        # Default recommendations as None
        fertilizer_recommendation = None
        herbicide_recommendation = None
        pesticide_recommendation = None

        # Main classification
        if predicted_label == "growth":
            predicted_label, growth_confidence = predict(image_path)
            growth_results = {
                "Predicted Label": predicted_label,
                "Confidence": f"{growth_confidence:.2f}%"
            }
            fertilizer_recommendation = "Apply appropriate fertilizer for crop growth support."

        elif predicted_label == "pest":
            pest_results = detect_pests(image_path) or {"No pests detected": ["No results"]}
            pesticide_recommendation = "Apply suitable pesticide to prevent pest damage."

        elif predicted_label == "weed":
            weed_data = predict_weed(image_path, weed_model, weed_idx_to_label, weed_device)
            weed_results = {
                "top_prediction": weed_data["top_prediction"],
                "all_predictions": weed_data["all_predictions"]
            }
            herbicide_recommendation = "Apply herbicide to reduce weed impact on crops."

        # Print for debug
        print("--- Final Analysis Results ---")
        print("Growth:", growth_results)
        print("Pest:", pest_results)
        print("Weed:", weed_results)

        # Save results
        with open(RESULTS_TEXT_FILE, 'a') as file:
            file.write(f"Analysis {len(read_results_from_text_file()) + 1}:\n")
            file.write(f"Growth Health Analysis: {growth_results}\n")
            file.write(f"Pest Detection: {pest_results}\n")
            file.write(f"Weed Severity Classification: {weed_results}\n")
            file.write("-" * 50 + "\n")

        return render_template('analysis_results.html',
                               image_path=image_path,
                               growth_results=growth_results,
                               pest_results=pest_results,
                               weed_results=weed_results,
                               fertilizer_recommendation=fertilizer_recommendation,
                               herbicide_recommendation=herbicide_recommendation,
                               pesticide_recommendation=pesticide_recommendation)

    return render_template('upload_image.html')

@app.route('/yield_prediction', methods=['GET', 'POST'])
def yield_prediction():
    return render_template('yield_prediction.html')

@app.route('/yield_prediction_Production', methods=['GET', 'POST'])
def yield_prediction_state():
    # Get available options from encoders
    states = state_encoder.classes_.tolist()
    districts = district_encoder.classes_.tolist()
    seasons = season_encoder.classes_.tolist()

    result = None
    if request.method == 'POST':
        try:
            # Collect user input
            user_input = {
                'State': request.form.get('state', '').strip(),
                'District': request.form.get('district', '').strip(),
                'Year': int(request.form.get('year', 0)),
                'Season': request.form.get('season', '').strip(),
                'Area': float(request.form.get('area', 0)),
                'Production': float(request.form.get('production', 0))
            }

            # Predict yield
            predicted_yield, confidence = predict_yield(user_input)

            if predicted_yield is None:
                result = {"error": "Invalid input or prediction failed. Please try again."}
            else:
                result = {
                    "predicted_yield": f"{predicted_yield:.2f} Tonne/Hectare",
                    "confidence_interval": f"[{confidence[0]:.2f}, {confidence[1]:.2f}] Tonne/Hectare"
                }
        except Exception as e:
            result = {"error": f"An error occurred: {e}"}

    return render_template(
        'yield_prediction_state.html',
        states=states,
        districts=districts,
        seasons=seasons,
        result=result
    )

@app.route('/yield_prediction_Fertilizer', methods=['GET', 'POST'])
def yield_prediction_fertilizer():
    result = None
    if request.method == 'POST':
        try:
            predicted_yield = predict_rice_yield(
                float(request.form.get('nitrogen', 0)),
                float(request.form.get('phosphorus', 0)),
                float(request.form.get('potassium', 0))
            )
            result = {"predicted_yield": f"{predicted_yield:.2f} Tonne/Hectare"}
        except Exception as e:
            result = {"error": f"An error occurred: {e}"}

    return render_template('yield_prediction_fertilizer.html', result=result)

# Updated Chatbot Route - No more Hugging Face issues!
@app.route('/chatbot', methods=['POST'])
def chatbot_route():
    """Handle chatbot requests with improved agricultural assistant."""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': 'Please enter a valid message.'})
        
        # Get response from improved chatbot
        response = chatbot.get_response(user_message)
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error in chatbot endpoint: {e}")
        return jsonify({'response': 'I apologize, but I encountered an error. Please try again with a different question.'})

# Add a route to serve the chatbot interface (optional)
@app.route('/chat')
def chat_interface():
    """Serve chatbot interface page."""
    chat_html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agricultural Assistant</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f0f8ff;
            }
            .chat-container {
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 20px;
                background-color: white;
                height: 400px;
                overflow-y: auto;
                margin-bottom: 20px;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
            }
            .user-message {
                background-color: #e3f2fd;
                text-align: right;
            }
            .bot-message {
                background-color: #f1f8e9;
                text-align: left;
                white-space: pre-line;
            }
            .input-container {
                display: flex;
                gap: 10px;
            }
            input[type="text"] {
                flex: 1;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            .header {
                text-align: center;
                color: #2e7d32;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌾 Agricultural Assistant 🌾</h1>
            <p>Ask me about farming, crops, soil, irrigation, fertilizers, pest control, and more!</p>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message bot-message">
                Hello! I'm your agricultural assistant. I can help you with:
                • Crop information (rice, wheat, cotton, tomato, maize)
                • Fertilizers and soil management
                • Pest control and disease management
                • Irrigation and weather guidance
                • Seasonal farming advice
                
                What would you like to know?
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="userInput" placeholder="Ask me about agriculture..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>

        <script>
            function sendMessage() {
                const input = document.getElementById('userInput');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message to chat
                addMessage(message, 'user-message');
                input.value = '';
                
                // Send to backend
                fetch('/chatbot', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({message: message})
                })
                .then(response => response.json())
                .then(data => {
                    addMessage(data.response, 'bot-message');
                })
                .catch(error => {
                    addMessage('Sorry, I encountered an error. Please try again.', 'bot-message');
                });
            }
            
            function addMessage(message, className) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${className}`;
                messageDiv.textContent = message;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        </script>
    </body>
    </html>
    '''
    return chat_html

if __name__ == '__main__':
    print("Starting Agricultural Application with Improved Chatbot...")
    print("Chatbot features:")
    print("✅ No token issues - works immediately")
    print("✅ Agriculture-focused knowledge base")
    print("✅ Fallback to OpenAI if API key provided")
    print("✅ Access chatbot at: http://localhost:5000/chat")
    app.run(debug=True)