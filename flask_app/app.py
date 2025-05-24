from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import json

app = Flask(__name__)
CORS(app)

# Load and preprocess the dataset
df = pd.read_csv('socialMediaImpact.csv')

@app.route('/')
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Invalid request data',
                'message': 'Message field is required'
            }), 400

        user_message = data['message']
        
        # Analyze sentiment
        sentiment = TextBlob(user_message).sentiment.polarity
        
        # Simple response logic based on sentiment
        if sentiment > 0:
            response = "That's great to hear! Let's continue with our assessment."
        elif sentiment < 0:
            response = "I understand this might be difficult. We're here to help."
        else:
            response = "Thank you for sharing. Let's explore this further."
        
        # Mock analysis results
        analysis = {
            'socialMediaScore': np.random.randint(60, 90),
            'anxietyScore': np.random.randint(40, 80),
            'depressionScore': np.random.randint(30, 70),
            'wellbeingScore': np.random.randint(50, 90)
        }
        
        return jsonify({
            'response': response,
            'analysis': analysis,
            'complete': False  # Set to True when assessment is complete
        })
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)