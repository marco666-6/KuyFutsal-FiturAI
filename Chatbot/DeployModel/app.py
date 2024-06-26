from flask import Flask, request, jsonify, json
from flask_cors import CORS
import joblib
import os
import re

app = Flask(__name__)

workdir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(workdir, 'Kuysal_chatbot_pipeline.pkl'))
CORS(app)

dataset = os.path.join(workdir, "intents.json")
# Load the intents
print(f"Loading intents from: {dataset}")
with open(dataset, 'r') as f:
    intents = json.load(f)

# Create a mapping from tags to responses
tag_to_response = {intent['tag']: intent['responses'][0] for intent in intents['intents']}

@app.route('/chatbot', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    if message:
        message = re.sub('[^a-zA-Z\'-]', ' ', message)  # Keep only alphabets and apostrophes
        message = message.lower()  # Convert to lowercase
        message = message.split()  # Split into words
        message = " ".join(message)  # Rejoin words to ensure clean spacing
        probabilities = model.predict_proba([message])[0]
        max_prob = max(probabilities)
        prediction = model.classes_[probabilities.argmax()]
        
        if max_prob < 0.2:
            response = "Sorry, I don't understand."
        else:
            response = tag_to_response.get(prediction, "Sorry, I don't understand.")
        
        return jsonify({'response': response})
    return jsonify({'error': 'No message provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)