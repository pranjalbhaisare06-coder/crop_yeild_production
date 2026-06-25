
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and preprocessor (updated to load random_forest_model.pkl)
model = joblib.load('random_forest_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():
    return 'Crop Yield Prediction API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        # Convert input JSON to DataFrame
        df = pd.DataFrame(json_)
        
        # Preprocess the input data
        processed_data = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        return jsonify({'prediction': list(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
    
    # For deployment in Colab, you might use ngrok or similar services
    # from flask_ngrok import run_with_ngrok
    # run_with_ngrok(app) # Start ngrok when app is run
    # app.run()
