from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your pre-trained model (replace 'model.pkl' with your actual model file)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    duration = float(request.form['duration'])
    protocol_type = int(request.form['protocol_type'])
    src_bytes = float(request.form['src_bytes'])
    dst_bytes = float(request.form['dst_bytes'])
    
    # Prepare input for the model
    input_data = np.array([[duration, protocol_type, src_bytes, dst_bytes]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Interpret prediction
    result = "Intrusion" if prediction[0] == 1 else "Not Intrusion"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
