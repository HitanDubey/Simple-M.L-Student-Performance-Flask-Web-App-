from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import joblib
import os
import sys

app = Flask(__name__)

print("=" * 60)
print("STUDENT PERFORMANCE PREDICTOR")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Current directory: {os.getcwd()}")
print("=" * 60)

def load_model_safely():
    """Load model with multiple fallback methods"""
    
    # Check if model file exists
    model_files = ['model_pickle.pkl', 'model_final.joblib', 'model_retrained.joblib']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"\n✅ Found model file: {model_file} ({os.path.getsize(model_file)} bytes)")
            
            # Try joblib first for .joblib files
            if model_file.endswith('.joblib'):
                try:
                    print(f"📦 Loading with joblib...")
                    model = joblib.load(model_file)
                    print(f"✅ SUCCESS with joblib!")
                    return model
                except Exception as e:
                    print(f"❌ Joblib failed: {e}")
            
            # Try pickle for .pkl files
            try:
                print(f"📦 Loading with pickle...")
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                print(f"✅ SUCCESS with pickle!")
                return model
            except Exception as e:
                print(f"❌ Pickle failed: {e}")
            
            # Try pickle with latin1 encoding
            try:
                print(f"📦 Loading with pickle (latin1)...")
                with open(model_file, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
                print(f"✅ SUCCESS with pickle (latin1)!")
                return model
            except Exception as e:
                print(f"❌ Pickle (latin1) failed: {e}")
    
    return None

# Load the model
model = load_model_safely()

if model is None:
    print("\n❌ CRITICAL: Could not load model with any method!")
    print("\nPlease run the retraining script first.")
else:
    print("\n✅ Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    
    # Try to get model info
    if hasattr(model, 'feature_names_in_'):
        print(f"Features: {model.feature_names_in_}")
    if hasattr(model, 'n_features_in_'):
        print(f"Number of features: {model.n_features_in_}")
    if hasattr(model, 'coef_'):
        print(f"Coefficients shape: {model.coef_.shape}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    try:
        # Get form data
        hours_studied = float(request.form['hours_studied'])
        previous_scores = float(request.form['previous_scores'])
        extracurricular = request.form['extracurricular']
        sleep_hours = float(request.form['sleep_hours'])
        papers_practiced = float(request.form['papers_practiced'])
        
        # Encode extracurricular
        extracurricular_encoded = 1 if extracurricular.lower() == 'yes' else 0
        
        # Create feature array
        features = np.array([[hours_studied, previous_scores, extracurricular_encoded, 
                              sleep_hours, papers_practiced]], dtype=np.float64)
        
        print(f"\nMaking prediction with features: {features[0]}")
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)
        
        print(f"Prediction result: {prediction}")
        
        # Determine category
        if prediction >= 80:
            category = "Excellent"
            color = "#4CAF50"
            icon = "🌟"
            message = "Outstanding performance! Keep up the great work!"
        elif prediction >= 60:
            category = "Good"
            color = "#2196F3"
            icon = "👍"
            message = "Good performance! You're on the right track."
        elif prediction >= 40:
            category = "Average"
            color = "#FF9800"
            icon = "⚡"
            message = "Average performance. There's room for improvement."
        else:
            category = "Needs Improvement"
            color = "#f44336"
            icon = "📚"
            message = "Needs improvement. Don't give up!"
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'category': category,
            'color': color,
            'icon': icon,
            'message': message,
            'features': {
                'Hours Studied': f"{hours_studied} hrs/day",
                'Previous Scores': f"{previous_scores}%",
                'Extracurricular': extracurricular,
                'Sleep Hours': f"{sleep_hours} hrs/day",
                'Papers Practiced': str(papers_practiced)
            }
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if model else 'degraded',
        'model_loaded': model is not None,
        'numpy_version': np.__version__,
        'python_version': sys.version
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting Flask server...")
    print("Access the app at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)