import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib
import os

print("=" * 60)
print("RETRAINING MODEL WITH CURRENT ENVIRONMENT")
print("=" * 60)

# Check if CSV file exists
if not os.path.exists('StudentPerformance.csv'):
    print("❌ ERROR: StudentPerformance.csv not found!")
    print("Please make sure the CSV file is in the current directory.")
    exit(1)

# Load the data
print("\n📊 Loading data from StudentPerformance.csv...")
df = pd.read_csv('StudentPerformance.csv')
print(f"✅ Loaded {len(df)} records")

# Display first few rows
print("\nFirst 5 rows of data:")
print(df.head())

# Encode categorical variable
print("\n🔄 Encoding categorical variables...")
le = LabelEncoder()
df['Extracurricular_Encoded'] = le.fit_transform(df['Extracurricular_Activities'])
print(f"✅ Extracurricular encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Prepare features and target
feature_cols = ['Hours_Studied', 'Previous_Scores', 'Extracurricular_Encoded', 
                'Sleep_Hours', 'Sample_Question Papers_Practiced']
X = df[feature_cols]
y = df['Performance_Index']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train the model
print("\n🔄 Training Linear Regression model...")
model = LinearRegression()
model.fit(X, y)

# Calculate R² score
r2_score = model.score(X, y)
print(f"✅ Model trained successfully!")
print(f"📈 R² score: {r2_score:.4f}")

# Display coefficients
print("\n📊 Model Coefficients:")
for feature, coef in zip(feature_cols, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# Save the model in multiple formats
print("\n💾 Saving model...")

# Save as joblib (recommended)
joblib.dump(model, 'model_retrained.joblib')
print("✅ Saved as model_retrained.joblib")

# Save as pickle
with open('model_retrained.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Saved as model_retrained.pkl")

# Also save with a name that app.py will look for
joblib.dump(model, 'model_final.joblib')
print("✅ Saved as model_final.joblib")

# Save model info
with open('model_info.txt', 'w') as f:
    f.write("MODEL INFORMATION\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model Type: {type(model).__name__}\n")
    f.write(f"NumPy Version: {np.__version__}\n")
    f.write(f"R² Score: {r2_score:.4f}\n\n")
    f.write("Features:\n")
    for feature in feature_cols:
        f.write(f"  - {feature}\n")
    f.write("\nCoefficients:\n")
    for feature, coef in zip(feature_cols, model.coef_):
        f.write(f"  {feature}: {coef:.4f}\n")
    f.write(f"  Intercept: {model.intercept_:.4f}\n")

print("✅ Saved model_info.txt")

print("\n" + "=" * 60)
print("🎉 RETRAINING COMPLETE!")
print("=" * 60)
print("\nYou can now run: python app.py")