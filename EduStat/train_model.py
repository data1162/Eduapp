# train_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/nigerian_student_performance_10k.csv")

# Preprocessing
features = ['gender', 'age', 'class_level', 'attendance_rate',
            'math_score', 'english_score', 'science_score',
            'study_hours', 'parental_support', 'food_security']

X = df[features]
y = df['at_risk']

# Encode categorical features
label_cols = ['gender', 'class_level', 'parental_support', 'food_security']
encoders = {col: LabelEncoder().fit(X[col]) for col in label_cols}
for col, encoder in encoders.items():
    X[col] = encoder.transform(X[col])

# Encode target
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)  # Yes/No -> 1/0

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'model/student_model.pkl')
joblib.dump(encoders, 'model/encoders.pkl')
joblib.dump(y_encoder, 'model/target_encoder.pkl')

print("✅ Model and encoders saved in model/")
