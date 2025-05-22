# generate_dataset.py

import pandas as pd
import numpy as np
from tqdm import tqdm

np.random.seed(42)
n_students = 10000

def generate_student(i):
    gender = np.random.choice(['Male', 'Female'])
    age = np.random.randint(10, 21)
    class_level = np.random.choice(['JSS1', 'JSS2', 'JSS3', 'SS1', 'SS2', 'SS3'])
    attendance = np.clip(np.random.normal(85, 12), 30, 100)
    math = np.clip(np.random.normal(60, 16), 0, 100)
    english = np.clip(np.random.normal(62, 14), 0, 100)
    science = np.clip(np.random.normal(58, 15), 0, 100)
    study_hours = np.random.randint(0, 25)
    parental_support = np.random.choice(['Yes', 'No'], p=[0.65, 0.35])
    food_security = np.random.choice(['Yes', 'No'], p=[0.85, 0.15])

    avg_score = (math + english + science) / 3
    at_risk = 'Yes' if (avg_score < 45 or attendance < 60) else 'No'

    return {
        'student_id': f'STUD{i+1:05}',
        'gender': gender,
        'age': age,
        'class_level': class_level,
        'attendance_rate': round(attendance, 1),
        'math_score': round(math, 1),
        'english_score': round(english, 1),
        'science_score': round(science, 1),
        'study_hours': study_hours,
        'parental_support': parental_support,
        'food_security': food_security,
        'at_risk': at_risk
    }

# Generate and save
students = [generate_student(i) for i in tqdm(range(n_students))]
df = pd.DataFrame(students)

df.to_csv("data/nigerian_student_performance_10k.csv", index=False)
print("✅ Dataset saved at data/nigerian_student_performance_10k.csv")
