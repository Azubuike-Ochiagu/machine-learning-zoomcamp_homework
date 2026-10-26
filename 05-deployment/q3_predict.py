import pickle
from pathlib import Path

from model import normalize_record

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

model_path = Path(__file__).with_name("pipeline.bin")

with model_path.open("rb") as model_file:
    pipeline = pickle.load(model_file)

record = normalize_record(lead)
probability = float(pipeline.predict_proba([record])[0, 1])

print("Conversion probability:", probability)
print(f"Q3 answer: {probability:.3f}")