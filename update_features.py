#!/usr/bin/env python3
"""
Update feature list to include new factor columns.
Run this after integrating new factors to update tokoname_p1_features.pkl
"""
import joblib

# Load existing features
old_features = joblib.load("tokoname_p1_features.pkl")
print("Old features:")
print(old_features)

# Add new factor columns
new_features = old_features + [
    "course_bias",
    "weather_suitability",
    "field_imbalance",
]

print("\nNew features:")
print(new_features)

# Save updated feature list
joblib.dump(new_features, "tokoname_p1_features.pkl")
print("\nSaved updated feature list to tokoname_p1_features.pkl")
print(f"Total features: {len(new_features)}")
