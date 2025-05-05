import pandas as pd
import re
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Feature groups ---
categorical_features = ['neighbourhood', 'room_type']
numerical_features = ['latitude', 'longitude', 'bedrooms', 'accommodates', 'bathrooms', 'beds']

# --- Pipelines ---
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# --- Column transformer ---
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_pipeline, categorical_features),
    ('num', numerical_pipeline, numerical_features),
])

# --- Final pipeline ---
model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- Data loading ---
def load_data():
    return pd.read_csv("listings.csv.gz", compression='gzip')

# --- Amenities parsing ---
def parse_amenities(amenities):
    if pd.isna(amenities):
        return []
    try:
        return re.findall(r'"([^"]*)', amenities)
    except:
        return []

# --- Data cleaning ---
def clean_data(df):
    df_clean = df.copy()
    df_clean["price"] = df_clean["price"].replace("[\$,]", "", regex=True).astype(float)

    # Fill some essential numeric columns
    numeric_cols = ["bathrooms", "bedrooms", "beds", "accommodates", "minimum_nights"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Parse amenities into lists
    df_clean["amenities_list"] = df_clean["amenities"].apply(parse_amenities)

    # Add top amenities as binary columns
    common_amenities = [
        "Wifi", "Kitchen", "Heating", "Air conditioning", "Washer", "Dryer", 
        "TV", "Hair dryer", "Iron", "Smoke alarm", "Fire extinguisher", 
        "Dishwasher", "Refrigerator", "Microwave", "Oven", "Stove", "Coffee maker",
        "Hot water", "Elevator", "Free parking"
    ]
    for amenity in common_amenities:
        df_clean[f"has_{amenity.lower().replace(' ', '_')}"] = df_clean["amenities_list"].apply(lambda x: 1 if amenity in x else 0)

    # Map neighbourhood field
    if "neighbourhood_cleansed" in df_clean.columns:
        df_clean["neighbourhood"] = df_clean["neighbourhood_cleansed"]

    return df_clean

# --- Run everything ---
if __name__ == "__main__":
    # Load and clean data
    df = load_data()
    df_clean = clean_data(df)
    
    # Drop rows with missing target
    df_clean = df_clean[df_clean['price'].notna()]
    
    # Select pipeline-compatible features and target
    X = df_clean[categorical_features + numerical_features]
    y = df_clean["price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline
    model_pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
