import pandas as pd
import re
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Feature groups
categorical_features = ['neighbourhood', 'room_type']
numerical_features = ['latitude', 'longitude', 'bedrooms', 'accommodates', 'bathrooms', 'beds']
amenity_features = [
        "Wifi", "Kitchen", "Heating", "Air conditioning", "Washer", "Dryer", 
        "TV", "Hair dryer", "Iron", "Smoke alarm", "Fire extinguisher", 
        "Dishwasher", "Refrigerator", "Microwave", "Oven", "Stove", "Coffee maker",
        "Hot water", "Elevator", "Free parking"
    ]
# Pipelines
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

# Full transformer
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_pipeline, categorical_features),
    ('num', numerical_pipeline, numerical_features),
])

# Final pipeline with a model
model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# load the data
def load_data():
    df = pd.read_csv("listings.csv.gz", compression='gzip')
    return df
df = load_data()
print(df)

# data cleaning
def clean_data(df):
    
    df_clean = df.copy()
    # convert price $ to numeric
    df_clean["price"] = df_clean["price"].replace("[\$,]", "", regex=True).astype(float)
    

    print(df_clean["price"].describe())

    # remove extreme outliers in price
    q1 = df_clean["price"].quantile(0.01)
    q3 = df_clean["price"].quantile(0.99)
    iqr = q3 - q1

    df_clean = df_clean[(df_clean["price"] >= max(0, q1 - 1.5 * iqr)) & (df_clean["price"] <= q3 + 1.5 * iqr)]
    print("\nPrice statistics after outlier removal:")
    print(df_clean["price"].describe())
    # handle missing values for numerical columns
    numeric_cols = ["bathrooms", "bedrooms", "beds", "accommodates", "minimum_nights"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # process amenities column (convert to list)
    df_clean["amenities_list"] = df_clean["amenities"].apply(parse_amenities)

    common_amenities = [
        "Wifi", "Kitchen", "Heating", "Air conditioning", "Washer", "Dryer", 
        "TV", "Hair dryer", "Iron", "Smoke alarm", "Fire extinguisher", 
        "Dishwasher", "Refrigerator", "Microwave", "Oven", "Stove", "Coffee maker",
        "Hot water", "Elevator", "Free parking"
    ]

    for amenity in common_amenities:
        df_clean[f"has_{amenity.lower().replace(' ', '_')}"] = df_clean["amenities_list"].apply(lambda x: 1 if amenity in x else 0)

    if "neighbourhood_cleansed" in df_clean.columns:
        df_clean["neighbourhood"] = df_clean["neighbourhood_cleansed"]

    # extract room features
    df_clean["is_entire_home"] = df_clean["room_type"].apply(lambda x:1 if x == "Entire home/apt" else 0)
    df_clean["is_private_room"] = df_clean["room_type"].apply(lambda x:1 if x == "Private room" else 0)
    df_clean["is_hotel_room"] = df_clean["room_type"].apply(lambda x:1 if x == "Hotel room" else 0)
    df_clean["is_shared_room"] = df_clean["room_type"].apply(lambda x:1 if x == "Shared room" else 0)

    # create title and description features
    if "name" in df_clean.columns:
        df_clean["name"] = df_clean["name"].fillna("")
        df_clean["title_word_count"] = df_clean["name"].fillna("").apply(lambda x:len(str(x).split()))
        df_clean["title_length"] = df_clean["name"].fillna("").apply(len)
    
    if "description" in df_clean.columns:
        df_clean["description"] = df_clean["description"].fillna("")
        df_clean["description_word_count"] = df_clean["description"].fillna("").apply(lambda x: len(str(x).split()))
        df_clean["description_length"] = df_clean["description"].apply(len)
    
    # add review score features
    review_score_cols = [
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication", "review_scores_location", 
        "review_scores_value"
    ]

    for col in review_score_cols:
        if col in df_clean.columns:
            col_mean = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(col_mean)

    # superhost feature
    if "host_is_superhost" in df_clean.columns:
        df_clean["host_is_superhost_num"] = df_clean["host_is_superhost"].apply(lambda x: 1 if x == "t" else 0)

    # check if instant bookable
    if "instant_bookable" in df_clean.columns:
        df_clean["instant_bookable_num"] = df_clean["instant_bookable"].apply(lambda x: 1 if x == "t" else 0)

    # add review count and frequency features
    if "number_of_reviews" in df_clean.columns:
        df_clean["number_of_reviews"] = df_clean["number_of_reviews"].fillna(0)
        df_clean["has_reviews"] = df_clean["number_of_reviews"].apply(lambda x: 1 if x > 0 else 0)

    if "reviews_per_month" in df_clean.columns:
        df_clean["reviews_per_monthn"] = df_clean["reviews_per_month"].fillna(0)

    # location features
    if "latitude" in df_clean.columns and "longitude" in df_clean.columns:
        # calc distance from city center
        city_center_lat = 59.3293
        city_center_lon = 18.0686

        df_clean["distance_to_center"] = np.sqrt(
            (df_clean["latitude"] - city_center_lat)**2 + (df_clean["longitude"] - city_center_lon)**2
        ) * 111 # aproximate conversation to kilometers

    return df_clean

def parse_amenities(amenities):
    if pd.isna(amenities):
        return []
    
    try:
        items = re.findall(r'"([^"]*)', amenities)
        return items
    except:
        try:
            cleaned_str = amenities.replace("\\", "")
            if cleaned_str.startswith("[") and cleaned_str.endswith("]"):
                return json.loads(clean_data)
    
        except:
            return []
    
    return []

# show the clean data
df_clean = clean_data(df)
print(df_clean["has_hot_water"])

def train_price_model(df_clean):
    
    features = [
        'bathrooms', 'bedrooms', 'beds', 'accommodates', 'minimum_nights',
        'is_entire_home', 'is_private_room',
    ]

    # amenity features
    amenity_cols = [col for col in df_clean.columns if col.startswith("has_")]
    features.extend(amenity_cols)

    # Add review score features if available
    review_score_cols = [
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication", "review_scores_location", 
        "review_scores_value"
    ]

    for col in review_score_cols:
        if col in df_clean.columns:
            features.append(col)
    
    # add host features
    host_features = ["host_is_superhost_num", "calculated_host_listings_count", "instant_bookable_num"]
    for feature in host_features:
        if feature in df_clean.columns:
            features.append(feature)
    
    # add review count feature
    review_features = ["number_of_reviews", "reviews_per_month", "has_reviews"]
    for feature in review_features:
        if feature in df_clean.columns:
            features.append(feature)
    
    df_features = df_clean[features].copy()

    # add neighborhood as dummy variables
    if "neighbourhood" in df_clean.columns:
        top_neighborhoods  = df_clean["neighbourhood"].value_counts().nlargest(30).index
        df_clean["neighbourhood_top"] = df_clean["neighbourhood"].apply(lambda x: x if x in top_neighborhoods else "Other")
        neighborhood_dummies = pd.get_dummies(df_clean["neighbourhood_top"], prefix="nbhd")
        df_model = pd.concat([df_features, neighborhood_dummies], axis=1)
    else:
        df_model = df_features.copy()

    # add property_type
    if "property_type" in df_clean.columns:
        # limit to top property types
        top_property_types = df_clean["property_type"].value_counts().nlargest(10).index
        df_clean["property_type_top"] = df_clean["property_type"].apply(lambda x: x if x in top_property_types else "Other")

        property_dummies = pd.get_dummies(df_clean["property_type_top"], prefix="prop")
        df_model = pd.concat([df_model, property_dummies], axis=1)


    
    # only numeric data
    df_model = df_model.select_dtypes(include=["number"])

    combined_df = pd.concat([df_model, df_clean["price"]], axis=1)
    combined_df = combined_df.dropna()

    x = combined_df.drop("price", axis=1)
    y = combined_df["price"]


    # split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

    # scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    model.fit(x_train_scaled, y_train)

    # eval the model
    y_pred = model.predict(x_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # get feature importance
    feature_importance = pd.DataFrame({
        "feature": x.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    def predict_price(x_new, model=model, scaler=scaler, feature_names=x.columns):
        # ensure x has all required features
        missing_cols = set(feature_names) - set(x_new.columns)
        for col in missing_cols:
            x_new[col] = 0

        x_new = x_new[feature_names]

        x_new_scaled = scaler.transform(x_new)

        return model.predict(x_new_scaled)
    
    return predict_price, x.columns, rmse, r2, feature_importance
df_clean = clean_data(df)  # Clean the data first

# Ensure no missing target values
df_clean = df_clean[df_clean['price'].notna()]
X = df_clean[categorical_features + numerical_features]
y = df_clean['price']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


model_pipeline.fit(x_train, y_train)


y_pred = model_pipeline.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

y_pred = model_pipeline.predict(x_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
# train = train_price_model(df_clean)
# advanced print with the train_price_model
model, feature_cols, rmse, r2, importance = train_price_model(df_clean)
print(f"Model RMSE: {rmse:.2f}")
print(f"Model R² Score: {r2:.4f}")
print("\nTop 10 Important Features:")
print(importance.head(10))