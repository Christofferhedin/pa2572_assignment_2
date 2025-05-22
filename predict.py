import pandas as pd
import re
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
import unicodedata
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,make_scorer,root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from geopy.geocoders import Nominatim
import pickle
from collections import Counter
import time
from nltk.util import ngrams

NORMALIZATION_RULES = [

    (re.compile(r"\b(air conditioning|central air conditioning|portable air conditioning)\b", re.I), "air conditioning"),
    (re.compile(r"\b(heating|central heating|radiant heating|split type ductless system)\b", re.I), "heating"),
    (re.compile(r"\b(hot water kettle|coffee( maker)?|bread maker|rice maker|toaster|blender)\b", re.I), "kitchen appliance"),
    (re.compile(r"\bmicrowave( oven)?\b", re.I), "microwave"),
    (re.compile(r"\bfreezer\b", re.I), "freezer"),
    (re.compile(r"\b(refrigerator|mini fridge|smeg refrigerator|siemens refrigerator|electrolux refrigerator)\b", re.I), "refrigerator"),
    (re.compile(r"\b(dishwasher|oven|stove|electric stove|induction stove|gas stove)\b", re.I), "cooking appliance"),
    (re.compile(r"\bclothing storage\b", re.I), "clothing storage"),


    (re.compile(r"\b(shower gel|body soap|shampoo|conditioner|bidet)\b", re.I), "bathroom essentials"),
    (re.compile(r"\b(bathroom|bathtub|baby bath|hot tub)\b", re.I), "bathroom"),
    (re.compile(r"\b(towels|beach towels|pool towels)\b", re.I), "towels"),
    (re.compile(r"\b(washing machine|washer|dryer|laundromat)\b", re.I), "laundry"),
    (re.compile(r"\b(iron|ironing board|clothes steamer)\b", re.I), "ironing equipment"),
    (re.compile(r"\b(hair dryer|hair straightener|hair curler)\b", re.I), "hair dryer"),

    (re.compile(r"\b(crib|pack n play|travel crib)\b", re.I), "crib / pack n play"),
    (re.compile(r"\bchanging table\b", re.I), "changing table"),
    (re.compile(r"\bhigh chair\b", re.I), "high chair"),
    (re.compile(r"\b(baby safety gates|outlet covers|baby monitor)\b", re.I), "child safety"),
    (re.compile(r"\b(children’s books and toys|books and reading material|board games|arcade games|life size games)\b", re.I), "kids’ entertainment"),


    (re.compile(r"\b(tv|hdtv)\b", re.I), "tv"),
    (re.compile(r"\b(sound system|bluetooth sound system|sonos|audiopro)\b", re.I), "sound system"),
    (re.compile(r"\b(game console|ps4|ping pong table|pool table|movie theater)\b", re.I), "entertainment"),
    (re.compile(r"\b(pocket wifi|ethernet connection|wifi)\b", re.I), "wifi / internet"),
    (re.compile(r"\b(airplay|chromecast|hbo max|apple tv|netflix|hulu|disney\+|amazon prime video)\b", re.I), "streaming services"),
    (re.compile(r"\b(ev charger|electric vehicle charger|charging station)\b", re.I), "ev charger"),


    (re.compile(r"\b(pool|hot tub)\b", re.I), "pool / hot tub"),
    (re.compile(r"\b(barbecue utensils|bbq grill)\b", re.I), "bbq grill"),
    (re.compile(r"\b(outdoor.*|backyard|patio|balcony|garden view|park view|beach view|bay view|canal view|courtyard view|city skyline view|lake view|waterfront|ski-in/ski-out)\b", re.I), "outdoor / view"),
    (re.compile(r"\b(outdoor kitchen|outdoor dining area)\b", re.I), "outdoor kitchen"),


    (re.compile(r"\b(parking|driveway parking|street parking|paid parking|carport)\b", re.I), "parking"),
    (re.compile(r"\b(bike storage|bike rack|bike parking)\b", re.I), "bike storage"),
    (re.compile(r"\b(car rental|car service|car|vehicle)\b", re.I), "car rental"),
    (re.compile(r"\bgarage\b", re.I), "garage"),


    (re.compile(r"\b(lock(box)?|smart lock|keypad)\b", re.I), "secure entry"),
    (re.compile(r"\b(smoke alarm|carbon monoxide alarm|fire extinguisher|first aid kit)\b", re.I), "safety equipment"),
    (re.compile(r"\b(security cameras|security system|security patrol)\b", re.I), "security system"),


    (re.compile(r"\b(elevator|self check-in|host greets you|building staff)\b", re.I), "guest support"),
    (re.compile(r"\b(cleaning|housekeeping)\b", re.I), "cleaning services"),
    (re.compile(r"\b(exercise equipment|gym)\b", re.I), "fitness equipment"),
]
def clean_data(df):

    df_clean = df.copy()
    # convert price $ to numeric
    df_clean["price"] = df_clean["price"].replace("[\$,]", "", regex=True).astype(float)

    # remove extreme outliers in price
    df_clean = df_clean[(df_clean["price"] >= df_clean["price"].quantile(0.05)) & (df_clean["price"] <= df_clean["price"].quantile(0.9))]

    # handle minimum_nights outliers
    df_clean = df_clean[(df_clean["minimum_nights"] >= 1) & (df_clean["minimum_nights"] <= df_clean["minimum_nights"].quantile(0.98))]

    # handle missing values for numerical columns
    numeric_cols = ["bathrooms", "bedrooms", "beds", "accommodates", "minimum_nights"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())


    if "neighbourhood_cleansed" in df_clean.columns:
        df_clean["neighbourhood"] = df_clean["neighbourhood_cleansed"]

    # extract room features
    df_clean["is_entire_home"] = df_clean["room_type"].apply(lambda x:1 if x == "Entire home/apt" else 0)
    df_clean["is_private_room"] = df_clean["room_type"].apply(lambda x:1 if x == "Private room" else 0)
    df_clean["is_hotel_room"] = df_clean["room_type"].apply(lambda x:1 if x == "Hotel room" else 0)
    df_clean["is_shared_room"] = df_clean["room_type"].apply(lambda x:1 if x == "Shared room" else 0)


    if "name" in df_clean.columns:
        df_clean["name"] = df_clean["name"].fillna("")
        df_clean["title_word_count"] = df_clean["name"].fillna("").apply(lambda x:len(str(x).split()))
        df_clean["title_length"] = df_clean["name"].fillna("").apply(len)

    if "description" in df_clean.columns:
        df_clean["description"] = df_clean["description"].fillna("")
        df_clean["description_word_count"] = df_clean["description"].fillna("").apply(lambda x: len(str(x).split()))
        df_clean["description_length"] = df_clean["description"].fillna("").apply(len)

    # add review score features
    review_score_cols = [
        "review_scores_rating", "review_scores_cleanliness",
        "review_scores_location", "review_scores_value"
    ]

    if all(col in df_clean.columns for col in review_score_cols):
        df_clean["avg_review_score"] = df_clean[review_score_cols].mean(axis=1)
        if "review_scores_location" in df_clean.columns and "review_scores_rating" in df_clean.columns:
            df_clean["location_premium"] = df_clean["review_scores_location"] - df_clean["review_scores_rating"]

    for col in review_score_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())


    if "latitude" in df_clean.columns and "longitude" in df_clean.columns:
        stockholm_center = (59.3293, 18.0686)
        df_clean["dist_to_center"] = ((df_clean["latitude"] - stockholm_center[0])**2 + (df_clean["longitude"] - stockholm_center[1])**2)**0.5

    if "availability_365" in df_clean.columns:
        df_clean["availability_rate"] = df_clean["availability_365"] / 365
        df_clean["scarcity"] = 1 - df_clean["availability_rate"]

    df_clean["num_amenities"] = df_clean["amenities"].apply(lambda x: len(ast.literal_eval(x)))


    return df_clean

def normalize_amenity(amenity: str) -> str:
    """Return a canonical label for very common patterns."""
    for pattern, replacement in NORMALIZATION_RULES:
        if pattern.search(amenity):
            return replacement
    return amenity

def parse_clean_and_normalize(col):
    def clean(item):
        # unicode normalize, lowercase, strip punctuation & whitespace
        s = unicodedata.normalize("NFKD", item).lower().strip('-–—•*.,:;!?()[]{}"\" ')
        s = re.sub(r"\s+", " ", s)
        return normalize_amenity(s)

    # parse literal list, clean + normalize, drop empties
    return col.apply(lambda raw: [
        cleaned for cleaned in
        (clean(i) for i in ast.literal_eval(raw) if isinstance(i, str))
        if cleaned and len(cleaned) > 2
    ])

class TopKMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=200):
        self.top_k = top_k
        self.mlb = MultiLabelBinarizer()
        self.top_labels_ = None

    def fit(self, X, y=None):
        # Flatten and count frequencies
        from collections import Counter
        counts = Counter(label for sample in X for label in sample)
        self.top_labels_ = set(label for label, _ in counts.most_common(self.top_k))
        # Filter inputs to top labels only
        filtered_X = [[label for label in sample if label in self.top_labels_] for sample in X]
        return self.mlb.fit(filtered_X)

    def transform(self, X):
        filtered_X = [[label for label in sample if label in self.top_labels_] for sample in X]
        return self.mlb.transform(filtered_X)

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_

# load the data
def load_data():
    df = pd.read_csv("listings.csv.gz", compression="gzip")
    return df
def clip_outliers(X):
    lower = np.quantile(X, 0.01, axis=0)
    upper = np.quantile(X, 0.99, axis=0)
    return np.clip(X, lower, upper)

def create_transformer():
    clean_norm = FunctionTransformer(parse_clean_and_normalize, validate=False)

    categorical_features = ["neighbourhood", "room_type"]
    numerical_features = [
        "latitude", "longitude", "bedrooms", "accommodates", "bathrooms",
        "beds", "minimum_nights","scarcity","dist_to_center", "avg_review_score", "location_premium"
    ]

    amenities_pipeline = Pipeline([
    ("clean_norm", clean_norm),
    ("topk_binarize", TopKMultiLabelBinarizer())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore") )
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ('outlier_clip', FunctionTransformer(clip_outliers, validate=False)),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_pipeline, categorical_features),
    ("amenities", amenities_pipeline, "amenities"),
    ("num", numerical_pipeline, numerical_features),
    ("num_amenities", "passthrough", ["num_amenities"])
])


    return preprocessor

def fit_model(df_clean):
    preprocessor = create_transformer()


    param_grid = [{
    'regressor__n_estimators': [400],
    'regressor__max_depth': [30],
    'regressor__min_samples_split': [2],
    'regressor__min_samples_leaf': [1],
    'regressor__max_features': ['sqrt'],
    'regressor__bootstrap': [False],
    'regressor__random_state': [42],
    'preprocessing__amenities__topk_binarize__top_k': [30],
    }
    # ,{
    #     'regressor': [HistGradientBoostingRegressor()],
    #     'regressor__max_iter': [100, 200],
    #     'regressor__learning_rate': [0.05, 0.1],
    #     'regressor__max_depth': [None, 10],
    #     'preprocessing__amenities__topk_binarize__top_k': [50, 70],
    #     'preprocessing__num__imputer': [SimpleImputer()],
    #     'preprocessing__num__scaler': [StandardScaler()]
    # }
    ]
    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=400,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=42

            ))
    ])
    scoring = {
    'rmse': make_scorer(lambda y_true, y_pred: -root_mean_squared_error(y_true, y_pred), greater_is_better=False),
    'r2': 'r2',
    'mae': make_scorer(mean_absolute_error, greater_is_better=False)
}
    grid_search = GridSearchCV(
    model_pipeline,  # your pipeline
    param_grid=param_grid,
    cv=5,
    scoring=scoring,
    refit="r2",
    n_jobs=-1,
    verbose=1,
    error_score='raise'
    )

    if df_clean is None:
        return

    X = df_clean.drop(columns=["price"])
    y = df_clean["price"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    # model_pipeline.fit(x_train, y_train)
    with open("my_model.pkl", "wb") as f:
        pickle.dump(best_model  , f, protocol=5)

    y_pred = best_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    top_features = pd.DataFrame()
    if hasattr(best_model.named_steps["regressor"], "feature_importances_"):
        feature_names = []
        fitted_preprocessor = best_model.named_steps["preprocessing"]
        # Get feature names from all transformers
        for name, transformer, columns in fitted_preprocessor.transformers_:
            if name == "cat":
                # For categorical columns, get the one-hot encoded feature names
                encoder = transformer.named_steps["onehot"]
                cat_feature_names = encoder.get_feature_names_out(columns)
                feature_names.extend(cat_feature_names)
            elif name == "amenities":
                # For amenities, get the binary feature names
                binarizer = transformer.named_steps["topk_binarize"]
                amenity_feature_names = binarizer.get_feature_names_out()
                feature_names.extend(amenity_feature_names)
            elif name == "num":
                # For numerical columns, use the column names
                feature_names.extend(columns)
            elif name == "num_amenities":
                # For passthrough features, use the column names
                feature_names.extend(columns)

        # Get feature importances
        importances = best_model.named_steps["regressor"].feature_importances_

        # Create a DataFrame of feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names[:len(importances)],  # Match length to importances
            'Importance': importances
        })

        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        # Save top 20 features
        top_features = feature_importance.head(10)


    return rmse, r2, mae, top_features


def load_model():
    with open("my_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


#address format(string): Street Address, City, Country
geolocator = Nominatim(user_agent="airbnb_geocoder")
def get_lat_long_from_address(address):
    """Returns latitude and longitude based on address
    address: Street Address, City, Country """
    try:
        time.sleep(1)
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print("Error:", e)
        return None, None

def get_top_neighbourhoods(df_clean ,n):
    """Returns the most expensive neighborhoods"""
    top_neighbourhoods = df_clean.groupby("neighbourhood_cleansed")["price"].agg(["mean", "count"])
    top_neighbourhoods = top_neighbourhoods[top_neighbourhoods["count"] > 10]
    return top_neighbourhoods.sort_values("mean", ascending=False).head(n)


def get_top_amenities(df_clean, n):
    """get top n amenities"""
    all_amenities = []
    for amenities_list in df_clean["amenities"].apply(ast.literal_eval):
        if isinstance(amenities_list, list):
            # apply normilization to each amenity
            normalized_amenites = [normalize_amenity(unicodedata.normalize("NFKD", item).lower().strip('-–—•*.,:;!?()[]{}"\" '))
                                   for item in amenities_list if isinstance(item, str)]
            all_amenities.extend(normalized_amenites)

    amenity_counts = Counter(all_amenities)
    top_amenities = amenity_counts.most_common(n)


    top_amenities_df = pd.DataFrame(top_amenities, columns=["Amenity", "Count"])

    return top_amenities_df



def get_dynamic_title_tips(df, neighborhood, room_type, ngram_size=2, top_n=10):
    """Returns title keyword tips based on the neighborhood. 
    Gets similiar listings and then returns the most frequently used keywords."""
    # Filter listings based on neighborhood and room type
    similar = df[
        (df["neighbourhood_cleansed"] == neighborhood + "s") &
        (df["room_type"] == room_type) &
        (df["review_scores_rating"] >= 4.5)  # lower threshold for more results
    ]

    if len(similar) < 5:
        return []

    titles = " ".join(similar["name"].fillna("").astype(str)).lower()
    words = re.findall(r"\b[a-z]{3,}\b", titles)

    # remove stopwords
    stop_words = {"and", "the", "with", "for", "from", "this", "that", "have", "has", "och", "med", "på", "för"}
    filtered_words = [w for w in words if w not in stop_words]

    # Generate n-grams
    n_grams = ngrams(filtered_words, ngram_size)
    phrase_counter = Counter([" ".join(gram) for gram in n_grams])

    # Return most common phrases
    return [phrase for phrase, count in phrase_counter.most_common(top_n)]


def predict_price(features_dict):
    """predict the price given the input features
    features_dict : dict
        - neighbourhood: str - Neighborhood name
        - room_type: str - Type of room (Entire home/apt, Private room, etc.)
        - latitude: float - Latitude coordinate
        - longitude: float - Longitude coordinate
        - bedrooms: int - Number of bedrooms
        - accommodates: int - Number of people it accommodates
        - bathrooms: float - Number of bathrooms
        - beds: int - Number of beds
        - minimum_nights: int - Minimum nights stay
        - amenities: list - List of amenities as strings

    """
    try:
        model = load_model()
        features_dict_copy = features_dict.copy()

        # convert amenities to proper format if it's not already a list
        if "amenities" in features_dict_copy:
            if isinstance(features_dict_copy["amenities"], list):
                features_dict_copy["amenities"] = json.dumps(features_dict_copy["amenities"])
            elif not isinstance(features_dict_copy["amenities"], str):
                features_dict_copy["amenities"] = "[]"
        else:
            features_dict_copy["amenities"] = "[]"

        # calc number of amenities
        features_dict_copy["num_amenities"] = len(ast.literal_eval(features_dict_copy["amenities"]) if isinstance(features_dict_copy["amenities"], str) else features_dict_copy["amenities"])


        df_features = pd.DataFrame([features_dict_copy])

        # make sure all required columns are present
        required_cols = ["neighbourhood", "room_type", "latitude", "longitude",
                        "bedrooms", "accommodates", "bathrooms", "beds", "minimum_nights", "amenities", "num_amenities", "dist_to_center", "scarcity", "avg_review_score", "location_premium"]

        df_clean = clean_data(load_data())

        for col in required_cols:
            if col not in df_features.columns:
                if col in ["bedrooms", "accommodates", "bathrooms", "beds", "minimum_nights"]:
                    # fill numeric columns with median values
                    df_features[col] = df_clean[col].median()
                elif col == "amenities":
                    df_features[col] = "[]"
                else:
                    # fill categorical columns with most frequent values
                    df_features[col] = df_clean[col].mode().iloc[0]



        # make prediciton
        predict_price = model.predict(df_features)[0]

        return round(predict_price, 2)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
    
if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    rmse, r2, mae, top_features = fit_model(df_clean)

    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")