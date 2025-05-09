import pandas as pd
import re
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder, StandardScaler,MinMaxScaler,MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import unicodedata
import ast
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from geopy.geocoders import Nominatim
import pickle
from collections import Counter
import time


NORMALIZATION_RULES = [

    # Climate Control
(re.compile(r"\b(air conditioning|central air conditioning|portable air conditioning)\b", re.I), "air conditioning"),
(re.compile(r"\b(heating|central heating|radiant heating)\b", re.I), "heating"),
(re.compile(r"\bsplit type ductless system\b", re.I), "ductless system"),

# Kitchen Appliances (Split finer)
(re.compile(r"\b(hot water kettle|electric kettle)\b", re.I), "kettle"),
(re.compile(r"\bcoffee( maker)?\b", re.I), "coffee maker"),
(re.compile(r"\bbread maker\b", re.I), "bread maker"),
(re.compile(r"\brice maker\b", re.I), "rice maker"),
(re.compile(r"\btoaster\b", re.I), "toaster"),
(re.compile(r"\bblender\b", re.I), "blender"),
(re.compile(r"\bmicrowave( oven)?\b", re.I), "microwave"),
(re.compile(r"\bfreezer\b", re.I), "freezer"),
(re.compile(r"\b(refrigerator|mini fridge|smeg refrigerator|siemens refrigerator|electrolux refrigerator)\b", re.I), "refrigerator"),
(re.compile(r"\b(dishwasher)\b", re.I), "dishwasher"),
(re.compile(r"\b(oven|stove|electric stove|induction stove|gas stove)\b", re.I), "stove/oven"),

# Storage and Laundry
(re.compile(r"\bclothing storage\b", re.I), "clothing storage"),
(re.compile(r"\b(washing machine|washer)\b", re.I), "washing machine"),
(re.compile(r"\b(dryer)\b", re.I), "dryer"),
(re.compile(r"\blaundromat\b", re.I), "laundromat"),

# Bathroom
(re.compile(r"\b(shower gel|body soap|shampoo|conditioner)\b", re.I), "bathroom toiletries"),
(re.compile(r"\bbidet\b", re.I), "bidet"),
(re.compile(r"\b(bathtub|baby bath|hot tub)\b", re.I), "bathtub / hot tub"),
(re.compile(r"\bbathroom\b", re.I), "bathroom"),
(re.compile(r"\b(towels|beach towels|pool towels)\b", re.I), "towels"),

# Cleaning & Grooming
(re.compile(r"\b(iron|ironing board|clothes steamer)\b", re.I), "ironing equipment"),
(re.compile(r"\b(hair dryer|hair straightener|hair curler)\b", re.I), "hair appliances"),

# Baby/Kids
(re.compile(r"\b(crib|pack n play|travel crib)\b", re.I), "crib"),
(re.compile(r"\bchanging table\b", re.I), "changing table"),
(re.compile(r"\bhigh chair\b", re.I), "high chair"),
(re.compile(r"\b(baby safety gates|outlet covers|baby monitor)\b", re.I), "baby safety"),
(re.compile(r"\b(children’s books and toys|books and reading material|board games|arcade games|life size games)\b", re.I), "children entertainment"),

# Entertainment
(re.compile(r"\b(tv|hdtv)\b", re.I), "tv"),
(re.compile(r"\b(sound system|bluetooth sound system|sonos|audiopro)\b", re.I), "sound system"),
(re.compile(r"\b(game console|ps4|ping pong table|pool table|movie theater)\b", re.I), "game or media equipment"),
(re.compile(r"\b(pocket wifi|ethernet connection|wifi)\b", re.I), "internet"),
(re.compile(r"\b(airplay|chromecast|hbo max|apple tv|netflix|hulu|disney\+|amazon prime video)\b", re.I), "streaming"),

# Outdoors
(re.compile(r"\b(pool)\b", re.I), "pool"),
(re.compile(r"\b(hot tub)\b", re.I), "hot tub"),
(re.compile(r"\b(barbecue utensils|bbq grill)\b", re.I), "bbq"),
(re.compile(r"\b(outdoor kitchen|outdoor dining area)\b", re.I), "outdoor kitchen"),
(re.compile(r"\b(backyard|patio|balcony|garden view|park view|beach view|bay view|canal view|courtyard view|city skyline view|lake view|waterfront|ski-in/ski-out)\b", re.I), "outdoor view"),

# Transportation
(re.compile(r"\b(parking|driveway parking|street parking|paid parking|carport)\b", re.I), "parking"),
(re.compile(r"\b(bike storage|bike rack|bike parking)\b", re.I), "bike storage"),
(re.compile(r"\b(car rental|car service)\b", re.I), "car rental"),
(re.compile(r"\bgarage\b", re.I), "garage"),

# Safety & Access
(re.compile(r"\b(lockbox|smart lock|keypad)\b", re.I), "keyless entry"),
(re.compile(r"\b(smoke alarm|carbon monoxide alarm|fire extinguisher|first aid kit)\b", re.I), "safety equipment"),
(re.compile(r"\b(security cameras|security system|security patrol)\b", re.I), "security"),

# Services and Extras
(re.compile(r"\b(elevator|self check-in|host greets you|building staff)\b", re.I), "guest access / support"),
(re.compile(r"\b(cleaning service|housekeeping)\b", re.I), "cleaning service"),
(re.compile(r"\b(exercise equipment|gym)\b", re.I), "fitness equipment"),
(re.compile(r"\bev charger\b", re.I), "ev charger"),

]
def clean_data(df):

    df_clean = df.copy()
    # convert price $ to numeric
    df_clean["price"] = df_clean["price"].replace("[\$,]", "", regex=True).astype(float)


    print(df_clean["price"].describe())

    # remove extreme outliers in price
    q1_price = df_clean["price"].quantile(0.05)
    q3_price = df_clean["price"].quantile(0.95)
    iqr_price = q3_price - q1_price

    df_clean = df_clean[(df_clean["price"] >= max(0, q1_price - 1.5 * iqr_price)) & (df_clean["price"] <= q3_price + 1.5 * iqr_price)]
    print("\nPrice statistics after outlier removal:")
    print(df_clean["price"].describe())

    # handle minimum_nights outliers
    q1_night = df_clean["minimum_nights"].quantile(0.01)
    q3_night = df_clean["minimum_nights"].quantile(0.99)
    iqr_night = q3_night - q1_night

    df_clean = df_clean[(df_clean["minimum_nights"] >= max(1, q1_night - 1.5 * iqr_night)) & (df_clean["minimum_nights"] <= q3_night + 1.5 * iqr_night)]

    print("\nMinimum nights statistics after outlier removal:")
    print(df_clean["minimum_nights"].describe())

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

    # create title and description features
    # if "name" in df_clean.columns:
    #     df_clean["name"] = df_clean["name"].fillna("")
    #     df_clean["title_word_count"] = df_clean["name"].fillna("").apply(lambda x:len(str(x).split()))
    #     df_clean["title_length"] = df_clean["name"].fillna("").apply(len)

    # if "description" in df_clean.columns:
    #     df_clean["description"] = df_clean["description"].fillna("")
    #     df_clean["description_word_count"] = df_clean["description"].fillna("").apply(lambda x: len(str(x).split()))
    #     df_clean["description_length"] = df_clean["description"].fillna("").apply(len)

    # add review score features
    # review_score_cols = [
    #     "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    #     "review_scores_checkin", "review_scores_communication", "review_scores_location",
    #     "review_scores_value"
    # ]

    # for col in review_score_cols:
    #     if col in df_clean.columns:
    #         df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # # superhost feature
    # if "host_is_superhost" in df_clean.columns:
    #     df_clean["host_is_superhost_num"] = df_clean["host_is_superhost"].apply(lambda x: 1 if x == "t" else 0)

    # # check if instant bookable
    # if "instant_bookable" in df_clean.columns:
    #     df_clean["instant_bookable_num"] = df_clean["instant_bookable"].apply(lambda x: 1 if x == "t" else 0)

    # # add review count and frequency features
    # if "number_of_reviews" in df_clean.columns:
    #     df_clean["number_of_reviews"] = df_clean["number_of_reviews"].fillna(0)
    #     df_clean["has_reviews"] = df_clean["number_of_reviews"].apply(lambda x: 1 if x > 0 else 0)

    # if "reviews_per_month" in df_clean.columns:
    #     df_clean["reviews_per_monthn"] = df_clean["reviews_per_month"].fillna(0)

    df_clean["num_amenities"] = df_clean["amenities"].apply(lambda x: len(ast.literal_eval(x)))

    print(df_clean.shape)

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
mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')
iterative_imputer = IterativeImputer(random_state=0)
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
def create_transformer():
    clean_norm = FunctionTransformer(parse_clean_and_normalize, validate=False)

    categorical_features = ["neighbourhood", "room_type"]
    numerical_features = ["latitude", "longitude", "bedrooms", "accommodates", "bathrooms", "beds", "minimum_nights"]

    amenities_pipeline = Pipeline([
    ("clean_norm", clean_norm),

    ("topk_binarize", TopKMultiLabelBinarizer(top_k=50))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", IterativeImputer(random_state=0)),
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
    'regressor__n_estimators': [100],
    'regressor__max_depth': [None],
    'regressor__min_samples_split': [2],
    'preprocessing__amenities__topk_binarize__top_k': [200],
    'preprocessing__num__imputer': [SimpleImputer()],
    'preprocessing__num__scaler': [minmax_scaler]
    },{
        'regressor': [HistGradientBoostingRegressor()],
        'regressor__max_iter': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [None, 10],
        'preprocessing__amenities__topk_binarize__top_k': [50, 70],
        'preprocessing__num__imputer': [SimpleImputer()],
        'preprocessing__num__scaler': [StandardScaler()]
    }]
    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    grid_search = GridSearchCV(
    model_pipeline,  # your pipeline
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_root_mean_squared_error',  # or 'r2', etc.
    n_jobs=-1,  # use all CPU cores
    verbose=1
)
    if df_clean is None:
        return

    X = df_clean.drop(columns=["price"])
    y = df_clean["price"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    # model_pipeline.fit(x_train, y_train)
    # with open("my_model.pkl", "wb") as f:
    #     pickle.dump(best_model  , f, protocol=5)

    y_pred = best_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)



    # Evaluate
    print("Best parameters found: ", grid_search.best_params_)
    return rmse, r2


def load_model():
    with open("my_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model




df = load_data()
df_clean = clean_data(df)
rmse, r2 = fit_model(df_clean)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
# train = train_price_model(df_clean)
# advanced print with the train_price_model

# # To get cleaned amenity names:
# feature_names = preprocessor.named_transformers_["amenities"].named_steps["topk_binarize"].get_feature_names_out()
# #print("Feature names:", feature_names)
# print(X_transformed)


#address format(string): Street Address, Neighborhood, City, Country
geolocator = Nominatim(user_agent="airbnb_geocoder")
def get_lat_long_from_address(address):
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

    top_neighbourhoods = df_clean.groupby("neighbourhood_cleansed")["price"].agg(["mean", "count"])
    top_neighbourhoods = top_neighbourhoods[top_neighbourhoods["count"] > 10]
    return top_neighbourhoods.sort_values("mean", ascending=False).head(n)

print(get_top_neighbourhoods(df_clean, 5))

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

    # convert to dataframe
    top_amenities_df = pd.DataFrame(top_amenities, columns=["Amenity", "Count"])

    return top_amenities_df

print(get_top_amenities(df_clean, 10))


def get_dynamic_title_tips(df, neighborhood, room_type):
    # Filter listings based on neighborhood and room type
    similar = df[
        (df["neighbourhood_cleansed"] == neighborhood + "s") &
        (df["room_type"] == room_type) &
        (df["review_scores_rating"] >= 4.8)
    ]

    print(similar)
    if len(similar) < 5:
        return []

    titles = " ".join(similar["name"].fillna("").astype(str)).lower()
    words = re.findall(r"\b[a-z]{3,}\b", titles)

    stop_words = {"and", "the", "with", "for", "from", "this", "that", "have", "has", "och", "med", "på", "för"}
    filtered = [w for w in words if w not in stop_words]

    most_common = [w for w, count in Counter(filtered).most_common(5)]
    return most_common


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
                        "bedrooms", "accommodates", "bathrooms", "beds", "minimum_nights", "amenities", "num_amenities"]

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

# sample_features = {
#     "neighbourhood": "Södermalms",
#         "room_type": "Private room",
#         "latitude": 59.31389,
#         "longitude": 18.06087,
#         "bedrooms": 1,
#         "accommodates": 2,
#         "bathrooms": 1.0,
#         "beds": 1,
#         "amenities": ["Hair dryer", "Hangers", "Long term stays allowed", "Host greets you", "Bathtub",
#                        "Luggage dropoff allowed", "Iron", "Essentials", "Free washer \u2013 In building",
#                         "Elevator", "Free dryer \u2013 In building", "Courtyard view", "Smoke alarm", "TV",
#                         "Garden view", "Dishes and silverware", "Shared backyard \u2013 Not fully fenced",
#                         "Outdoor playground", "Heating", "Hot water", "Shampoo", "Bed linens",
#                         "Extra pillows and blankets", "Lock on bedroom door", "Fast wifi \u2013 399 Mbps",
#                         "Park view", "Refrigerator", "Microwave", "Coffee maker"],
#         "num_amenities": len(sample_features["amenities"])
# }
# predicted_price = predict_price(sample_features)
# print(f"Predicted price: ${predicted_price} per night")