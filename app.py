import streamlit as st
from otherpy import predict_price, load_model, load_data, clean_data, get_top_amenities, get_lat_long_from_address, normalize_amenity, get_dynamic_title_tips, fit_model,parse_clean_and_normalize,TopKMultiLabelBinarizer,clip_outliers
from collections import Counter
import re
import ast
import json
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim


@st.cache_resource
def load_cached_model():
    return load_model()

@st.cache_data
def load_cached_data():
    df = load_data()
    df_clean = clean_data(df)
    return df, df_clean

# Function to get neighborhood from coordinates
@st.cache_data
def get_neighborhood_from_coords(lat, lng):
    try:
        geolocator = Nominatim(user_agent="airbnb_estimator")
        location = geolocator.reverse((lat, lng), exactly_one=True)
        address = location.raw["address"]
        # Try to get neighborhood or district
        for key in ["neighbourhood", "suburb", "district", "quarter"]:
            if key in address:
                return address[key]
        return address.get("city_district", "Unknown neighborhood")
    except:
        return "Unknown neighborhood"

# get the top amenities to show in the ui
@st.cache_data
def get_top_amenities_for_ui(n=30):
    df, df_clean = load_cached_data()
    top_amenities = get_top_amenities(df_clean, n)
    return top_amenities["Amenity"].tolist()

# get insight for standing out
@st.cache_data
def get_competetive_insights(df_clean):
    insights = {}

    # Top rated listings features
    top_rated = df_clean[df_clean["review_scores_rating"] >= 4.9]
    if len(top_rated) > 10:
        # Most common amenities in top rated listings
        all_amenities = []
        for amenities_list in top_rated["amenities"].apply(ast.literal_eval):
            if isinstance(amenities_list, list):
                all_amenities.extend([normalize_amenity(item.lower().strip()) for item in amenities_list if isinstance(item, str)])
        top_amenities_counter = Counter(all_amenities)
        insights["top_amenities"] = [item for item, count in top_amenities_counter.most_common(10)]

        # average length of title and description
        if "name" in top_rated.columns:
            insights["avg_title_length"] = round(top_rated["name"].fillna("").apply(len).mean())

        if "description" in top_rated.columns:
            insights["avg_description_length"] = round(top_rated["description"].fillna("").apply(len).mean())

        # common words in titels
        if "name" in top_rated.columns:
            all_title_text = " ".join(top_rated["name"].fillna("").astype(str)).lower()

            # Tokenize
            words = re.findall(r"\b[a-z]{3,}\b", all_title_text)

            # Define stop words
            stop_words = {"and", "the", "with", "for", "from", "this", "that", "have", "has",
                        "och", "med", "på", "för", "ett", "en", "det", "av"}

            # Create bigrams
            bigrams = [
                f"{words[i]} {words[i + 1]}"
                for i in range(len(words) - 1)
                if words[i] not in stop_words and words[i + 1] not in stop_words
            ]

            # Count and assign
            bigram_counter = Counter(bigrams)
            insights["top_title_words"] = [phrase for phrase, count in bigram_counter.most_common(10)]

    return insights

@st.cache_data
def get_model_stats(df_clean):
    """Returns Rmse score, r^2 score, mae score and top fe"""
    try:
        # load stored model metrics
        rmse, r2, mae, top_features = fit_model(df_clean)

        return {
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "top_features": top_features
        }
    except Exception as e:
        st.error(f"Error {e}")
        return None

def create_prediction_vs_actual_plot():
    df, df_clean = load_cached_data()
    model = load_cached_model()

    # subset of data for visualization
    sample_size = min(1000, len(df_clean))
    sample_df = df_clean.sample(n=sample_size, random_state=42)

    x = sample_df.drop(columns=["price"])
    y_actual = sample_df["price"]

    try:
        y_pred = model.predict(x)
        
        results_df = pd.DataFrame({
            "Actual": y_actual,
            "Predicted": y_pred
        })

        # create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(results_df["Actual"], results_df["Predicted"], alpha=0.5)

        max_val = max(results_df["Actual"].max(), results_df["Predicted"].max())
        min_val = min(results_df["Actual"].min(), results_df["Predicted"].min())
        ax.plot([min_val, max_val], [min_val, max_val], "r--")

        ax.set_xlabel("Actual price (SEK)")
        ax.set_ylabel("Predicted price (SEK)")
        ax.set_title("Model predictions vs actual price")

        return fig
    except Exception as e:
        st.error(f"Error creating prediction plot: {e}")
        return None

# streamlit app
def main():
    st.title("Stockholm Airbnb Price Estimator")
    st.write("Find out how much your property could earn on Airbnb and get tips to make your listing stand out!")

    # load the data and model
    try:
        df, df_clean = load_cached_data()
        model = load_cached_model()
        st.success("Data and model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data or model: {e}")
        return

    # create tabs
    tab1, tab2, tab3 = st.tabs(["Price Estimator", "Market Insights", "Model Statistics"])

    with tab1:
        st.header("Estimate your rental price")
        st.write("Enter details about your property to get an estimated nightly price.")

        # property location
        st.subheader("Location")
        col1, col2 = st.columns(2)
        with col1:
            address = st.text_input("Address", "Götgatan 14, Stockholm, Sweden")
            # neighborhood = st.text_input("Neighborhood")
            use_address = st.checkbox("Use address for location", value=True)

        with col2:
            if use_address:
                lat, lng = get_lat_long_from_address(address)
                if lat and lng:
                    st.success(f"Found location: {lat:.4f}, {lng:.4f}")
                    neighborhood = get_neighborhood_from_coords(lat, lng)
                    #neighborhood = address.split(",")[1].strip()
                    st.write(f"Neighborhood: {neighborhood}")
                else:
                    st.error("Could not geocode address. Please enter coordinates manually.")
                    lat = None
                    lng = None
            if not use_address or lat is None:
                lat = st.number_input("Latitude", value=59.31389, format="%.5f")
                lng = st.number_input("Longitude", value=18.06087, format="%.5f")
                neighborhood = get_neighborhood_from_coords(lat, lng)
                st.write(f"Neighborhood: {neighborhood}")

        # property details
        st.subheader("Property details")
        col1, col2 = st.columns(2)

        with col1:
            room_type = st.selectbox(
                "Room Type",
                ["Entire home/apt", "Private room", "Hotel room", "Shared room"],
                index=0
            )
            bedrooms = st.number_input("Number of bedrooms", min_value=0, value=1)
            beds = st.number_input("Number of beds", min_value=1, value=1)

        with col2:
            accommodates = st.number_input("Maximum guests", min_value=1, value=2)
            bathrooms = st.number_input("Number of bathrooms", min_value=0.5, value=1.0, step=0.5)
            minimum_nights = st.number_input("Minimum nights stay", min_value=1, value=2)

        # amenities selection
        st.subheader("Amenities")
        st.write("Select amenities available in your property:")

        # get top amenities and split into columns
        top_amenities = get_top_amenities_for_ui(30)

        cols = st.columns(3)
        selected_amenities = []

        for i, amenity in enumerate(top_amenities):
            col_idx = i % 3
            if cols[col_idx].checkbox(amenity, value=(amenity in ["wifi / internet", "heating", "tv"])):
                selected_amenities.append(amenity)

        # other amenities
        other_amenity = st.text_input("Add other amenity (Enter to add)")
        if other_amenity and other_amenity not in selected_amenities:
            selected_amenities.append(other_amenity)
            st.write(f"Added {other_amenity} to amenities list")

        st.write(f"Total selected amenities: {len(selected_amenities)}")

        # estimate button
        if st.button("Estimate price"):
            features = {
                "neighbourhood": neighborhood,
                "room_type": room_type,
                "latitude": lat,
                "longitude": lng,
                "bedrooms": bedrooms,
                "accommodates": accommodates,
                "bathrooms": bathrooms,
                "beds": beds,
                "amenities": selected_amenities
            }

            print("\n===== FEATURES USED FOR PRICE ESTIMATION =====")
            print(json.dumps(features, indent=2))
            print("=============================================\n")

            estimated_price = predict_price(features)

            if estimated_price:
                st.success(f"Estimated nightly price: {estimated_price:.2f} SEK")

                # show price range
                st.write("Suggested price range:")
                low_price = max(1, estimated_price * 0.85)
                high_price = estimated_price * 1.15

                col1, col2, col3 = st.columns(3)
                col1.metric("Lower end", f"{low_price:.0f} SEK")
                col2.metric("Estimated", f"{estimated_price:.0f} SEK")
                col3.metric("Higher end", f"{high_price:.0f} SEK")

                # recomendations based on amenities
                insights = get_competetive_insights(df_clean)
                missing_top_amenities = [item for item in insights.get("top_amenities", []) if item not in selected_amenities]
                title_keywords = get_dynamic_title_tips(df_clean, neighborhood, room_type)

                if title_keywords:
                    st.subheader("Suggested title keywords:")
                    st.info(f"{', '.join(title_keywords)}")
                else:
                    generic_title_keywords = get_competetive_insights(df_clean)["top_title_words"]
                    st.subheader("Suggested title keywords:")
                    st.info(f"{', '.join(generic_title_keywords)}")



                if missing_top_amenities:
                    st.info(f"""
                    **Stand out from competition by adding these popular amenities:**
                    {", ".join(missing_top_amenities[:5])}
                    """)
                else:
                    st.error("Couldn't estimate price. Please check your inputs")

    with tab2:
        st.header("Market Insights")

        insights = get_competetive_insights(df_clean)

        st.subheader("Top Neighborhoods by Average Price")

        top_neighborhoods = df_clean.groupby("neighbourhood")["price"].agg(["mean", "count"])
        top_neighborhoods = top_neighborhoods[top_neighborhoods["count"] > 10]
        top_neighborhoods = top_neighborhoods.sort_values("mean", ascending=False).head(10)

        # format for display
        top_neighborhoods["mean"] = top_neighborhoods["mean"].map("{:.0f} SEK".format)
        st.dataframe(top_neighborhoods.reset_index().rename(columns={"mean": "Avg Price", "count": "# Listings", "neighbourhood": "Neighborhood"}))

        # top amenities in high rated listings
        st.subheader("Most popular amenities in top rated listings")
        if "top_amenities" in insights:
            for amenity in insights["top_amenities"][:10]:
                st.write(f"- {amenity}")

        # Title and description tips
        st.subheader("Tips for lisiting title and description")
        if "top_title_words" in insights:
            st.write("Popular words in top rated listing titels:")
            st.write(", ".join(insights["top_title_words"]))

        if "avg_title_length" in insights:
            st.write(f"Ideal title length: {insights['avg_title_length']} characters")

        if "avg_description_length" in insights:
            st.write(f"Ideal description length {insights['avg_description_length']} characters")

        # room type analys
        st.subheader("average price by room type")
        room_type_prices = df_clean.groupby("room_type")["price"].mean().sort_values(ascending=False)

        # simple bar chart
        st.bar_chart(room_type_prices)

    with tab3:
        st.header("Model Statistics")
        st.write("Examine how well our price prediction model performs and which features are most important.")

        # get model stats
        model_stats = get_model_stats(df_clean)

        # display metrics
        if model_stats is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="RMSE (Root Mean Square Error)",
                    value=f"{model_stats['rmse']:.2f} SEK",
                    help="Average prediction error. Lower values indicate better accuracy."
                )

            with col2:
                st.metric(
                    label="R² Score",
                    value=f"{model_stats['r2']:.4f}",
                    help="Proportion of variance explained by the model. Higher values (closer to 1) indicate better fit."
                )
            with col3:
                st.metric(
                    label="MAE (Mean Absolute Error)",
                    value=f"{model_stats['mae']:.2f} SEK",
                    help="Average absolute error. Lower values indicate better accuracy."
                )

            if model_stats and "top_features" in model_stats:
                top_features_df = model_stats["top_features"]

                st.subheader("Top features used in the model")
                top_features_df.insert(0, "Number", range(1, len(top_features_df) + 1))

                st.table(top_features_df[["Number", "Feature", "Importance"]])
            else:
                st.warning("Top features data not avilable")

        st.subheader("Prediction accuracy")
        st.write("This chart shows the models predictions compared to actual prices:")
        pred_fig = create_prediction_vs_actual_plot()
        
        if pred_fig:
            st.pyplot(pred_fig)
            st.write("""
            Points above the line are overestimated, points below are underestimated.
            The closer points are to the line, the more accurate the model.
            """)


if __name__ == "__main__":
    main()