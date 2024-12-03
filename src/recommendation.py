# Importing libraries
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load the cleaned dataset (used for product data)
data = pd.read_csv('../output/cleaned_data.csv')

# Load the trained recommendation model
model = joblib.load('../output/recommendation_model.pkl')
print("Model loaded successfully.")


# Define a function for recommending products
def recommend_products(user_input, top_n=5):
    """
    Recommend products based on user input.
    :param user_input: A dictionary of user preferences (e.g., product type, region).
    :param top_n: Number of recommendations to return.
    :return: A list of recommended products.
    """
    # Create a DataFrame for the input
    user_data = pd.DataFrame([user_input])

    # Encode categorical variables the same way as training data
    user_data_encoded = pd.get_dummies(user_data, columns=['product_type', 'region', 'category'], drop_first=True)

    # Align the user_data with the model's feature columns
    user_data_encoded = user_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict probabilities for each product
    probabilities = model.predict_proba(user_data_encoded)[:, 1]  # Probabilities for "purchase_made = 1"

    # Recommend top N products based on probabilities
    product_probabilities = pd.Series(probabilities, index=data['product_type'].unique())
    recommended_products = product_probabilities.nlargest(top_n)

    return recommended_products


# Example user input
user_preferences = {
    "region": "North America",
    "category": "Solar",
    "product_type": "Solar Panel"
}

# Get recommendations
print("\nUser preferences:")
print(user_preferences)

recommendations = recommend_products(user_preferences)
print("\nRecommended products:")
print(recommendations)