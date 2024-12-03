from flask import Flask, request, render_template
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('../src/recommendation_model.pkl')

# Load cleaned dataset for product types
data = pd.read_csv('../output/cleaned_data.csv')
product_types = data['product_type'].unique()
regions = data['region'].unique()
categories = data['category'].unique()


# Recommendation function
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


# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html', product_types=product_types, regions=regions, categories=categories)


# Route for handling recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from the form
    user_preferences = {
        'product_type': request.form['product_type'],
        'region': request.form['region'],
        'category': request.form['category']
    }

    # Get recommendations
    recommendations = recommend_products(user_preferences)
    return render_template('recommendations.html', preferences=user_preferences, recommendations=recommendations)


# Run the Flask app
if _name_ == '_main_':
    app.run(debug=True)