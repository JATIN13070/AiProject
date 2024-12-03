# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load cleaned data
# Adjust the file path to your cleaned dataset
data = pd.read_csv('../output/cleaned_data.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Check if the target column exists (e.g., 'purchase_made')
if 'purchase_made' not in data.columns:
    raise ValueError("The target column 'purchase_made' is missing from the dataset. Add it to continue.")

# Encode categorical features (e.g., 'product_type', 'region', 'category')
data_encoded = pd.get_dummies(data, columns=['product_type', 'region', 'category'], drop_first=True)

# Define the feature set (X) and the target variable (y)
X = data_encoded.drop('purchase_made', axis=1)  # Replace 'purchase_made' with your actual target column name
y = data_encoded['purchase_made']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
print("\nTraining the Random Forest Classifier...")
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, '../output/recommendation_model.pkl')
print("\nTrained model saved as 'recommendation_model.pkl' in the output folder.")