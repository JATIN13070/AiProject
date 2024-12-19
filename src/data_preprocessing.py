# Importing essential libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('data/renewable_energy_data.csv')  # Replace with your actual file name
print("Dataset loaded successfully!")

# Display the first few rows
print("\nPreview of the raw dataset:")
print(data.head())

# Check for missing values
print("\nSummary of missing values:")
print(data.isnull().sum())

# Step 1: Handling Missing Values
# Option 1: Fill missing numerical values with the mean
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    data[col].fillna(data[col].mean(), inplace=True)

# Option 2: Fill missing categorical values with the mode
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

print("\nMissing values after imputation:")
print(data.isnull().sum())

# Step 2: Encoding Categorical Data
# Using Label Encoding for simplicity
encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

print("\nSample of encoded dataset:")
print(data.head())

# Step 3: Feature Scaling
# Scaling numerical features to a standard range
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

print("\nSample of scaled dataset:")
print(data.head())

# Step 4: Removing Duplicates
# Check for duplicate rows
duplicate_count = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

# Remove duplicates if any
data = data.drop_duplicates()
print(f"Dataset shape after removing duplicates: {data.shape}")

# Step 5: Saving the Preprocessed Data
data.to_csv('data/preprocessed_data.csv', index=False)
print("\nPreprocessed dataset saved to 'data/preprocessed_data.csv'.")
