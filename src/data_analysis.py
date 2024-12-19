# Importing essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data/renewable_energy_data.csv')  # Replace with your actual data file

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Check for missing values
print("\nMissing Values Summary:")
print(data.isnull().sum())

# Handling missing values (if any)
data.fillna(method='ffill', inplace=True)  # Forward fill as a simple imputation method
print("\nMissing Values After Imputation:")
print(data.isnull().sum())

# Basic statistics about the dataset
print("\nDataset Description:")
print(data.describe())

# Data Analysis

# 1. Energy usage distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Energy_Usage'], bins=30, kde=True, color='blue')
plt.title('Energy Usage Distribution', fontsize=16)
plt.xlabel('Energy Usage (kWh)')
plt.ylabel('Frequency')
plt.savefig('visuals/energy_usage_distribution.png')
plt.show()

# 2. Product category popularity
category_counts = data['Product_Category'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
plt.title('Popularity of Product Categories', fontsize=16)
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.savefig('visuals/product_category_popularity.png')
plt.show()

# 3. Relationship between energy usage and feedback rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Energy_Usage', y='Feedback_Rating', data=data, hue='Product_Category', palette='cool')
plt.title('Energy Usage vs. Feedback Rating', fontsize=16)
plt.xlabel('Energy Usage (kWh)')
plt.ylabel('Feedback Rating')
plt.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('visuals/energy_vs_feedback.png')
plt.show()

# 4. Correlation heatmap
correlation = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig('visuals/correlation_matrix.png')
plt.show()

# 5. Top recommended products based on feedback
top_products = (
    data.groupby('Product_Category')['Feedback_Rating']
    .mean()
    .sort_values(ascending=False)
    .head(5)
)
print("\nTop Recommended Product Categories:")
print(top_products)

# Save results
top_products.to_csv('output/top_recommended_products.csv')

# Summary statistics for energy usage by category
usage_summary = data.groupby('Product_Category')['Energy_Usage'].mean().sort_values()
print("\nAverage Energy Usage by Product Category:")
print(usage_summary)

# Save the usage summary
usage_summary.to_csv('output/energy_usage_summary.csv')

print("\nData analysis completed! Visualizations and outputs have been saved.")
