# penguin_species_classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# For handling warnings
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set(style="whitegrid")

# 1. Load the Dataset
from palmerpenguins import load_penguins

df = load_penguins()

# Display the first few rows
print("First 5 Rows of the Dataset:")
print(df.head())

# 2. Data Exploration
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Species Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='species', palette='viridis')
plt.title('Species Distribution')
plt.show()

# Pairplot
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 3. Data Preprocessing
# Drop rows with missing 'sex' values
df_clean = df.dropna(subset=['sex'])
print(f"\nDataset shape after dropping missing values: {df_clean.shape}")

# One-Hot Encoding for 'island' and 'sex'
df_encoded = pd.get_dummies(df_clean, columns=['island', 'sex'], drop_first=True)

# Label Encoding for 'species'
le = LabelEncoder()
df_encoded['species'] = le.fit_transform(df_encoded['species'])

# Display the first few rows of the encoded dataset
print("\nFirst 5 Rows of the Encoded Dataset:")
print(df_encoded.head())

# Features and Target
X = df_encoded.drop('species', axis=1)
y = df_encoded['species']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 4. Model Building
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Support Vector Machine
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# 5. Model Evaluation
# Confusion Matrix for Random Forest
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Feature Importance from Random Forest
importances = rf.feature_importances_
feature_names = X.columns
feature_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_imp, y=feature_imp.index, palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
