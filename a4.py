import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create Synthetic Dataset
data = {
    'Area_sqft': [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750],
    'Bedrooms': [1, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    'Bathrooms': [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    'Location': ['Downtown', 'Downtown', 'Suburb', 'Suburb', 'Suburb', 
                 'Uptown', 'Uptown', 'Uptown', 'City Center', 'City Center'],
    'Rent': [1500, 2000, 2200, 2700, 3000, 3500, 4000, 4500, 5000, 5500]
}

df = pd.DataFrame(data)
print("Synthetic Housing Data:")
print(df)

# 2. Data Preprocessing
df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)
print("\nData after One-Hot Encoding:")
print(df_encoded)

# 3. Feature Selection
X = df_encoded.drop('Rent', axis=1)
y = df_encoded['Rent']

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Features:")
print(X_train)

print("\nTesting Features:")
print(X_test)

# 5. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Prediction and Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 ): {r2:.2f}")
