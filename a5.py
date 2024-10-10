# weather_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the weather dataset
df = pd.read_csv('weather_data.csv')

# Display the first few rows
print("First 5 Rows of the Dataset:")
print(df.head())

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# a. Line plot for Temperature over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x=df.index, y='Temperature (°C)', marker='o')
plt.title('Daily Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# b. Bar chart for Precipitation
plt.figure(figsize=(10, 6))
sns.barplot(x=df.index, y='Precipitation (mm)', data=df, palette='Blues_d')
plt.title('Daily Precipitation')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# c. Scatter plot for Temperature vs. Humidity
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(
    data=df, 
    x='Temperature (°C)', 
    y='Humidity (%)', 
    hue='Precipitation (mm)', 
    palette='viridis', 
    size='Wind Speed (km/h)', 
    sizes=(50, 200),
    legend='full'
)
plt.title('Temperature vs. Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.legend(title='Precipitation (mm) and Wind Speed (km/h)')
plt.tight_layout()
plt.show()

# d. Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# e. Histogram for Wind Speed
plt.figure(figsize=(8, 6))
sns.histplot(df['Wind Speed (km/h)'], bins=5, kde=True, color='green')
plt.title('Wind Speed Distribution')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
